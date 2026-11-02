import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
    _env_path = Path(__file__).resolve().parents[1] / '.env'
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.logging_middleware import AuditLoggingMiddleware
from api.model_loader import load_state, model_version
from api.routers import audit as audit_router
from api.routers import experiments as experiments_router
from api.routers import workflow_studio as workflow_studio_router
from api.schemas import PredictRequest, PredictResponse
from api.security import require_predict_token, safe_identifier
from credit_risk_model.config import get_settings
from credit_risk_model.features.scorecard import probability_to_score
from credit_risk_model.governance.audit import AuditStore, AuditWriteError, PredictionEvent
from credit_risk_model.models.contract import FeatureContractError

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    state = load_state()
    logger.info("startup environment=%s model_loaded=%s reason=%s",
                settings.environment, state.loaded, state.reason)
    yield

app = FastAPI(title="Credit Risk Model Validation Platform", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuditLoggingMiddleware)
app.include_router(audit_router.router)
app.include_router(experiments_router.router)
app.include_router(workflow_studio_router.router)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error(request: Request, status: int, detail, headers=None) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse({"detail": detail, "request_id": rid}, status_code=status,
                        headers={**(headers or {}), **({"x-request-id": rid} if rid else {})})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # Only location, message and type: never the submitted input or context.
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
              for e in exc.errors()]
    return _error(request, 422, errors)


@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException):
    return _error(request, exc.status_code, exc.detail, getattr(exc, "headers", None))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.exception("unhandled error request_id=%s", _request_id(request))
    return _error(request, 500, "Internal server error")


@app.get("/health")
def health():
    state = load_state()
    return {"status": "ok", "model_loaded": state.loaded, "reason": state.reason,
            "model_name": get_settings().champion_name, "model_version": model_version(),
            "environment": get_settings().environment}


@app.get("/model-card")
def model_card():
    state = load_state()
    entry, model = state.entry, state.model
    meta = getattr(model, "metadata", {}) if model else {}
    card = {
        "name": get_settings().champion_name,
        "version": model_version(),
        # Fixed disclaimer: this is a demonstration platform, whatever the inventory says.
        "status": "not-approved-for-production",
        "model_loaded": state.loaded,
        "serving_reason": state.reason,
        "training": {k: meta.get(k) for k in (
            "algorithm", "class_weight", "random_state", "training_rows",
            "training_default_rate", "training_data_sha256", "holdout_auroc",
            "library_versions", "trained_at_utc")} if meta else None,
        "features": meta.get("features") if meta else None,
    }
    if entry:
        card["inventory"] = {
            "status": entry.status.value, "model_type": entry.model_type, "owner": entry.owner,
            "purpose": entry.purpose, "intended_use": entry.intended_use,
            "risk_tier": entry.risk_tier.value if entry.risk_tier else None,
            "limitations": entry.limitations, "use_restrictions": entry.use_restrictions,
            "approved_by": entry.approved_by, "approval_date": entry.approval_date,
            "next_review_date": entry.next_review_date,
            "last_validation_date": entry.last_validation_date,
            "validation_report": entry.validation_report,
            "open_findings": len(entry.open_findings()),
        }
    return card


@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(require_predict_token)])
def predict(payload: PredictRequest, request: Request):
    start = time.perf_counter()
    if payload.request_id and not request.headers.get("x-request-id"):
        request.state.request_id = payload.request_id
    request_id = request.state.request_id
    state = load_state()
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model unavailable: " + state.reason)
    try:
        if hasattr(state.model, "specs"):
            pd_value = float(state.model.predict_proba(payload.features)[0])
        else:
            import pandas as pd

            from credit_risk_model.models import predict_pd

            df = pd.DataFrame([payload.features])
            pd_value = float(predict_pd(state.model, df)[0])
    except FeatureContractError as exc:
        raise HTTPException(status_code=422, detail={"feature_errors": exc.issues}) from exc
    score = int(probability_to_score(pd_value))
    settings = get_settings()
    event = PredictionEvent(
        request_id=request_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        model_name=settings.champion_name, model_version=model_version(),
        model_status=state.entry.status.value if state.entry else "unregistered",
        probability_of_default=pd_value, score=score,
        input_feature_names=sorted(payload.features),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        environment=settings.environment,
        api_user_or_client_id=safe_identifier(request.headers.get("x-client-id")))
    try:
        AuditStore().write_prediction(event)
    except AuditWriteError as exc:
        logger.error("audit write failed request_id=%s", request_id)
        raise HTTPException(status_code=503,
                            detail="Audit trail unavailable; prediction withheld") from exc
    return PredictResponse(request_id=request_id, probability_of_default=pd_value, score=score,
                           model_name=settings.champion_name, model_version=model_version())
