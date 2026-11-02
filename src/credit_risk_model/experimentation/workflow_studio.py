"""Workflow Studio Runner: executes the 8 stages and 14 agents for the interactive workflow planner."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from credit_risk_model.agents.specialists import (
    ApiEngineerAgent,
    ChallengerAgent,
    DataQualityAgent,
    DevOpsAgent,
    ExperimentTrackingAgent,
    ExplainabilityAgent,
    FeatureEngineeringAgent,
    GovernanceAgent,
    ModelingAgent,
    ModelRiskReviewerAgent,
    MonitoringAgent,
    PipelineEngineer,
    QaTestAgent,
    ReportHandoffAgent,
    TargetLeakageAgent,
    ValidationAgent,
)
from credit_risk_model.config import get_settings
from credit_risk_model.data.synthetic import generate as generate_synthetic_dataset

logger = logging.getLogger("workflow_studio")


class WorkflowRunRequest(BaseModel):
    dataset: str = "synthetic"  # synthetic | lendingclub | custom
    target: str = "loan_status"
    champion: str = "logistic_regression"
    challenger: str = "hist_gradient_boosting"
    oot_pct: float = 20.0
    seed: int = 42
    stages: list[str] = Field(default_factory=lambda: [
        "data_intake", "feature_engineering", "train_champion",
        "train_challenger", "validate", "governance", "api_tests", "monitor_deploy"
    ])
    agents: list[str] = Field(default_factory=lambda: [
        "data_quality", "target_leakage", "feature_engineering",
        "modeling", "challenger", "validation", "model_risk_review",
        "explainability", "governance", "api_engineer", "qa_test",
        "monitoring", "devops", "report_handoff"
    ])


class WorkflowRunResponse(BaseModel):
    run_id: str
    status: str
    events_url: str
    artifacts_url: str


class WorkflowEvent(BaseModel):
    time: str
    type: str  # "info", "ok", "warn", "error"
    msg: str
    stage: str | None = None
    stage_num: int | None = None
    agent: str | None = None


STAGE_METADATA = {
    "data_intake": {"num": 1, "name": "Data intake & profiling", "agents": ["Data Quality Agent", "Target & Leakage Agent"]},
    "feature_engineering": {"num": 2, "name": "Data preparation & feature engineering", "agents": ["Feature Engineering Agent", "Pipeline Engineer"]},
    "train_champion": {"num": 3, "name": "Champion model training", "agents": ["Modeling Agent", "Experiment Tracking Agent"]},
    "train_challenger": {"num": 4, "name": "Challenger model training", "agents": ["Challenger Agent", "Experiment Tracking Agent"]},
    "validate": {"num": 5, "name": "Independent validation & stress tests", "agents": ["Validation Agent", "Model Risk Reviewer"]},
    "governance": {"num": 6, "name": "Explainability & governance pack", "agents": ["Explainability Agent", "Governance Agent"]},
    "api_tests": {"num": 7, "name": "API packaging & tests", "agents": ["API Engineer Agent", "QA & Test Agent"]},
    "monitor_deploy": {"num": 8, "name": "Monitoring, CI/CD & final report", "agents": ["Monitoring Agent", "DevOps Agent", "Report & Handoff Agent"]},
}


class WorkflowStudioState:
    def __init__(self, run_id: str, request: WorkflowRunRequest, root_dir: Path):
        self.run_id = run_id
        self.request = request
        self.root_dir = root_dir
        self.status = "queued"
        self.progress_pct = 0
        self.current_stage: str | None = None
        self.completed_stages: list[str] = []
        self.active_agents: list[str] = []
        self.completed_agents: list[str] = []
        self.events: list[WorkflowEvent] = []
        self.artifacts: dict[str, str] = {}
        self.metrics: dict[str, Any] = {}
        self.error: str | None = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None
        self._lock = threading.Lock()

    def add_event(self, msg: str, event_type: str = "info", stage: str | None = None,
                  stage_num: int | None = None, agent: str | None = None):
        t = datetime.now().strftime("%H:%M:%S")
        ev = WorkflowEvent(time=t, type=event_type, msg=msg, stage=stage, stage_num=stage_num, agent=agent)
        with self._lock:
            self.events.append(ev)
        logger.info("[Run %s] %s · %s", self.run_id, agent or stage or "SYSTEM", msg)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "status": self.status,
                "progress_pct": self.progress_pct,
                "current_stage": self.current_stage,
                "completed_stages": list(self.completed_stages),
                "active_agents": list(self.active_agents),
                "completed_agents": list(self.completed_agents),
                "events_count": len(self.events),
                "artifacts": dict(self.artifacts),
                "metrics": dict(self.metrics),
                "error": self.error,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
            }


class WorkflowStudioRunner:
    """Manages and executes Studio runs."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._runs = {}
                cls._instance._runs_lock = threading.Lock()
            return cls._instance

    def _get_base_dir(self, run_id: str) -> Path:
        settings = get_settings()
        base = settings.artifacts_dir / "runs" / run_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    def get_run(self, run_id: str) -> WorkflowStudioState | None:
        with self._runs_lock:
            return self._runs.get(run_id)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._runs_lock:
            runs = list(self._runs.values())
        return [r.to_dict() for r in reversed(runs[-limit:])]

    def create_run(self, request: WorkflowRunRequest) -> WorkflowStudioState:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        base_dir = self._get_base_dir(run_id)
        state = WorkflowStudioState(run_id, request, base_dir)
        with self._runs_lock:
            self._runs[run_id] = state
        return state

    def start_run_async(self, request: WorkflowRunRequest) -> WorkflowStudioState:
        state = self.create_run(request)
        thread = threading.Thread(target=self._execute_run, args=(state,), daemon=True)
        thread.start()
        return state

    def _execute_run(self, state: WorkflowStudioState):
        state.status = "running"
        req = state.request
        state.add_event(f"Workflow execution started. Seed={req.seed}, OOT={req.oot_pct}%, Stages={len(req.stages)}",
                        event_type="ok", agent="SYSTEM")

        total_stages = max(len(req.stages), 1)
        done_stages = 0

        frame = None
        splits = None
        champion_model = None

        def ensure_data():
            nonlocal frame, splits
            if frame is None:
                frame = generate_synthetic_dataset(rows=5000, seed=req.seed)
            if splits is None:
                pipe_eng = PipelineEngineer()
                target_col = "target" if "target" in frame.columns else frame.columns[-1]
                date_col = "application_date" if "application_date" in frame.columns else None
                splits, _ = pipe_eng.run(frame, target_col, date_col, req.oot_pct)
            return frame, splits

        try:
            for stage_key in req.stages:
                stage_meta = STAGE_METADATA.get(stage_key, {"num": done_stages + 1, "name": stage_key, "agents": []})
                s_num = stage_meta["num"]
                s_name = stage_meta["name"]
                agents = stage_meta["agents"]

                state.current_stage = stage_key
                state.active_agents = list(agents)
                state.add_event(f"Stage {s_num}: {s_name}", event_type="info", stage=stage_key, stage_num=s_num)

                # =====================================================================
                # Stage 1: Data intake & profiling
                # =====================================================================
                if stage_key == "data_intake":
                    state.add_event("Inspecting dataset source and file availability...", agent="Data Quality Agent")
                    dq_agent = DataQualityAgent()
                    frame, dq_res = dq_agent.run(req.dataset, seed=req.seed)
                    state.add_event(dq_res.summary, event_type=dq_res.status, agent=dq_agent.name)

                    tl_agent = TargetLeakageAgent()
                    tl_res = tl_agent.run(frame, req.target, state.root_dir)
                    state.artifacts.update(tl_res.artifacts)
                    state.add_event(tl_res.summary, event_type=tl_res.status, agent=tl_agent.name)

                # =====================================================================
                # Stage 2: Data preparation & feature engineering
                # =====================================================================
                elif stage_key == "feature_engineering":
                    if frame is None:
                        frame = generate_synthetic_dataset(rows=5000, seed=req.seed)

                    state.add_event("Building temporal holdout splits and fold-safe preprocessing...", agent="Feature Engineering Agent")
                    pipe_eng = PipelineEngineer()
                    target_col = "target" if "target" in frame.columns else frame.columns[-1]
                    date_col = "application_date" if "application_date" in frame.columns else None
                    splits, p_res = pipe_eng.run(frame, target_col, date_col, req.oot_pct)
                    state.add_event(p_res.summary, event_type=p_res.status, agent=pipe_eng.name)

                    fe_agent = FeatureEngineeringAgent()
                    feature_cols = [c for c in splits.train.columns if c not in ("target", splits.date_column)]
                    _specs, fe_res = fe_agent.run(splits.train, feature_cols)
                    state.add_event(fe_res.summary, event_type=fe_res.status, agent=fe_agent.name)

                # =====================================================================
                # Stage 3: Champion model training
                # =====================================================================
                elif stage_key == "train_champion":
                    if splits is None:
                        _, splits = ensure_data()

                    state.add_event("Fitting interpretable Logistic Regression champion...", agent="Modeling Agent")
                    mod_agent = ModelingAgent()
                    exclude = (splits.date_column,) if splits.date_column else ()
                    champion_model, mod_res = mod_agent.run(splits.train, "target", req.seed, exclude, state.root_dir)
                    state.artifacts.update(mod_res.artifacts)

                    test_features = splits.test.drop(columns=["target", splits.date_column] if splits.date_column else ["target"])
                    p_test = champion_model.predict_proba(test_features)
                    val_agent = ValidationAgent()
                    v_metrics, _ = val_agent.run(splits.test["target"], p_test)
                    state.metrics["champion_auroc"] = v_metrics["auroc"]
                    state.metrics["champion_brier"] = v_metrics["brier"]

                    track_agent = ExperimentTrackingAgent()
                    state.add_event(f"Champion trained & fingerprinted. Test AUROC: {v_metrics['auroc']:.4f}, Brier: {v_metrics['brier']:.4f}",
                                    event_type="ok", agent=track_agent.name)

                # =====================================================================
                # Stage 4: Challenger model training
                # =====================================================================
                elif stage_key == "train_challenger":
                    if splits is None:
                        _, splits = ensure_data()

                    chal_algo = "histgb"
                    if "forest" in req.challenger.lower():
                        chal_algo = "random_forest"

                    state.add_event(f"Fitting challenger ensemble ({chal_algo}) on equivalent folds...", agent="Challenger Agent")
                    chal_agent = ChallengerAgent()
                    exclude = (splits.date_column,) if splits.date_column else ()
                    _challenger_model, chal_res = chal_agent.run(
                        splits.train, splits.test, "target", chal_algo, req.seed, exclude, state.root_dir
                    )
                    state.artifacts.update(chal_res.artifacts)
                    state.metrics["challenger_auroc"] = chal_res.details.get("auroc", 0.0)
                    state.add_event(chal_res.summary, event_type=chal_res.status, agent=chal_agent.name)

                # =====================================================================
                # Stage 5: Independent validation & stress tests
                # =====================================================================
                elif stage_key == "validate":
                    val_agent = ValidationAgent()
                    state.add_event("Computing multi-metric suite: AUROC, KS, Brier, and Population PSI...", agent=val_agent.name)
                    if splits is None or champion_model is None:
                        _, splits = ensure_data()
                        mod_agent = ModelingAgent()
                        champion_model, _ = mod_agent.run(splits.train, "target", req.seed, (), state.root_dir)

                    test_features = splits.test.drop(columns=["target", splits.date_column] if splits.date_column else ["target"])
                    p_champ_test = champion_model.predict_proba(test_features)
                    _val_metrics, _val_res = val_agent.run(splits.test["target"], p_champ_test)

                    train_features = splits.train.drop(columns=["target", splits.date_column] if splits.date_column else ["target"])
                    p_champ_train = champion_model.predict_proba(train_features)

                    mrr_agent = ModelRiskReviewerAgent()
                    mrr_res = mrr_agent.run(p_champ_train, p_champ_test, state.run_id, req.dataset, state.metrics, state.root_dir)
                    state.artifacts.update(mrr_res.artifacts)
                    state.metrics["population_psi"] = mrr_res.details.get("population_psi", 0.0)
                    state.add_event(mrr_res.summary, event_type=mrr_res.status, agent=mrr_agent.name)

                # =====================================================================
                # Stage 6: Explainability & governance pack
                # =====================================================================
                elif stage_key == "governance":
                    state.add_event("Assembling governance pack, model card, and evidence checklist...", agent="Governance Agent")
                    if champion_model is None:
                        _, splits = ensure_data()
                        mod_agent = ModelingAgent()
                        champion_model, _ = mod_agent.run(splits.train, "target", req.seed, (), state.root_dir)

                    gov_agent = GovernanceAgent()
                    gov_res = gov_agent.run(champion_model, req.dataset, state.metrics, state.root_dir)
                    state.artifacts.update(gov_res.artifacts)

                    exp_agent = ExplainabilityAgent()
                    exp_res = exp_agent.run(champion_model)
                    state.add_event(gov_res.summary, event_type=gov_res.status, agent=gov_agent.name)
                    state.add_event(exp_res.summary, event_type=exp_res.status, agent=exp_agent.name)

                # =====================================================================
                # Stage 7: API packaging & tests
                # =====================================================================
                elif stage_key == "api_tests":
                    api_agent = ApiEngineerAgent()
                    api_res = api_agent.run()
                    state.add_event(api_res.summary, event_type=api_res.status, agent=api_agent.name)

                    qa_agent = QaTestAgent()
                    qa_res = qa_agent.run(state.root_dir)
                    state.artifacts.update(qa_res.artifacts)
                    state.add_event(qa_res.summary, event_type=qa_res.status, agent=qa_agent.name)

                # =====================================================================
                # Stage 8: Monitoring, CI/CD & final report
                # =====================================================================
                elif stage_key == "monitor_deploy":
                    mon_agent = MonitoringAgent()
                    state.add_event("Calculating baseline drift indicators and CI/CD manifest...", agent=mon_agent.name)
                    mon_res = mon_agent.run(state.root_dir)
                    state.artifacts.update(mon_res.artifacts)

                    devops_agent = DevOpsAgent()
                    devops_res = devops_agent.run()
                    state.add_event(devops_res.summary, event_type=devops_res.status, agent=devops_agent.name)

                    handoff_agent = ReportHandoffAgent()
                    handoff_res = handoff_agent.run(state.run_id, state.artifacts)
                    state.add_event(handoff_res.summary, event_type=handoff_res.status, agent=handoff_agent.name)

                # Stage completion
                state.completed_stages.append(stage_key)
                state.completed_agents.extend([a for a in agents if a not in state.completed_agents])
                state.active_agents = []
                done_stages += 1
                state.progress_pct = int((done_stages / total_stages) * 100)

                # Small sleep to simulate fluid processing and give frontend polling time to observe
                time.sleep(0.3)

            state.status = "completed"
            state.completed_at = datetime.now(timezone.utc).isoformat()
            state.add_event("Workflow execution completed successfully. All artifacts persisted.", event_type="ok", agent="SYSTEM")

        except Exception as exc:
            logger.exception("Workflow studio run failed")
            state.status = "failed"
            state.error = str(exc)
            state.completed_at = datetime.now(timezone.utc).isoformat()
            state.add_event(f"Execution failed: {exc}", event_type="error", agent="SYSTEM")
