"""FastAPI Router for the Credit Risk Workflow Studio (Planner Frontend Integration)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import HTMLResponse

from credit_risk_model.experimentation.workflow_studio import (
    WorkflowRunRequest,
    WorkflowRunResponse,
    WorkflowStudioRunner,
)

router = APIRouter(tags=["workflow_studio"])


@router.post("/workflows/credit-risk/run", response_model=WorkflowRunResponse)
def run_credit_risk_workflow(payload: WorkflowRunRequest):
    """Launch the 8-stage Credit Risk Workflow configured from the Frontend Planner."""
    runner = WorkflowStudioRunner()
    state = runner.start_run_async(payload)
    return WorkflowRunResponse(
        run_id=state.run_id,
        status="running",
        events_url=f"/runs/{state.run_id}/events",
        artifacts_url=f"/runs/{state.run_id}/artifacts",
    )


@router.get("/runs/{run_id}")
def get_run_status(run_id: str) -> dict[str, Any]:
    """Retrieve full run status, progress, stage states, and artifacts."""
    runner = WorkflowStudioRunner()
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run.to_dict()


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str, since_index: int = Query(0, ge=0)) -> dict[str, Any]:
    """Retrieve event logs for streaming into the Frontend Run Console."""
    runner = WorkflowStudioRunner()
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    with run._lock:
        events = [e.model_dump() for e in run.events[since_index:]]
        total = len(run.events)
    return {
        "run_id": run_id,
        "status": run.status,
        "progress_pct": run.progress_pct,
        "current_stage": run.current_stage,
        "active_agents": list(run.active_agents),
        "completed_stages": list(run.completed_stages),
        "events": events,
        "total_events": total,
    }


@router.get("/runs/{run_id}/artifacts")
def get_run_artifacts(run_id: str) -> dict[str, Any]:
    """List all deliverable files produced by the workflow run."""
    runner = WorkflowStudioRunner()
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    items = []
    for name, path_str in run.artifacts.items():
        p = Path(path_str)
        items.append({
            "name": name,
            "path": path_str,
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "download_url": f"/runs/{run_id}/artifacts/{name}",
        })
    return {"run_id": run_id, "artifacts": items}


@router.get("/runs/{run_id}/artifacts/{artifact_name}")
def download_run_artifact(run_id: str, artifact_name: str):
    """Download a specific deliverable produced by the workflow."""
    runner = WorkflowStudioRunner()
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    path_str = run.artifacts.get(artifact_name)
    if not path_str or not Path(path_str).exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_name}' not found")
    content = Path(path_str).read_bytes()
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{artifact_name}"'},
    )


@router.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
    """List recent Studio runs."""
    runner = WorkflowStudioRunner()
    return runner.list_runs(limit=limit)


@router.get("/planner", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def serve_planner_ui():
    """Serve the interactive HTML workflow planner."""
    candidate_paths = [
        Path(__file__).resolve().parents[2] / "credit_risk_agent_workflow_planner.html",
        Path(__file__).resolve().parents[3] / "credit_risk_agent_workflow_planner.html",
    ]
    for p in candidate_paths:
        if p.exists():
            return HTMLResponse(content=p.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Planner HTML not found")
