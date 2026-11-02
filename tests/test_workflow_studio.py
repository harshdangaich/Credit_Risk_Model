"""Tests for Workflow Studio endpoints and execution."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from api.main import app


def test_serve_planner_html(project):
    client = TestClient(app)
    response = client.get("/planner")
    assert response.status_code == 200
    assert "Credit Risk · Agent Orchestrator" in response.text
    assert "01 / Pipeline plan" in response.text


def test_workflow_studio_execution_and_events(project):
    client = TestClient(app)
    payload = {
        "dataset": "synthetic",
        "target": "loan_status",
        "champion": "logistic_regression",
        "challenger": "hist_gradient_boosting",
        "oot_pct": 20,
        "seed": 42,
        "stages": ["data_intake", "feature_engineering", "train_champion"],
        "agents": ["data_quality", "target_leakage", "feature_engineering", "modeling"],
    }

    # 1. Start run
    res = client.post("/workflows/credit-risk/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert data["status"] in ("queued", "running")
    run_id = data["run_id"]

    # 2. Wait briefly for execution
    for _ in range(30):
        status_res = client.get(f"/runs/{run_id}")
        assert status_res.status_code == 200
        run_data = status_res.json()
        if run_data["status"] in ("completed", "failed"):
            break
        time.sleep(0.3)

    assert run_data["status"] == "completed"
    assert "data_intake" in run_data["completed_stages"]
    assert "feature_engineering" in run_data["completed_stages"]
    assert "train_champion" in run_data["completed_stages"]

    # 3. Check events endpoint
    ev_res = client.get(f"/runs/{run_id}/events")
    assert ev_res.status_code == 200
    ev_data = ev_res.json()
    assert ev_data["run_id"] == run_id
    assert len(ev_data["events"]) > 0

    # 4. Check artifacts endpoint
    art_res = client.get(f"/runs/{run_id}/artifacts")
    assert art_res.status_code == 200
    art_data = art_res.json()
    assert len(art_data["artifacts"]) > 0
    art_names = [a["name"] for a in art_data["artifacts"]]
    assert "quality_summary.json" in art_names
    assert "champion.joblib" in art_names
