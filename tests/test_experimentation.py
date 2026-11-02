"""Tests for the multi-agent ML experimentation platform."""
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from credit_risk_model.agents.data_agent import DataAgent
from credit_risk_model.agents.evaluation_agent import EvaluationAgent
from credit_risk_model.agents.experiment_agent import ExperimentAgent
from credit_risk_model.agents.orchestrator import Orchestrator
from credit_risk_model.agents.planner_agent import PlannerAgent
from credit_risk_model.agents.tuning_agent import TuningAgent
from credit_risk_model.experimentation.schemas import (
    CandidatePromotionRequest,
    ExperimentObjective,
    ExperimentPlan,
    ExperimentStatus,
    WorkflowStatus,
)
from credit_risk_model.experimentation.store import ExperimentStore
from credit_risk_model.governance.inventory import ModelInventory


@pytest.fixture
def temp_exp_store(tmp_path):
    return ExperimentStore(root=tmp_path / "experiments")


@pytest.fixture
def sample_objective():
    return ExperimentObjective(
        name="test_experiment",
        target_metric="auroc",
        min_target_metric=0.60,
        max_brier_score=0.15,
        max_psi=0.25,
        dataset_rows=500,
        dataset_seed=42,
        candidate_algorithms=["histgb", "random_forest"],
        max_experiments=2,
        time_budget_seconds=60,
    )


def test_data_agent_valid(sample_objective):
    agent = DataAgent()
    splits, report = agent.prepare_data(sample_objective)
    assert report.is_valid is True
    assert splits is not None
    assert len(splits.train) > 0
    assert len(splits.test) > 0
    assert report.default_rate > 0


def test_data_agent_invalid():
    agent = DataAgent()
    obj = ExperimentObjective(dataset_ref="non_existent_file.csv")
    splits, report = agent.prepare_data(obj)
    assert splits is None
    assert report.is_valid is False
    assert len(report.issues) > 0


def test_tuning_agent_proposals(sample_objective):
    agent = TuningAgent()
    proposal_1 = agent.propose_configuration(sample_objective, iteration=1, history=[])
    assert proposal_1.algorithm in ("histgb", "random_forest")
    assert isinstance(proposal_1.hyperparameters, dict)
    assert len(proposal_1.hyperparameters) > 0

    proposal_2 = agent.propose_configuration(
        sample_objective,
        iteration=2,
        history=[],
        previous_feedback={"target_metric_improvement": -0.05},
    )
    assert proposal_2.iteration == 2
    assert "Previous iteration underperformed" in proposal_2.rationale


def test_experiment_agent_and_store(temp_exp_store, sample_objective):
    data_agent = DataAgent()
    splits, _ = data_agent.prepare_data(sample_objective)
    assert splits is not None

    exp_agent = ExperimentAgent(store=temp_exp_store)
    model, run = exp_agent.run_baseline("test-wf", splits, seed=42)
    assert model is not None
    assert run.status == ExperimentStatus.SUCCESS
    assert run.artifact_path is not None
    assert Path(run.artifact_path).exists()
    assert run.artifact_sha256 is not None

    loaded_model = temp_exp_store.load_artifact(run.artifact_path)
    assert loaded_model is not None


def test_evaluation_agent(sample_objective):
    data_agent = DataAgent()
    splits, _ = data_agent.prepare_data(sample_objective)
    assert splits is not None

    exp_agent = ExperimentAgent()
    model, _ = exp_agent.run_baseline("test-wf", splits, seed=42)

    eval_agent = EvaluationAgent()
    metrics, passed, feedback = eval_agent.evaluate_model(model, splits, sample_objective)

    assert "auroc" in metrics
    assert "brier_score" in metrics
    assert "population_psi" in metrics
    assert isinstance(passed, bool)
    assert "checks" in feedback


def test_orchestrator_end_to_end(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments"
    inv_file = tmp_path / "model_inventory.yaml"
    shutil.copy("governance/model_inventory.yaml", inv_file)

    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    store = ExperimentStore(root=exp_dir)
    orchestrator = Orchestrator(store=store)

    obj = ExperimentObjective(
        name="quick_test",
        target_metric="auroc",
        min_target_metric=0.60,
        max_brier_score=0.15,
        max_psi=0.30,
        dataset_rows=400,
        dataset_seed=42,
        candidate_algorithms=["histgb"],
        max_experiments=2,
        time_budget_seconds=30,
    )

    workflow = orchestrator.run_workflow(obj)
    assert workflow.status in (WorkflowStatus.AWAITING_APPROVAL, WorkflowStatus.COMPLETED)
    assert len(workflow.experiments) >= 2  # baseline + at least 1 candidate
    assert workflow.selected_candidate_id is not None
    assert workflow.candidate_reasoning is not None

    saved_wf = store.get_workflow(workflow.workflow_id)
    assert saved_wf is not None
    assert saved_wf.workflow_id == workflow.workflow_id

    # Test candidate promotion
    with patch("credit_risk_model.agents.orchestrator.get_settings") as mock_settings:
        from credit_risk_model.config import get_settings
        real_settings = get_settings()
        mock_settings.return_value = real_settings
        mock_settings.return_value.inventory_path = inv_file

        promo_req = CandidatePromotionRequest(
            approved_by="TestSupervisor",
            target_status="under_review",
            approval_notes="Approved for benchmark testing",
        )
        promo_resp = orchestrator.promote_candidate(
            workflow.workflow_id,
            workflow.selected_candidate_id,
            promo_req,
        )

        assert promo_resp.approved_by == "TestSupervisor"
        assert promo_resp.promoted_model_name.startswith("candidate_")

        # Verify entry in inventory
        updated_inv = ModelInventory.load(inv_file)
        entry = updated_inv.get(promo_resp.promoted_model_name, promo_resp.promoted_model_version)
        assert entry is not None
        assert entry.approved_by == "TestSupervisor"


def test_api_experiments_endpoints(tmp_path, monkeypatch):
    client = TestClient(app)

    # 1. Create experiment workflow via POST
    payload = {
        "name": "api_test",
        "target_metric": "auroc",
        "min_target_metric": 0.55,
        "dataset_rows": 300,
        "max_experiments": 1,
        "candidate_algorithms": ["histgb"],
    }
    resp = client.post("/api/experiments", json=payload)
    assert resp.status_code == 200
    wf = resp.json()
    wf_id = wf["workflow_id"]
    assert wf["status"] in ("awaiting_approval", "completed")

    # 2. List experiment workflows
    list_resp = client.get("/api/experiments")
    assert list_resp.status_code == 200
    assert any(w["workflow_id"] == wf_id for w in list_resp.json())

    # 3. Get single experiment workflow
    get_resp = client.get(f"/api/experiments/{wf_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["workflow_id"] == wf_id

    # 4. Get workflow results comparison
    results_resp = client.get(f"/api/experiments/{wf_id}/results")
    assert results_resp.status_code == 200
    assert "experiments" in results_resp.json()
    assert len(results_resp.json()["experiments"]) >= 1

    # 5. Non-existent workflow returns 404
    bad_resp = client.get("/api/experiments/non-existent-wf")
    assert bad_resp.status_code == 404

    # 6. Retrieve generated experiment plan via GET
    plan_resp = client.get(f"/api/experiments/{wf_id}/plan")
    assert plan_resp.status_code == 200
    plan_data = plan_resp.json()
    assert plan_data["workflow_id"] == wf_id
    assert "flowchart TD" in plan_data["plan_markdown"]
    assert "Planner Agent" in plan_data["plan_markdown"]


def test_planner_agent_generates_valid_plan(temp_exp_store, sample_objective):
    planner = PlannerAgent(store=temp_exp_store)
    plan = planner.generate_plan(sample_objective)

    assert isinstance(plan, ExperimentPlan)
    assert plan.objective_name == sample_objective.name
    assert plan.target_metric == sample_objective.target_metric
    assert len(plan.planned_iterations) == sample_objective.max_experiments
    assert len(plan.execution_steps) > 0
    assert "Planner Agent" in plan.agent_responsibilities
    assert "Orchestrator" in plan.agent_responsibilities

    # Verify markdown file
    assert plan.markdown_path is not None
    plan_file = Path(plan.markdown_path)
    assert plan_file.exists()
    assert plan_file.name == "experiment_plan.md"
    assert "```mermaid" in plan.markdown_plan
    assert "flowchart TD" in plan.markdown_plan


def test_planner_agent_mermaid_diagram_present(temp_exp_store, sample_objective):
    planner = PlannerAgent(store=temp_exp_store)
    plan = planner.generate_plan(sample_objective)

    mermaid_section = plan.markdown_plan
    assert "User([User Objective]) --> Planner[Planner Agent: LLM]" in mermaid_section
    assert "Planner --> PlanDoc[experiment_plan.md]" in mermaid_section
    assert "PlanDoc --> Orch[Existing Orchestrator]" in mermaid_section
    assert "Check -- Yes --> Tune" in mermaid_section
    assert "Check -- No --> Select[Candidate Selection]" in mermaid_section


def test_invalid_plan_rejected_by_orchestrator(sample_objective):
    orchestrator = Orchestrator()
    invalid_plan = ExperimentPlan(
        workflow_id="wf-bad",
        objective_name="mismatched_name",
        target_metric="auroc",
        hypothesis="test",
        markdown_plan="test",
        markdown_path="non_existent_path",
    )
    with pytest.raises(ValueError, match="does not match"):
        orchestrator.run_workflow(sample_objective, plan=invalid_plan)


def test_orchestrator_executes_with_plan(tmp_path, sample_objective):
    store = ExperimentStore(root=tmp_path / "exp")
    planner = PlannerAgent(store=store)
    plan = planner.generate_plan(sample_objective)

    orchestrator = Orchestrator(store=store)
    workflow = orchestrator.run_workflow(sample_objective, plan=plan)

    assert workflow.workflow_id == plan.workflow_id
    assert workflow.plan is not None
    assert workflow.plan_path == plan.markdown_path
    assert Path(workflow.plan_path).exists()


def test_tuning_agent_uses_llm_planned_hyperparameters(sample_objective):
    from credit_risk_model.agents.tuning_agent import TuningAgent
    from credit_risk_model.experimentation.schemas import PlannedIteration

    tuning = TuningAgent()
    planned_params = {"learning_rate": 0.07, "max_depth": 4, "min_samples_leaf": 45}
    plan = ExperimentPlan(
        workflow_id="wf-plan-test",
        objective_name=sample_objective.name,
        target_metric=sample_objective.target_metric,
        hypothesis="Testing planned hyperparameters",
        planned_iterations=[
            PlannedIteration(
                iteration=1,
                algorithm="histgb",
                target_hyperparameters=planned_params,
                rationale="LLM planned hyperparameter test",
            )
        ],
        markdown_plan="plan",
        markdown_path="path",
    )

    proposal = tuning.propose_configuration(
        objective=sample_objective,
        iteration=1,
        history=[],
        plan=plan,
    )

    assert proposal.algorithm == "histgb"
    assert proposal.hyperparameters["learning_rate"] == 0.07
    assert proposal.hyperparameters["max_depth"] == 4
    assert proposal.hyperparameters["min_samples_leaf"] == 45
    assert "LLM planned" in proposal.rationale


