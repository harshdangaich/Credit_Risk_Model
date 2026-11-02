"""Orchestrator: coordinates specialized agents, workflow state, and bounded feedback loops."""
from __future__ import annotations

import time
import uuid
from datetime import date, datetime, timezone
from typing import Any

from credit_risk_model.agents.data_agent import DataAgent
from credit_risk_model.agents.evaluation_agent import EvaluationAgent
from credit_risk_model.agents.experiment_agent import ExperimentAgent
from credit_risk_model.agents.tuning_agent import TuningAgent
from credit_risk_model.config import get_settings
from credit_risk_model.experimentation.schemas import (
    CandidatePromotionRequest,
    CandidatePromotionResponse,
    ExperimentObjective,
    ExperimentPlan,
    ExperimentStatus,
    WorkflowResult,
    WorkflowStatus,
)
from credit_risk_model.experimentation.store import ExperimentStore
from credit_risk_model.governance.audit import record_change
from credit_risk_model.governance.inventory import (
    ModelEntry,
    ModelInventory,
    ModelStatus,
    RiskTier,
)


class Orchestrator:
    """Coordinates data preparation, baseline training, iterative tuning, evaluation, and governance promotion."""

    def __init__(
        self,
        store: ExperimentStore | None = None,
        data_agent: DataAgent | None = None,
        tuning_agent: TuningAgent | None = None,
        experiment_agent: ExperimentAgent | None = None,
        evaluation_agent: EvaluationAgent | None = None,
    ):
        self.store = store or ExperimentStore()
        self.data_agent = data_agent or DataAgent()
        self.tuning_agent = tuning_agent or TuningAgent()
        self.experiment_agent = experiment_agent or ExperimentAgent(self.store)
        self.evaluation_agent = evaluation_agent or EvaluationAgent()

    def run_workflow(
        self,
        objective: ExperimentObjective,
        plan: ExperimentPlan | None = None,
    ) -> WorkflowResult:
        if plan is not None:
            if plan.objective_name != objective.name:
                raise ValueError(
                    f"Plan objective '{plan.objective_name}' does not match '{objective.name}'"
                )
            if not plan.markdown_path or not plan.planned_iterations:
                raise ValueError("Supplied experiment plan is invalid or incomplete")
            workflow_id = plan.workflow_id
        else:
            workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
            from credit_risk_model.agents.planner_agent import PlannerAgent
            plan = PlannerAgent(self.store).generate_plan(objective, workflow_id=workflow_id)

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start_time = time.perf_counter()

        workflow = WorkflowResult(
            workflow_id=workflow_id,
            objective=objective,
            status=WorkflowStatus.RUNNING,
            created_at_utc=created_at,
            plan=plan,
            plan_path=plan.markdown_path,
        )
        self.store.save_workflow(workflow)

        # 1. Data Preparation & Validation
        splits, data_report = self.data_agent.prepare_data(objective)
        if not data_report.is_valid or splits is None:
            workflow.status = WorkflowStatus.FAILED
            workflow.error_message = f"Data validation failed: {'; '.join(data_report.issues)}"
            workflow.completed_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.store.save_workflow(workflow)
            return workflow

        # 2. Baseline Model Execution & Evaluation
        base_model, base_run = self.experiment_agent.run_baseline(
            workflow_id=workflow_id,
            splits=splits,
            seed=objective.dataset_seed,
        )
        if base_run.status == ExperimentStatus.FAILED:
            workflow.status = WorkflowStatus.FAILED
            workflow.error_message = f"Baseline training failed: {base_run.error_message}"
            workflow.completed_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.store.save_workflow(workflow)
            return workflow

        base_metrics, _, _ = self.evaluation_agent.evaluate_model(
            model=base_model,
            splits=splits,
            objective=objective,
        )
        base_run.metrics = base_metrics
        workflow.baseline_metrics = base_metrics
        workflow.experiments.append(base_run)
        self.store.save_workflow(workflow)

        # 3. Bounded Iterative Feedback Loop
        previous_feedback: dict[str, Any] | None = None
        max_runs = min(objective.max_experiments, 10)

        for iteration in range(1, max_runs + 1):
            elapsed = time.perf_counter() - start_time
            if elapsed >= objective.time_budget_seconds:
                break

            proposal = self.tuning_agent.propose_configuration(
                objective=objective,
                iteration=iteration,
                history=workflow.experiments,
                previous_feedback=previous_feedback,
                plan=plan,
            )

            cand_model, cand_run = self.experiment_agent.run_candidate(
                workflow_id=workflow_id,
                experiment_id=proposal.experiment_id,
                algorithm=proposal.algorithm,
                hyperparameters=proposal.hyperparameters,
                iteration=iteration,
                splits=splits,
                seed=objective.dataset_seed,
            )

            if cand_run.status == ExperimentStatus.SUCCESS and cand_model is not None:
                cand_metrics, passed_criteria, feedback = self.evaluation_agent.evaluate_model(
                    model=cand_model,
                    splits=splits,
                    objective=objective,
                    baseline_metrics=workflow.baseline_metrics,
                )
                cand_run.metrics = cand_metrics
                cand_run.passed_criteria = passed_criteria
                cand_run.criteria_feedback = feedback
                previous_feedback = feedback
            else:
                previous_feedback = {"error": cand_run.error_message}

            workflow.experiments.append(cand_run)
            self.store.save_workflow(workflow)

        # 4. Candidate Model Selection
        eligible = [
            exp for exp in workflow.experiments
            if exp.iteration > 0 and exp.status == ExperimentStatus.SUCCESS and exp.passed_criteria
        ]

        target_metric = objective.target_metric.lower()
        if eligible:
            best_candidate = max(eligible, key=lambda x: x.metrics.get(target_metric, 0))
            workflow.selected_candidate_id = best_candidate.experiment_id
            workflow.candidate_reasoning = (
                f"Candidate {best_candidate.experiment_id} ({best_candidate.model_type}) satisfied all "
                f"criteria with {target_metric}={best_candidate.metrics.get(target_metric)} "
                f"(baseline={workflow.baseline_metrics.get(target_metric)})."
            )
            workflow.status = WorkflowStatus.AWAITING_APPROVAL
        else:
            candidates = [exp for exp in workflow.experiments if exp.iteration > 0 and exp.status == ExperimentStatus.SUCCESS]
            if candidates:
                best_candidate = max(candidates, key=lambda x: x.metrics.get(target_metric, 0))
                workflow.selected_candidate_id = best_candidate.experiment_id
                workflow.candidate_reasoning = (
                    f"Candidate {best_candidate.experiment_id} achieved best {target_metric}="
                    f"{best_candidate.metrics.get(target_metric)}, but did not strictly meet all criteria: "
                    f"{best_candidate.criteria_feedback.get('reasons')}."
                )
                workflow.status = WorkflowStatus.AWAITING_APPROVAL
            else:
                workflow.candidate_reasoning = "No candidate models completed successfully."
                workflow.status = WorkflowStatus.COMPLETED

        workflow.completed_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.store.save_workflow(workflow)

        record_change(
            "experiment_workflow_completed",
            workflow_id=workflow_id,
            selected_candidate=workflow.selected_candidate_id,
            status=workflow.status.value,
            runs_count=len(workflow.experiments),
        )
        return workflow

    def promote_candidate(
        self,
        workflow_id: str,
        experiment_id: str,
        request: CandidatePromotionRequest,
    ) -> CandidatePromotionResponse:
        """Promote an approved candidate model to the governance model inventory."""
        workflow = self.store.get_workflow(workflow_id)
        if not workflow:
            raise LookupError(f"Workflow {workflow_id} not found")

        matching_run = next((r for r in workflow.experiments if r.experiment_id == experiment_id), None)
        if not matching_run or matching_run.status != ExperimentStatus.SUCCESS:
            raise ValueError(f"Experiment {experiment_id} is not an eligible successful run")

        settings = get_settings()
        inventory_path = settings.inventory_path
        today = date.today()

        inventory = ModelInventory.load(inventory_path)
        candidate_version = f"candidate-{experiment_id}"
        if request.promote_as_champion:
            model_name = settings.champion_name
            target_status = ModelStatus.ACTIVE
            for e in inventory.entries:
                if e.model_name == model_name and e.status == ModelStatus.ACTIVE:
                    e.status = ModelStatus.RETIRED
            purpose = "probability_of_default_scoring"
            risk_tier = RiskTier.HIGH
            risk_tier_rationale = "PD estimates feed credit decisions; promoted champion from verified multi-agent workflow."
            intended_use = "Production probability of default estimation and credit scoring."
            limitations = ["Trained on verified dataset; subject to continuous monitoring and annual review."]
            use_restrictions = ["Demonstration credit scoring platform."]
        else:
            model_name = f"candidate_{matching_run.model_type}"
            target_status = ModelStatus.ACTIVE if request.target_status == "active" else ModelStatus.UNDER_REVIEW
            purpose = "candidate_credit_risk_model"
            risk_tier = RiskTier.MEDIUM
            risk_tier_rationale = "Experimental credit risk model approved for validation benchmarking."
            intended_use = "Independent validation and benchmarking against champion."
            limitations = ["Experimental model from multi-agent workflow; requires independent validation."]
            use_restrictions = ["Not for production credit decisions."]

        new_entry = ModelEntry(
            model_name=model_name,
            model_version=candidate_version,
            model_type=matching_run.model_type,
            owner="ml-experimentation-agent",
            purpose=purpose,
            status=target_status,
            artifact_path=str(matching_run.artifact_path),
            approved_by=request.approved_by,
            approval_date=today if target_status == ModelStatus.ACTIVE else None,
            risk_tier=risk_tier,
            risk_tier_rationale=risk_tier_rationale,
            intended_use=intended_use,
            limitations=limitations,
            use_restrictions=use_restrictions,
            artifact_sha256=matching_run.artifact_sha256,
        )

        existing = inventory.get(model_name, candidate_version)
        if existing:
            inventory.entries.remove(existing)
        inventory.entries.append(new_entry)

        # Persist updated inventory
        import yaml
        raw_models = [e.model_dump(mode="json", exclude_none=True) for e in inventory.entries]
        inventory_path.write_text(yaml.safe_dump({"models": raw_models}, sort_keys=False), encoding="utf-8")

        workflow.promotion_status = "approved"
        workflow.promotion_details = {
            "model_name": model_name,
            "model_version": candidate_version,
            "approved_by": request.approved_by,
            "approval_date": str(today),
            "target_status": request.target_status,
        }
        self.store.save_workflow(workflow)

        record_change(
            "model_promoted",
            workflow_id=workflow_id,
            experiment_id=experiment_id,
            promoted_model=model_name,
            model_version=candidate_version,
            approved_by=request.approved_by,
        )

        return CandidatePromotionResponse(
            workflow_id=workflow_id,
            experiment_id=experiment_id,
            promoted_model_name=model_name,
            promoted_model_version=candidate_version,
            inventory_status=new_entry.status.value,
            approval_date=str(today),
            approved_by=request.approved_by,
        )
