"""Tuning Agent: proposes bounded, feedback-driven hyperparameter configurations."""
from __future__ import annotations

import uuid
from typing import Any, ClassVar

from credit_risk_model.experimentation.schemas import (
    ExperimentObjective,
    ExperimentPlan,
    ExperimentRun,
    TuningProposal,
)


class TuningAgent:
    """Proposes bounded hyperparameter configurations for candidate models based on evaluation feedback."""

    HISTGB_SEARCH_SPACE: ClassVar[list[dict[str, Any]]] = [
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 150, "min_samples_leaf": 50},
        {"learning_rate": 0.03, "max_depth": 4, "max_iter": 200, "min_samples_leaf": 30},
        {"learning_rate": 0.08, "max_depth": 3, "max_iter": 120, "min_samples_leaf": 40},
        {"learning_rate": 0.02, "max_depth": 5, "max_iter": 250, "min_samples_leaf": 50},
    ]

    RANDOM_FOREST_SEARCH_SPACE: ClassVar[list[dict[str, Any]]] = [
        {"n_estimators": 150, "min_samples_leaf": 20, "max_depth": 6},
        {"n_estimators": 200, "min_samples_leaf": 15, "max_depth": 8},
        {"n_estimators": 250, "min_samples_leaf": 30, "max_depth": 5},
    ]

    def _resolve_planned_hyperparameters(self, algo: str, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert planned LLM hyperparameter specs (scalars or ranges) into concrete validated values."""
        resolved: dict[str, Any] = {}
        for k, v in raw.items():
            val = v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else v
            resolved[k] = val

        if algo == "histgb":
            lr = float(resolved.get("learning_rate", 0.05))
            resolved["learning_rate"] = min(max(lr, 0.01), 0.2)
            depth = int(resolved.get("max_depth", 3))
            resolved["max_depth"] = min(max(depth, 2), 8)
            leaf = int(resolved.get("min_samples_leaf", 50))
            resolved["min_samples_leaf"] = min(max(leaf, 10), 200)
            iters = int(resolved.get("max_iter", 150))
            resolved["max_iter"] = min(max(iters, 50), 300)
        elif algo == "random_forest":
            trees = int(resolved.get("n_estimators", 150))
            resolved["n_estimators"] = min(max(trees, 50), 300)
            depth = int(resolved.get("max_depth", 6))
            resolved["max_depth"] = min(max(depth, 2), 10)
            leaf = int(resolved.get("min_samples_leaf", 20))
            resolved["min_samples_leaf"] = min(max(leaf, 5), 100)

        return resolved

    def propose_configuration(
        self,
        objective: ExperimentObjective,
        iteration: int,
        history: list[ExperimentRun],
        previous_feedback: dict[str, Any] | None = None,
        plan: ExperimentPlan | None = None,
    ) -> TuningProposal:
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        supported = [algo for algo in objective.candidate_algorithms if algo in ("histgb", "random_forest")]
        if not supported:
            supported = ["histgb"]

        # Check if plan contains a planned iteration for this step
        planned_it = None
        if plan and plan.planned_iterations:
            planned_it = next((it for it in plan.planned_iterations if it.iteration == iteration), None)

        if planned_it and planned_it.algorithm in supported:
            algo = planned_it.algorithm
            params = self._resolve_planned_hyperparameters(algo, planned_it.target_hyperparameters)
            rationale = planned_it.rationale or f"Iteration {iteration}: Proposing LLM-planned {algo} configuration."
        else:
            algo_idx = (iteration - 1) % len(supported)
            algo = supported[algo_idx]
            space = self.HISTGB_SEARCH_SPACE if algo == "histgb" else self.RANDOM_FOREST_SEARCH_SPACE
            space_idx = ((iteration - 1) // len(supported)) % len(space)
            params = dict(space[space_idx])
            rationale = f"Iteration {iteration}: Proposing bounded {algo} configuration."

        # Feedback adjustments
        if previous_feedback:
            metric_diff = previous_feedback.get("target_metric_improvement")
            checks = previous_feedback.get("checks", {})
            brier_passed = checks.get("max_brier_score", {}).get("passed", True)
            psi_passed = checks.get("max_psi", {}).get("passed", True)

            if metric_diff is not None and metric_diff < 0:
                rationale += " Previous iteration underperformed; shifting hyperparameter bounds."
                if algo == "histgb" and "learning_rate" in params:
                    params["learning_rate"] = max(0.01, round(float(params["learning_rate"]) * 0.8, 3))
                    params["min_samples_leaf"] = int(params["min_samples_leaf"]) + 10
                elif algo == "random_forest":
                    if "max_depth" in params:
                        params["max_depth"] = max(3, int(params["max_depth"]) - 1)
                    if "min_samples_leaf" in params:
                        params["min_samples_leaf"] = int(params["min_samples_leaf"]) + 10

            if not brier_passed:
                rationale += " Calibration hurdle breached; regularizing learning rate & leaf counts."
                if algo == "histgb" and "learning_rate" in params:
                    params["learning_rate"] = max(0.01, round(float(params["learning_rate"]) * 0.7, 3))
                if "min_samples_leaf" in params:
                    params["min_samples_leaf"] = int(params["min_samples_leaf"]) + 15

            if not psi_passed:
                rationale += " Population stability breached; pruning tree depth to prevent drift."
                if "max_depth" in params:
                    params["max_depth"] = max(3, int(params["max_depth"]) - 2)

        return TuningProposal(
            experiment_id=exp_id,
            algorithm=algo,
            hyperparameters=params,
            rationale=rationale,
            iteration=iteration,
        )
