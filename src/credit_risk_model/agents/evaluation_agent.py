"""Evaluation Agent: evaluates candidate models against baseline and acceptance criteria."""
from __future__ import annotations

from typing import Any

from credit_risk_model.experimentation.schemas import ExperimentObjective
from credit_risk_model.models import predict_pd
from credit_risk_model.validation.calibration import calibration
from credit_risk_model.validation.discrimination import discrimination
from credit_risk_model.validation.oot import evaluate_sample
from credit_risk_model.validation.stability import psi
from credit_risk_model.validation.suite import Splits


class EvaluationAgent:
    """Evaluates discrimination, calibration, stability, and out-of-time performance for candidate models."""

    def evaluate_model(
        self,
        model: Any,
        splits: Splits,
        objective: ExperimentObjective,
        baseline_metrics: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        date_col = splits.date_column
        exclude = [c for c in ("target", date_col) if c and c in splits.test.columns]
        test_X = splits.test.drop(columns=exclude)
        y_test = splits.test["target"].astype(int)

        p_test = predict_pd(model, test_X)

        disc = discrimination(y_test, p_test)
        cal = calibration(y_test, p_test)

        train_X = splits.train.drop(columns=[c for c in ("target", date_col) if c and c in splits.train.columns])
        p_train = predict_pd(model, train_X)
        score_psi = psi(p_train, p_test)

        metrics: dict[str, Any] = {
            "auroc": round(disc["auroc"], 4),
            "gini": round(disc["gini"], 4),
            "ks": round(disc["ks"], 4),
            "brier_score": round(cal["brier"], 4),
            "log_loss": round(cal["log_loss"], 4),
            "calibration_ratio": round(cal["calibration_ratio"], 4) if cal.get("calibration_ratio") else None,
            "population_psi": round(score_psi, 4),
        }

        if splits.oot is not None:
            oot_X = splits.oot.drop(columns=[c for c in ("target", date_col) if c and c in splits.oot.columns])
            oot_eval = evaluate_sample(splits.oot["target"], predict_pd(model, oot_X), splits.oot.get(date_col))
            if oot_eval.get("status") == "assessed":
                metrics["oot_auroc"] = round(oot_eval["auroc"], 4)
            else:
                metrics["oot_auroc"] = None

        criteria_checks: dict[str, Any] = {}
        passed = True
        reasons: list[str] = []

        target_metric = objective.target_metric.lower()
        candidate_target_val = metrics.get(target_metric, metrics["auroc"])

        if objective.min_target_metric is not None:
            met = candidate_target_val >= objective.min_target_metric
            criteria_checks["min_target_metric"] = {
                "metric": target_metric,
                "required": objective.min_target_metric,
                "achieved": candidate_target_val,
                "passed": met,
            }
            if not met:
                passed = False
                reasons.append(f"{target_metric} ({candidate_target_val}) < required ({objective.min_target_metric})")

        if objective.max_brier_score is not None:
            met = metrics["brier_score"] <= objective.max_brier_score
            criteria_checks["max_brier_score"] = {
                "required": objective.max_brier_score,
                "achieved": metrics["brier_score"],
                "passed": met,
            }
            if not met:
                passed = False
                reasons.append(f"brier_score ({metrics['brier_score']}) > allowed ({objective.max_brier_score})")

        if objective.max_psi is not None:
            met = metrics["population_psi"] <= objective.max_psi
            criteria_checks["max_psi"] = {
                "required": objective.max_psi,
                "achieved": metrics["population_psi"],
                "passed": met,
            }
            if not met:
                passed = False
                reasons.append(f"population_psi ({metrics['population_psi']}) > allowed ({objective.max_psi})")

        improvement: float | None = None
        if baseline_metrics and target_metric in baseline_metrics:
            base_val = baseline_metrics[target_metric]
            improvement = round(candidate_target_val - base_val, 4)
            criteria_checks["vs_baseline"] = {
                "baseline": base_val,
                "candidate": candidate_target_val,
                "improvement": improvement,
            }
            if improvement <= 0:
                reasons.append(f"Did not improve on baseline {target_metric} ({candidate_target_val} vs {base_val})")

        feedback: dict[str, Any] = {
            "checks": criteria_checks,
            "target_metric_improvement": improvement,
            "reasons": reasons,
        }
        return metrics, passed, feedback
