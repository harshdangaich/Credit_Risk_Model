"""Experiment Agent: executes model training runs and persists isolated artifacts."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from credit_risk_model.experimentation.schemas import ExperimentRun, ExperimentStatus
from credit_risk_model.experimentation.store import ExperimentStore
from credit_risk_model.models.challenger import build_challenger
from credit_risk_model.models.champion import ChampionModel
from credit_risk_model.models.contract import infer_specs
from credit_risk_model.validation.suite import Splits


class ExperimentAgent:
    """Executes baseline and candidate training runs, capturing metrics, runtimes, and isolated artifacts."""

    def __init__(self, store: ExperimentStore | None = None):
        self.store = store or ExperimentStore()

    def run_baseline(
        self,
        workflow_id: str,
        splits: Splits,
        seed: int = 42,
    ) -> tuple[Any, ExperimentRun]:
        exp_id = "baseline-champion"
        start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.perf_counter()
        exclude = (splits.date_column,) if splits.date_column else ()

        try:
            model = ChampionModel.fit(
                splits.train,
                target="target",
                random_state=seed,
                class_weight=None,
                exclude=exclude,
                model_name="credit_risk_champion",
                model_version="baseline",
            )
            duration = round(time.perf_counter() - t0, 3)
            end_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

            artifact_path, digest = self.store.save_artifact(workflow_id, exp_id, model)
            run = ExperimentRun(
                experiment_id=exp_id,
                workflow_id=workflow_id,
                iteration=0,
                model_type="logistic_regression",
                hyperparameters={"class_weight": None, "random_state": seed},
                status=ExperimentStatus.SUCCESS,
                start_time_utc=start_utc,
                end_time_utc=end_utc,
                duration_seconds=duration,
                artifact_path=artifact_path,
                artifact_sha256=digest,
            )
            return model, run
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            duration = round(time.perf_counter() - t0, 3)
            end_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            run = ExperimentRun(
                experiment_id=exp_id,
                workflow_id=workflow_id,
                iteration=0,
                model_type="logistic_regression",
                hyperparameters={"class_weight": None, "random_state": seed},
                status=ExperimentStatus.FAILED,
                start_time_utc=start_utc,
                end_time_utc=end_utc,
                duration_seconds=duration,
                error_message=str(exc),
            )
            return None, run

    def run_candidate(
        self,
        workflow_id: str,
        experiment_id: str,
        algorithm: str,
        hyperparameters: dict[str, Any],
        iteration: int,
        splits: Splits,
        seed: int = 42,
    ) -> tuple[Any, ExperimentRun]:
        start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.perf_counter()
        exclude = (splits.date_column,) if splits.date_column else ()

        try:
            pipeline = build_challenger(
                splits.train,
                target="target",
                algorithm=algorithm,
                class_weight=None,
                random_state=seed,
                hyperparameters=hyperparameters,
                exclude=exclude,
            )
            duration = round(time.perf_counter() - t0, 3)
            end_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

            feature_cols = [c for c in splits.train.columns if c not in ("target", *exclude)]
            specs = infer_specs(splits.train[feature_cols])
            candidate_model = ChampionModel(
                pipeline=pipeline,
                feature_names=feature_cols,
                feature_specs=specs,
                metadata={
                    "model_name": f"candidate_{algorithm}",
                    "model_version": experiment_id,
                    "algorithm": algorithm,
                    "hyperparameters": hyperparameters,
                },
            )

            artifact_path, digest = self.store.save_artifact(workflow_id, experiment_id, candidate_model)
            run = ExperimentRun(
                experiment_id=experiment_id,
                workflow_id=workflow_id,
                iteration=iteration,
                model_type=algorithm,
                hyperparameters=hyperparameters,
                status=ExperimentStatus.SUCCESS,
                start_time_utc=start_utc,
                end_time_utc=end_utc,
                duration_seconds=duration,
                artifact_path=artifact_path,
                artifact_sha256=digest,
            )
            return candidate_model, run
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            duration = round(time.perf_counter() - t0, 3)
            end_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            run = ExperimentRun(
                experiment_id=experiment_id,
                workflow_id=workflow_id,
                iteration=iteration,
                model_type=algorithm,
                hyperparameters=hyperparameters,
                status=ExperimentStatus.FAILED,
                start_time_utc=start_utc,
                end_time_utc=end_utc,
                duration_seconds=duration,
                error_message=str(exc),
            )
            return None, run
