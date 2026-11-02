"""Data Agent: dataset inspection, schema validation, and preparation."""
from __future__ import annotations

from credit_risk_model.data.sources import load_dataset
from credit_risk_model.experimentation.schemas import DataValidationReport, ExperimentObjective
from credit_risk_model.validation.suite import Splits, split_dataset


class DataAgent:
    """Inspects, validates, and prepares datasets for the experimentation platform."""

    def prepare_data(self, objective: ExperimentObjective) -> tuple[Splits | None, DataValidationReport]:
        issues: list[str] = []
        try:
            frame, source = load_dataset(
                data=objective.dataset_ref,
                rows=objective.dataset_rows,
                seed=objective.dataset_seed,
            )
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            report = DataValidationReport(
                rows=0,
                columns=[],
                target_column="target",
                default_rate=0.0,
                missing_rate_by_feature={},
                constant_features=[],
                is_valid=False,
                issues=[f"Failed to load dataset: {exc}"],
            )
            return None, report

        if frame.empty:
            issues.append("Dataset is empty")
        if "target" not in frame.columns:
            issues.append("Missing required 'target' column")

        if issues:
            report = DataValidationReport(
                rows=len(frame),
                columns=list(frame.columns),
                target_column="target",
                default_rate=0.0,
                missing_rate_by_feature={},
                constant_features=[],
                is_valid=False,
                issues=issues,
            )
            return None, report

        y = frame["target"].dropna()
        if y.nunique() != 2 or not set(y.unique()).issubset({0, 1}):
            issues.append(f"Target column must contain binary 0/1 classes, found: {sorted(y.unique())}")

        default_rate = float(y.mean()) if len(y) else 0.0
        date_column = source.get("date_column") if source.get("date_column") in frame else None
        feature_cols = [c for c in frame.columns if c not in ("target", date_column)]

        missing_rates = {col: float(frame[col].isna().mean()) for col in feature_cols}
        constant_features = [col for col in feature_cols if frame[col].nunique(dropna=True) <= 1]

        if constant_features:
            issues.append(f"Constant features detected: {constant_features}")
        for col, rate in missing_rates.items():
            if rate > 0.5:
                issues.append(f"High missingness in feature '{col}': {rate:.1%}")

        is_valid = len([i for i in issues if "must contain" in i or "empty" in i or "Missing" in i]) == 0

        report = DataValidationReport(
            rows=len(frame),
            columns=list(frame.columns),
            target_column="target",
            default_rate=default_rate,
            missing_rate_by_feature=missing_rates,
            constant_features=constant_features,
            is_valid=is_valid,
            issues=issues,
        )

        if not is_valid:
            return None, report

        splits = split_dataset(
            frame,
            target="target",
            date_column=date_column,
            test_fraction=0.25,
            oot_fraction=0.2 if date_column else 0.0,
            seed=objective.dataset_seed,
        )
        return splits, report
