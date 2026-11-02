"""Tree-based challenger models.

Supported algorithms are listed in ``SUPPORTED_ALGORITHMS``. XGBoost and LightGBM need the
``challengers`` extra; requesting them without it raises a clear error instead of silently
falling back to another algorithm. Training is unweighted unless ``class_weight`` is given.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from .champion import build_preprocessor, split_xy
from .contract import infer_specs

SUPPORTED_ALGORITHMS = ("histgb", "random_forest", "xgboost", "lightgbm")


class UnsupportedAlgorithmError(ValueError):
    pass


def _estimator(algorithm: str, class_weight, random_state: int,
               hyperparameters: dict | None = None):
    params = dict(hyperparameters or {})
    if algorithm == "histgb":
        base_params = {
            "max_iter": 150, "learning_rate": 0.05, "max_depth": 3,
            "min_samples_leaf": 50, "class_weight": class_weight,
            "random_state": random_state,
        }
        base_params.update(params)
        return HistGradientBoostingClassifier(**base_params)
    if algorithm == "random_forest":
        base_params = {
            "n_estimators": 200, "min_samples_leaf": 20,
            "class_weight": class_weight, "random_state": random_state,
        }
        base_params.update(params)
        return RandomForestClassifier(**base_params)
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("algorithm 'xgboost' needs the 'challengers' extra: "
                               "pip install -e '.[challengers]'") from exc
        if class_weight is not None:
            raise UnsupportedAlgorithmError("class_weight is not supported for xgboost")
        base_params = {
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
            "eval_metric": "logloss", "random_state": random_state,
        }
        base_params.update(params)
        return XGBClassifier(**base_params)
    if algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("algorithm 'lightgbm' needs the 'challengers' extra: "
                               "pip install -e '.[challengers]'") from exc
        base_params = {
            "n_estimators": 200, "learning_rate": 0.05, "class_weight": class_weight,
            "random_state": random_state, "verbosity": -1,
        }
        base_params.update(params)
        return LGBMClassifier(**base_params)
    raise UnsupportedAlgorithmError(
        f"Unsupported challenger algorithm {algorithm!r}; "
        f"choose one of {', '.join(SUPPORTED_ALGORITHMS)}")


def build_challenger(frame: pd.DataFrame, target: str = "target", algorithm: str = "histgb",
                     *, class_weight=None, random_state: int = 42,
                     hyperparameters: dict | None = None,
                     exclude: tuple[str, ...] = ()) -> Pipeline:
    """Fit and return a challenger pipeline (preprocessing + estimator)."""
    estimator = _estimator(algorithm, class_weight, random_state, hyperparameters=hyperparameters)
    X, y = split_xy(frame, target, exclude)
    specs = infer_specs(X)
    pipe = Pipeline([("preprocess", build_preprocessor(specs, scale=False, dense=True)),
                     ("model", estimator)])
    pipe.fit(X, y)
    pipe.feature_names_ = list(X.columns)
    pipe.challenger_metadata_ = {
        "algorithm": algorithm, "class_weight": class_weight,
        "random_state": random_state,
        "hyperparameters": dict(hyperparameters or {}),
    }
    return pipe
