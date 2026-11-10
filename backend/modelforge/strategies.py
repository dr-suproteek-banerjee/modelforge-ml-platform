from typing import Any

import optuna
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from .neural import TorchClassifier, TorchRegressor, torch_available
from .schemas import ExperimentInput, Params


def estimator(config: ExperimentInput, params: Params) -> Any:
    classification = config.task == "classification"
    seed = config.seed
    match config.algorithm:
        case "logistic":
            return LogisticRegression(max_iter=600, random_state=seed, **params)
        case "linear":
            return LinearRegression()
        case "random_forest":
            cls = RandomForestClassifier if classification else RandomForestRegressor
            return cls(random_state=seed, n_jobs=1, **params)
        case "gradient_boosting":
            cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
            return cls(random_state=seed, **params)
        case "torch_mlp":
            if not torch_available():
                raise ValueError(
                    "PyTorch is optional. Enable the torch dependency on the API and worker."
                )
            neural = TorchClassifier if classification else TorchRegressor
            return neural(
                seed=seed,
                hidden=int(params.get("hidden", 32)),
                learning_rate=float(params.get("learning_rate", 0.01)),
            )
        case "xgboost":
            cls = XGBClassifier if classification else XGBRegressor
            return cls(random_state=seed, n_jobs=1, tree_method="hist", **params)
    raise ValueError("Unsupported model algorithm.")


def search_space(trial: optuna.Trial, config: ExperimentInput) -> Params:
    if config.algorithm == "logistic":
        return {"C": trial.suggest_float("C", 0.001, 100, log=True)}
    if config.algorithm == "torch_mlp":
        return {
            "hidden": trial.suggest_int("hidden", 16, 64, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.02, log=True),
        }
    params: Params = {
        "n_estimators": trial.suggest_int("n_estimators", 30, 150, step=30),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
    }
    if config.algorithm == "random_forest":
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 8)
    else:
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.3, log=True)
    return params
