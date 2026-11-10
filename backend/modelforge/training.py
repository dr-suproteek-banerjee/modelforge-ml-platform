from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .drift import baseline_for
from .evaluation import evaluate
from .preprocessing import Split, build_preprocessor, split_dataset
from .schemas import ExperimentInput, Metrics, Params
from .strategies import estimator, search_space


class TrainingCancelled(Exception):
    pass


@dataclass
class TrainingResult:
    pipeline: Pipeline
    split: Split
    metrics: Metrics
    baseline: dict


def train(
    frame: pd.DataFrame, config: ExperimentInput, progress: Callable[[int], None]
) -> TrainingResult:
    split = split_dataset(frame, config)
    progress(12)
    classification = config.task == "classification"
    fold_type = StratifiedKFold if classification else KFold
    folds = fold_type(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    scoring = "f1_macro" if classification else "neg_root_mean_squared_error"

    def build(params: Params) -> Pipeline:
        return Pipeline(
            [
                ("preprocess", build_preprocessor(split.schema, config.preprocessing)),
                ("estimate", estimator(config, params)),
            ]
        )

    def validate(params: Params, stage: int) -> np.ndarray:
        scores = []
        for train_idx, validation_idx in folds.split(split.train, split.y_train):
            progress(stage)
            scores.append(
                float(
                    cross_val_score(
                        build(params),
                        split.train,
                        split.y_train,
                        scoring=scoring,
                        cv=[(train_idx, validation_idx)],
                        error_score="raise",
                    )[0]
                )
            )
        return np.asarray(scores)

    params: Params = {}
    if config.trials:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.seed),
        )
        scores_by_trial: dict[int, np.ndarray] = {}

        def objective(trial: optuna.Trial) -> float:
            scores = validate(search_space(trial, config), 15 + 55 * trial.number // config.trials)
            scores_by_trial[trial.number] = scores
            return float(scores.mean())

        study.optimize(objective, n_trials=config.trials, n_jobs=1)
        params = study.best_params
        scores = scores_by_trial[study.best_trial.number]
    else:
        scores = validate(params, 35)

    progress(75)
    pipeline = build(params)
    pipeline.fit(split.train, split.y_train)
    progress(85)
    sign = 1 if classification else -1
    key = "f1_macro" if classification else "rmse"
    cv = {f"{key}_mean": sign * float(scores.mean()), f"{key}_std": float(scores.std())}
    metrics = evaluate(pipeline, split, config, cv, params)
    baseline = baseline_for(split.train, split.schema, config.seed)
    return TrainingResult(pipeline, split, metrics, baseline)
