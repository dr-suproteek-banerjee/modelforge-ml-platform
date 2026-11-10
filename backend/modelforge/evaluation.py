import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from .preprocessing import Split
from .schemas import ExperimentInput, Metrics, Params


def evaluate(
    pipeline: Pipeline, split: Split, config: ExperimentInput, cv: dict[str, float], params: Params
) -> Metrics:
    prediction = pipeline.predict(split.test)
    y_test = split.y_test.to_numpy()
    labels = list(range(len(split.labels)))
    scorer = "f1_macro" if labels else "neg_root_mean_squared_error"
    importance = permutation_importance(
        pipeline,
        split.test,
        y_test,
        n_repeats=3,
        random_state=config.seed,
        scoring=scorer,
        max_samples=min(1000, len(split.test)),
    )
    metrics = Metrics(
        scores={},
        cross_validation=cv,
        labels=split.labels,
        best_parameters=params,
        feature_importance=[
            {"feature": name, "importance": float(score)}
            for name, score in sorted(
                zip(split.schema, importance.importances_mean, strict=True),
                key=lambda pair: -pair[1],
            )
        ],
    )
    if labels:
        metrics.scores = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y_test, pipeline.predict_proba(split.test), labels=labels)),
        }
        metrics.confusion_matrix = confusion_matrix(y_test, prediction, labels=labels).tolist()
        report = classification_report(
            y_test,
            prediction,
            labels=labels,
            target_names=[f"class:{x}" for x in labels],
            output_dict=True,
            zero_division=0,
        )
        metrics.classification_report = {
            (split.labels[int(k[6:])] if k.startswith("class:") else k): v
            for k, v in report.items()
            if isinstance(v, dict)
            if k.startswith("class:")
        }
    else:
        metrics.scores = {
            "rmse": float(root_mean_squared_error(y_test, prediction)),
            "mae": float(mean_absolute_error(y_test, prediction)),
            "r2": float(r2_score(y_test, prediction)),
        }
        indices = np.linspace(0, len(y_test) - 1, min(250, len(y_test)), dtype=int)
        metrics.residuals = [
            {
                "actual": float(y_test[i]),
                "prediction": float(prediction[i]),
                "residual": float(y_test[i] - prediction[i]),
            }
            for i in indices
        ]
    return metrics
