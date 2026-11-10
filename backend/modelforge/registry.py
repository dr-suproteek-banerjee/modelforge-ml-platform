import platform
from importlib.metadata import version
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.pipeline import Pipeline

from .database import Dataset, Model
from .preprocessing import normalize
from .schemas import ExperimentInput, PredictionView
from .storage import store
from .strategies import estimator
from .training import TrainingResult

# This list belongs to the application, never to a request or a serialized file.
TRUSTED_SKLEARN_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def library_versions(algorithm: str) -> dict[str, str]:
    result = {name: version(name) for name in ("scikit-learn", "numpy", "scipy", "pandas", "skops")}
    if algorithm == "xgboost":
        import xgboost

        result["xgboost"] = xgboost.__version__
    if algorithm == "torch_mlp":
        result["torch"] = version("torch")
    return result


def save_model(key: str, result: TrainingResult, config: ExperimentInput, dataset: Dataset) -> dict:
    archive = BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as package:
        package.writestr("preprocess.skops", sio.dumps(result.pipeline["preprocess"]))
        model = result.pipeline["estimate"]
        if config.algorithm == "xgboost":
            package.writestr("estimator.ubj", bytes(model.get_booster().save_raw(raw_format="ubj")))
        elif config.algorithm == "torch_mlp":
            package.writestr("estimator.npz", model.export_weights())
        else:
            package.writestr("estimator.skops", sio.dumps(model))
    checksum = store.put(key, archive.getvalue())
    return {
        "format": 1,
        "sha256": checksum,
        "schema": result.split.schema,
        "labels": result.split.labels,
        "config": config.model_dump(),
        "baseline": result.baseline,
        "dataset_version": dataset.version,
        "dataset_sha256": dataset.sha256,
        "python": platform.python_version(),
        "libraries": library_versions(config.algorithm),
        "best_parameters": result.metrics.best_parameters,
        "train_rows": len(result.split.train),
        "holdout_rows": len(result.split.test),
    }


def load_model(model: Model) -> Pipeline:
    metadata = model.metadata_
    config = ExperimentInput.model_validate(metadata["config"])
    if metadata["format"] != 1 or metadata["libraries"] != library_versions(config.algorithm):
        raise ValueError("Artifact requires the original locked library versions.")
    with ZipFile(BytesIO(store.get(model.artifact_key, metadata["sha256"]))) as package:
        transform = sio.loads(package.read("preprocess.skops"), trusted=TRUSTED_SKLEARN_TYPES)
        if config.algorithm == "xgboost":
            predict = estimator(config, metadata["best_parameters"])
            predict.load_model(bytearray(package.read("estimator.ubj")))
        elif config.algorithm == "torch_mlp":
            predict = estimator(config, metadata["best_parameters"])
            predict.import_weights(package.read("estimator.npz"), len(metadata["labels"]))
        else:
            predict = sio.loads(package.read("estimator.skops"), trusted=TRUSTED_SKLEARN_TYPES)
    return Pipeline([("preprocess", transform), ("estimate", predict)])


def prepare_inference(model: Model, frame: pd.DataFrame, *, allow_target: bool) -> pd.DataFrame:
    schema = model.metadata_["schema"]
    config = model.metadata_["config"]
    allowed = set(schema)
    if allow_target:
        allowed.update([config["target"], *config["preprocessing"]["exclude"]])
    extra = set(frame.columns) - allowed
    missing = set(schema) - set(frame.columns)
    if missing or extra:
        raise ValueError(
            f"Schema mismatch. Missing: {', '.join(sorted(missing)) or 'none'}. "
            f"Unexpected: {', '.join(sorted(extra)) or 'none'}."
        )
    return normalize(frame, schema)


def predict_model(model: Model, frame: pd.DataFrame, original: pd.DataFrame) -> PredictionView:
    from time import perf_counter

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        root_mean_squared_error,
    )

    start = perf_counter()
    pipeline = load_model(model)
    predictions = pipeline.predict(frame)
    labels = model.metadata_["labels"]
    target = model.metadata_["config"]["target"]
    probabilities = None
    performance = None
    decoded: list[str | float]
    if labels:
        decoded = [labels[int(index)] for index in predictions]
        probabilities = [
            dict(zip(labels, map(float, row), strict=True)) for row in pipeline.predict_proba(frame)
        ]
    else:
        if not np.isfinite(predictions).all():
            raise ValueError("Model produced a non-finite prediction for this input.")
        decoded = list(map(float, predictions))
    if target in original:
        truth = original[target]
        if truth.isna().any():
            raise ValueError("Ground-truth target contains missing values.")
        if labels:
            actual = truth.astype(str)
            if not set(actual).issubset(labels):
                raise ValueError("Ground truth includes a class unknown to this model.")
            performance = {
                "accuracy": float(accuracy_score(actual, decoded)),
                "f1_macro": float(
                    f1_score(actual, decoded, labels=labels, average="macro", zero_division=0)
                ),
            }
        else:
            numeric = pd.to_numeric(truth, errors="coerce")
            if not np.isfinite(numeric).all() or (np.abs(numeric) > 1e30).any():
                raise ValueError("Ground truth requires finite numeric values up to 1e30.")
            performance = {
                "mae": float(mean_absolute_error(numeric, predictions)),
                "rmse": float(root_mean_squared_error(numeric, predictions)),
            }
    return PredictionView(
        model_id=model.id,
        predictions=decoded,
        probabilities=probabilities,
        latency_ms=(perf_counter() - start) * 1000,
        performance=performance,
    )
