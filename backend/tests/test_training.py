from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from modelforge import registry
from modelforge.database import Dataset, Model
from modelforge.datasets import profile, read_csv
from modelforge.neural import torch_available
from modelforge.preprocessing import build_preprocessor, split_dataset
from modelforge.schemas import ExperimentInput, Preprocessing
from modelforge.storage import LocalArtifacts
from modelforge.training import train


def config(task="classification", algorithm="logistic", **kwargs):
    return ExperimentInput(
        name="benchmark",
        dataset_id="fixture",
        task=task,
        algorithm=algorithm,
        target="species" if task == "classification" else "price",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("task", "algorithm", "trials"),
    [
        ("classification", "logistic", 3),
        ("classification", "random_forest", 0),
        ("classification", "gradient_boosting", 0),
        ("classification", "xgboost", 2),
        ("regression", "linear", 0),
        ("regression", "random_forest", 0),
        ("regression", "gradient_boosting", 0),
        ("regression", "xgboost", 0),
        pytest.param(
            "classification",
            "torch_mlp",
            0,
            marks=pytest.mark.skipif(not torch_available(), reason="optional torch extra"),
        ),
        pytest.param(
            "regression",
            "torch_mlp",
            0,
            marks=pytest.mark.skipif(not torch_available(), reason="optional torch extra"),
        ),
    ],
)
def test_heldout_quality_and_artifact_round_trip(
    task, algorithm, trials, iris_bytes, regression_bytes, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(registry, "store", LocalArtifacts(tmp_path))
    data = iris_bytes if task == "classification" else regression_bytes
    frame = read_csv(data)
    spec = config(task, algorithm, trials=trials)
    stages = []
    result = train(frame, spec, stages.append)
    assert sorted(stages) == stages
    assert bool(result.metrics.best_parameters) == bool(trials)
    metrics = result.metrics
    if task == "classification":
        assert metrics.scores["accuracy"] >= 0.85
        assert 0.85 <= metrics.scores["f1_macro"] <= 1
        assert np.asarray(metrics.confusion_matrix).sum() == len(result.split.test)
        assert set(metrics.classification_report) == set(result.split.labels)
        assert metrics.cross_validation["f1_macro_mean"] >= 0.80
    else:
        assert metrics.scores["r2"] > 0.85
        assert metrics.scores["rmse"] > metrics.scores["mae"] > 0
        assert len(metrics.residuals) == len(result.split.test)
        assert metrics.cross_validation["rmse_mean"] > 0
    assert {x["feature"] for x in metrics.feature_importance} == set(result.split.schema)
    key = "models/test.zip"
    metadata = registry.save_model(key, result, spec, Dataset(version=1, sha256="fixture"))
    model = Model(id="fixture", artifact_key=key, metadata_=metadata)
    restored = registry.load_model(model)
    np.testing.assert_allclose(
        restored.predict(result.split.test), result.pipeline.predict(result.split.test)
    )
    metadata["libraries"]["scikit-learn"] = "incompatible"
    with pytest.raises(ValueError, match="original locked"):
        registry.load_model(model)
    metadata["libraries"] = registry.library_versions(algorithm)
    (tmp_path / key).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        registry.load_model(model)


def test_preprocessor_and_holdout_have_no_fitted_state_leakage(regression_bytes):
    frame = read_csv(regression_bytes)
    spec = config("regression", "linear")
    baseline = split_dataset(frame, spec)
    frame.loc[baseline.test.index, "area"] = 10**8
    frame.loc[baseline.test.index, "district"] = "holdout_only"
    split = split_dataset(frame, spec)
    assert not set(split.train.index) & set(split.test.index)
    assert sorted([*split.train.index, *split.test.index]) == list(frame.index)
    assert "price" not in split.schema
    fitted = build_preprocessor(split.schema, spec.preprocessing).fit(split.train)
    imputer = fitted.named_transformers_["numeric"]["impute"]
    assert imputer.statistics_[0] == pytest.approx(split.train.area.median())
    assert imputer.statistics_[0] != pytest.approx(frame.area.median())
    encoded = fitted.named_transformers_["categorical"]["encode"]
    assert "holdout_only" not in encoded.categories_[0]
    assert np.isfinite(fitted.transform(split.test)).all()


def test_preprocessing_imputes_missing_and_accepts_unseen_categories():
    data = pd.DataFrame({"numeric": [1, np.nan, 5], "category": ["x", np.nan, "y"]})
    encoder = build_preprocessor(
        {"numeric": "numeric", "category": "categorical"},
        Preprocessing(categorical_imputer="constant", scale=False),
    )
    encoded = encoder.fit_transform(data)
    assert encoded[1, 0] == 3
    unseen = encoder.transform(pd.DataFrame({"numeric": [None], "category": ["new"]}))
    assert unseen[0, 0] == 3
    assert np.sum(unseen[0, 1:]) == 0


def test_csv_schema_and_profile(iris_bytes):
    with pytest.raises(ValueError, match="unique"):
        read_csv(b"x,x\n1,2\n")
    with pytest.raises(ValueError, match="match the header"):
        read_csv(b"x,y\n1,2,3\n")
    data = read_csv(b"x,y,label\n1,,blue\n2,3,NA\n")
    report = profile(data)
    assert report.preview[0]["y"] is None
    assert report.preview[1]["label"] == "NA"
    assert report.correlations["x"]["y"] is None
    assert profile(read_csv(iris_bytes)).rows == 150


def test_configuration_is_allowlisted(iris_bytes):
    with pytest.raises(ValidationError, match="compatible"):
        config("classification", "linear")
    with pytest.raises(ValidationError, match="search space"):
        config("regression", "linear", trials=2)
    with pytest.raises(ValidationError):
        Preprocessing(numeric_imputer="lambda x: x")
    with pytest.raises(ValueError, match="Unknown excluded"):
        split_dataset(
            read_csv(iris_bytes), config(preprocessing=Preprocessing(exclude=["unknown"]))
        )
