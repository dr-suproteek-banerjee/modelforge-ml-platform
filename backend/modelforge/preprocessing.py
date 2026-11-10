from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schemas import ExperimentInput, Preprocessing


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    schema: dict[str, str]
    labels: list[str]


def split_dataset(frame: pd.DataFrame, config: ExperimentInput) -> Split:
    if config.target not in frame:
        raise ValueError("Target column does not exist in this dataset.")
    unknown = set(config.preprocessing.exclude) - set(frame.columns)
    if unknown:
        raise ValueError(f"Unknown excluded columns: {', '.join(sorted(unknown))}.")
    features = frame.drop(columns=[config.target, *config.preprocessing.exclude])
    if features.shape[1] == 0 or len(features) < 20:
        raise ValueError("Training requires at least one feature and 20 rows.")
    target = frame[config.target]
    if target.isna().any():
        raise ValueError("Target contains missing values.")
    labels: list[str] = []
    if config.task == "classification":
        labels = sorted(target.astype(str).unique().tolist())
        if not 2 <= len(labels) <= 30 or target.value_counts().min() < 2:
            raise ValueError(
                "Classification requires 2–30 classes, with at least two examples per class."
            )
        target = target.astype(str).map({label: index for index, label in enumerate(labels)})
    else:
        if not pd.api.types.is_numeric_dtype(target) or target.nunique() < 2:
            raise ValueError(
                "Regression requires a numeric target with at least two distinct values."
            )
        target = target.astype(float)
    schema = {
        str(name): "numeric"
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)
        else "categorical"
        for name, column in features.items()
    }
    features = normalize(features, schema)
    try:
        train, test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=config.test_size,
            random_state=config.seed,
            stratify=target if labels else None,
        )
    except ValueError as exc:
        raise ValueError(
            "The holdout cannot represent all classes. Increase the holdout or add samples."
        ) from exc
    if labels and (
        y_train.nunique() != len(labels)
        or y_test.nunique() != len(labels)
        or y_train.value_counts().min() < config.cv_folds
    ):
        raise ValueError(
            "Every class must appear in holdout and in each training fold. Reduce folds or add data."
        )
    if not labels and len(train) < 2 * config.cv_folds:
        raise ValueError(
            "Reduce the number of folds or add data so each fold has at least two rows."
        )
    return Split(train, test, y_train, y_test, schema, labels)


def normalize(frame: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for name, kind in schema.items():
        source = frame[name]
        if kind == "numeric":
            if source.map(lambda value: isinstance(value, (bool, np.bool_))).any():
                raise ValueError(f"Feature {name!r} requires numeric values.")
            values = pd.to_numeric(source, errors="coerce")
            present = values.dropna().to_numpy(dtype=float)
            if (
                (source.notna() & values.isna()).any()
                or not np.isfinite(present).all()
                or (np.abs(present) > 1e30).any()
            ):
                raise ValueError(f"Feature {name!r} requires finite numeric values up to 1e30.")
            result[name] = values.astype(float)
        else:
            result[name] = source.map(
                lambda value: str(value) if pd.notna(value) else np.nan
            ).astype(object)
    return result


def build_preprocessor(schema: dict[str, str], config: Preprocessing) -> ColumnTransformer:
    numeric = [key for key, kind in schema.items() if kind == "numeric"]
    categorical = [key for key, kind in schema.items() if kind == "categorical"]
    numeric_steps = [
        ("impute", SimpleImputer(strategy=config.numeric_imputer, keep_empty_features=True))
    ]
    if config.scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy=config.categorical_imputer,
                                fill_value="__missing__",
                                keep_empty_features=True,
                            ),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=24)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )
