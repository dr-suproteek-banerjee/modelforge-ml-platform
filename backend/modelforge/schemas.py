from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Task = Literal["classification", "regression"]
Algorithm = Literal[
    "logistic", "linear", "random_forest", "gradient_boosting", "xgboost", "torch_mlp"
]
Status = Literal["draft", "queued", "running", "completed", "failed", "cancelled"]
Cell = str | float | int | bool | None
Params = dict[str, float | int]


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Preprocessing(StrictInput):
    numeric_imputer: Literal["median", "mean", "most_frequent"] = "median"
    categorical_imputer: Literal["most_frequent", "constant"] = "most_frequent"
    scale: bool = True
    exclude: list[str] = Field(default_factory=list, max_length=100)


class ExperimentInput(StrictInput):
    name: str = Field(min_length=1, max_length=120)
    dataset_id: str = Field(min_length=1, max_length=36)
    target: str = Field(min_length=1, max_length=120)
    task: Task
    algorithm: Algorithm
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    cv_folds: int = Field(default=3, ge=2, le=8)
    trials: int = Field(default=0, ge=0, le=30)
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)

    @model_validator(mode="after")
    def compatible(self) -> "ExperimentInput":
        if (self.algorithm == "linear" and self.task != "regression") or (
            self.algorithm == "logistic" and self.task != "classification"
        ):
            raise ValueError("Choose a model compatible with the task.")
        if self.algorithm == "linear" and self.trials:
            raise ValueError("Ordinary least squares has no search space. Set trials to zero.")
        if self.target in self.preprocessing.exclude:
            raise ValueError("The target cannot appear in excluded features.")
        return self


class DatasetView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    version: int
    rows: int
    columns: list[str]
    sha256: str
    created_at: datetime


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing: int
    unique: int
    statistics: dict[str, float]
    distribution: list[dict[str, str | int]]


class ProfileView(BaseModel):
    rows: int
    columns: list[ColumnProfile]
    preview: list[dict[str, Cell]]
    correlations: dict[str, dict[str, float | None]]


class Metrics(BaseModel):
    scores: dict[str, float]
    cross_validation: dict[str, float]
    labels: list[str] = Field(default_factory=list)
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    classification_report: dict[str, dict[str, float]] = Field(default_factory=dict)
    residuals: list[dict[str, float]] = Field(default_factory=list)
    feature_importance: list[dict[str, str | float]]
    best_parameters: Params


class ExperimentView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    config: ExperimentInput
    status: Status
    progress: int
    error: str | None
    metrics: Metrics | None
    mlflow_run: str | None
    created_at: datetime


class ModelMetadata(BaseModel):
    format: int
    sha256: str
    schema_: dict[str, str] = Field(validation_alias="schema", serialization_alias="schema")
    labels: list[str]
    config: ExperimentInput
    dataset_version: int
    dataset_sha256: str
    python: str
    libraries: dict[str, str]
    best_parameters: Params
    train_rows: int
    holdout_rows: int


class ModelView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    experiment_id: str
    name: str
    version: int
    artifact_key: str
    metadata_: ModelMetadata = Field(serialization_alias="metadata")
    created_at: datetime


class PredictionInput(StrictInput):
    rows: list[dict[str, Cell]] = Field(min_length=1, max_length=256)


class PredictionView(BaseModel):
    model_id: str
    predictions: list[str | float]
    probabilities: list[dict[str, float]] | None = None
    latency_ms: float
    performance: dict[str, float] | None = None


class DriftFeature(BaseModel):
    feature: str
    method: str
    statistic: float
    p_value: float | None = None
    adjusted_p_value: float | None = None
    flagged: bool
    baseline_missing: float
    current_missing: float
    baseline: dict[str, int] = Field(default_factory=dict)
    current: dict[str, int] = Field(default_factory=dict)


class DriftView(BaseModel):
    model_id: str
    rows: int
    features: list[DriftFeature]


class ActivityView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    action: str
    resource_id: str
    details: dict
    created_at: datetime


PageSize = Annotated[int, Field(ge=1, le=200)]
