from typing import Annotated, TypeVar
from zipfile import BadZipFile

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from skops.io.exceptions import UntrustedTypesFoundException
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from .database import Activity, Dataset, Experiment, Model, db_session, new_id, record
from .datasets import profile, read_csv
from .drift import compare
from .jobs import train_job, transition
from .neural import torch_available
from .preprocessing import split_dataset
from .registry import predict_model, prepare_inference
from .schemas import (
    ActivityView,
    DatasetView,
    DriftView,
    ExperimentInput,
    ExperimentView,
    Metrics,
    ModelView,
    PageSize,
    PredictionInput,
    PredictionView,
    ProfileView,
)
from .settings import settings
from .storage import store

app = FastAPI(title="ModelForge", version="0.1.0")
DB = Annotated[Session, Depends(db_session)]
Entity = TypeVar("Entity", Dataset, Experiment, Model)


@app.middleware("http")
async def same_site_writes(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"} and (
        request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}
    ):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"detail": "Open ModelForge directly to submit changes."}, status_code=403
        )
    return await call_next(request)


def require(db: Session, entity: type[Entity], identifier: str) -> Entity:
    item = db.get(entity, identifier)
    if item is None:
        raise HTTPException(404, "Resource not found.")
    return item


def upload_bytes(file: UploadFile) -> bytes:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(422, "Select a .csv file.")
    data = file.file.read(settings.upload_limit + 1)
    if len(data) > settings.upload_limit:
        raise HTTPException(413, "CSV exceeds the upload limit.")
    return data


def csv_frame(data: bytes, *, max_rows: int = 25_000) -> pd.DataFrame:
    try:
        return read_csv(data, max_rows=max_rows)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def stored_frame(dataset: Dataset) -> pd.DataFrame:
    try:
        return read_csv(store.get(dataset.artifact_key, dataset.sha256))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            409, "Dataset artifact is missing or failed integrity verification."
        ) from exc


@app.get("/health")
def health(db: DB) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
        db.execute(select(Dataset.id).limit(1))
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database or migrations unavailable.") from exc
    return {"status": "ok"}


@app.get("/api/v1/capabilities")
def capabilities() -> dict[str, bool]:
    return {"torch_mlp": torch_available()}


@app.post("/api/v1/datasets", response_model=DatasetView, status_code=201)
def create_dataset(
    db: DB,
    file: Annotated[UploadFile, File()],
    name: Annotated[str, Form(min_length=1, max_length=120)],
) -> Dataset:
    name = name.strip()
    if not name or "\x00" in name:
        raise HTTPException(422, "Provide a nonblank dataset name.")
    data = upload_bytes(file)
    frame = csv_frame(data)
    identifier = new_id()
    key = f"datasets/{identifier}.csv"
    digest = store.put(key, data)
    persisted = False
    try:
        for attempt in range(5):
            latest = db.scalar(select(func.max(Dataset.version)).where(Dataset.name == name))
            dataset = Dataset(
                id=identifier,
                name=name,
                version=(latest or 0) + 1,
                rows=len(frame),
                columns=list(frame.columns),
                sha256=digest,
                artifact_key=key,
            )
            db.add(dataset)
            record(db, "dataset_uploaded", identifier, version=dataset.version, rows=len(frame))
            try:
                db.commit()
                persisted = True
                return dataset
            except IntegrityError as exc:
                db.rollback()
                if attempt == 4:
                    raise HTTPException(
                        409, "Dataset version was updated concurrently. Retry upload."
                    ) from exc
    finally:
        if not persisted:
            store.delete(key)
    raise RuntimeError("Version retry exhausted.")


@app.get("/api/v1/datasets", response_model=list[DatasetView])
def list_datasets(db: DB, limit: PageSize = 100) -> list[Dataset]:
    return list(db.scalars(select(Dataset).order_by(Dataset.created_at.desc()).limit(limit)))


@app.get("/api/v1/datasets/{identifier}/profile", response_model=ProfileView)
def dataset_profile(identifier: str, db: DB) -> ProfileView:
    return profile(stored_frame(require(db, Dataset, identifier)))


@app.post("/api/v1/experiments", response_model=ExperimentView, status_code=201)
def create_experiment(config: ExperimentInput, db: DB) -> Experiment:
    if config.algorithm == "torch_mlp" and not torch_available():
        raise HTTPException(422, "Enable the optional PyTorch dependency before using this model.")
    if not config.name.strip() or "\x00" in config.name:
        raise HTTPException(422, "Provide a nonblank experiment name.")
    config.name = config.name.strip()
    dataset = require(db, Dataset, config.dataset_id)
    try:
        split_dataset(stored_frame(dataset), config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    experiment = Experiment(id=new_id(), dataset_id=config.dataset_id, config=config.model_dump())
    db.add(experiment)
    record(db, "experiment_created", experiment.id)
    db.commit()
    return experiment


@app.get("/api/v1/experiments", response_model=list[ExperimentView])
def list_experiments(db: DB, limit: PageSize = 100) -> list[Experiment]:
    return list(db.scalars(select(Experiment).order_by(Experiment.created_at.desc()).limit(limit)))


@app.get("/api/v1/experiments/{identifier}", response_model=ExperimentView)
def get_experiment(identifier: str, db: DB) -> Experiment:
    return require(db, Experiment, identifier)


@app.post("/api/v1/experiments/{identifier}/train", response_model=ExperimentView, status_code=202)
def start_training(identifier: str, db: DB) -> Experiment:
    require(db, Experiment, identifier)
    job = new_id()
    if not transition(
        db,
        identifier,
        ["draft", "failed", "cancelled"],
        status="queued",
        job_id=job,
        progress=0,
        error=None,
    ):
        raise HTTPException(409, "Only draft, failed, or cancelled experiments can be queued.")
    record(db, "training_queued", identifier)
    db.commit()
    try:
        train_job.delay(identifier, job)
    except Exception as exc:
        if transition(
            db,
            identifier,
            ["queued"],
            job=job,
            status="failed",
            error="Training queue is unavailable.",
        ):
            record(db, "queue_unavailable", identifier)
        db.commit()
        raise HTTPException(
            503, "Training queue is unavailable. Retry when the worker is online."
        ) from exc
    db.expire_all()
    return require(db, Experiment, identifier)


@app.post("/api/v1/experiments/{identifier}/cancel", response_model=ExperimentView)
def cancel_training(identifier: str, db: DB) -> Experiment:
    require(db, Experiment, identifier)
    if not transition(db, identifier, ["queued", "running"], status="cancelled"):
        raise HTTPException(409, "Only queued or running experiments can be cancelled.")
    record(db, "training_cancelled", identifier)
    db.commit()
    db.expire_all()
    return require(db, Experiment, identifier)


@app.get("/api/v1/experiments/{identifier}/metrics", response_model=Metrics)
def metrics(identifier: str, db: DB) -> Metrics:
    value = require(db, Experiment, identifier).metrics
    if value is None:
        raise HTTPException(409, "Evaluation is available after training completes.")
    return Metrics.model_validate(value)


@app.get("/api/v1/models", response_model=list[ModelView])
def list_models(db: DB, limit: PageSize = 100) -> list[Model]:
    return list(db.scalars(select(Model).order_by(Model.created_at.desc()).limit(limit)))


def inference(model: Model, frame: pd.DataFrame, *, batch: bool) -> PredictionView:
    try:
        prepared = prepare_inference(model, frame, allow_target=batch)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        return predict_model(model, prepared, frame)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("Ground truth") or message.startswith("Ground-truth"):
            raise HTTPException(422, message) from exc
        raise HTTPException(
            409, "Model artifact is incompatible or failed integrity verification."
        ) from exc
    except (OSError, RuntimeError, BadZipFile, UntrustedTypesFoundException) as exc:
        raise HTTPException(409, "Model artifact is unavailable or incompatible.") from exc


@app.post("/api/v1/models/{identifier}/predict", response_model=PredictionView)
def predict(identifier: str, request: PredictionInput, db: DB) -> PredictionView:
    model = require(db, Model, identifier)
    expected = set(model.metadata_["schema"])
    if any(set(row) != expected for row in request.rows):
        raise HTTPException(
            422,
            "Every row must provide exactly the registered feature columns. Use null for missing values.",
        )
    if any(
        isinstance(value, str) and (len(value) > 4096 or "\x00" in value)
        for row in request.rows
        for value in row.values()
    ):
        raise HTTPException(
            422, "Text values may contain at most 4,096 characters and no null byte."
        )
    result = inference(model, pd.DataFrame(request.rows, dtype=object), batch=False)
    record(
        db,
        "online_predicted",
        identifier,
        rows=len(result.predictions),
        latency_ms=result.latency_ms,
    )
    db.commit()
    return result


@app.post("/api/v1/models/{identifier}/batch-predict", response_model=PredictionView)
def batch_predict(identifier: str, file: Annotated[UploadFile, File()], db: DB) -> PredictionView:
    model = require(db, Model, identifier)
    result = inference(model, csv_frame(upload_bytes(file), max_rows=5000), batch=True)
    record(
        db,
        "batch_predicted",
        identifier,
        rows=len(result.predictions),
        latency_ms=result.latency_ms,
        performance=result.performance,
    )
    db.commit()
    return result


@app.post("/api/v1/models/{identifier}/drift", response_model=DriftView)
def drift(identifier: str, file: Annotated[UploadFile, File()], db: DB) -> DriftView:
    model = require(db, Model, identifier)
    frame = csv_frame(upload_bytes(file), max_rows=5000)
    try:
        prepared = prepare_inference(model, frame, allow_target=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    features = compare(model.metadata_["baseline"], prepared)
    record(
        db,
        "drift_measured",
        identifier,
        rows=len(prepared),
        flagged=sum(x.flagged for x in features),
    )
    db.commit()
    return DriftView(model_id=identifier, rows=len(prepared), features=features)


@app.get("/api/v1/activity", response_model=list[ActivityView])
def activity(db: DB, limit: PageSize = 100, resource_id: str | None = None) -> list[Activity]:
    statement = select(Activity)
    if resource_id is not None:
        statement = statement.where(Activity.resource_id == resource_id)
    return list(db.scalars(statement.order_by(Activity.id.desc()).limit(limit)))
