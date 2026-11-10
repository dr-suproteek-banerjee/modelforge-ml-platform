import logging

from billiard.exceptions import SoftTimeLimitExceeded
from celery import Celery
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import Dataset, Experiment, Model, SessionLocal, new_id, record
from .datasets import read_csv
from .registry import load_model, save_model
from .schemas import ExperimentInput
from .settings import settings
from .storage import store
from .tracking import log_run
from .training import TrainingCancelled, train

logger = logging.getLogger(__name__)
celery_app = Celery("modelforge", broker=settings.redis_url)
celery_app.conf.update(
    task_always_eager=settings.eager,
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    task_soft_time_limit=840,
    task_time_limit=900,
    worker_prefetch_multiplier=1,
    broker_connection_timeout=3,
    task_publish_retry=False,
)


def transition(
    session: Session,
    identifier: str,
    expected: list[str],
    *,
    job: str | None = None,
    **fields: object,
) -> bool:
    statement = update(Experiment).where(
        Experiment.id == identifier, Experiment.status.in_(expected)
    )
    if job is not None:
        statement = statement.where(Experiment.job_id == job)
    changed = session.execute(
        statement.values(**fields).returning(Experiment.id)
    ).scalar_one_or_none()
    return changed is not None


@celery_app.task(name="modelforge.train")
def train_job(identifier: str, job: str) -> None:
    with SessionLocal() as session:
        if not transition(session, identifier, ["queued"], job=job, status="running", progress=5):
            return
        record(session, "training_started", identifier)
        session.commit()
        experiment = session.get_one(Experiment, identifier)
        dataset = session.get_one(Dataset, experiment.dataset_id)
        config = ExperimentInput.model_validate(experiment.config)
        session.commit()

    def progress(percent: int) -> None:
        with SessionLocal() as session:
            if not transition(session, identifier, ["running"], job=job, progress=percent):
                raise TrainingCancelled
            session.commit()

    model_id = new_id()
    key = f"models/{model_id}.zip"
    published = False
    try:
        frame = read_csv(store.get(dataset.artifact_key, dataset.sha256))
        result = train(frame, config, progress)
        progress(92)
        metadata = save_model(key, result, config, dataset)
        # Verify that the produced model can be read before publishing its registry entry.
        candidate = Model(id=model_id, artifact_key=key, metadata_=metadata)
        load_model(candidate).predict(result.split.test.head(1))
        progress(96)
        with SessionLocal() as session:
            for attempt in range(5):
                try:
                    if not transition(
                        session,
                        identifier,
                        ["running"],
                        job=job,
                        status="completed",
                        progress=100,
                        metrics=result.metrics.model_dump(),
                    ):
                        raise TrainingCancelled
                    latest = session.scalar(
                        select(func.max(Model.version)).where(Model.name == config.name)
                    )
                    session.add(
                        Model(
                            id=model_id,
                            experiment_id=identifier,
                            name=config.name,
                            version=(latest or 0) + 1,
                            artifact_key=key,
                            metadata_=metadata,
                        )
                    )
                    record(session, "model_registered", model_id, experiment_id=identifier)
                    session.commit()
                    published = True
                    break
                except IntegrityError:
                    session.rollback()
                    if attempt == 4:
                        raise
        run_id = log_run(identifier, config, result.metrics)
        with SessionLocal() as session:
            transition(session, identifier, ["completed"], job=job, mlflow_run=run_id)
            if run_id is None:
                record(session, "tracking_failed", identifier)
            session.commit()
    except TrainingCancelled:
        return
    except Exception as exc:
        logger.exception("Training job failed for %s", identifier)
        if isinstance(exc, SoftTimeLimitExceeded):
            message = "Training exceeded the time limit. Reduce dataset size, folds, or trials."
        elif isinstance(exc, ValueError):
            message = str(exc).splitlines()[0][:350]
        else:
            message = "Training failed. Check the worker logs for this experiment ID."
        with SessionLocal() as session:
            if transition(
                session, identifier, ["running"], job=job, status="failed", error=message
            ):
                record(session, "training_failed", identifier)
            session.commit()
    finally:
        if not published:
            store.delete(key)
