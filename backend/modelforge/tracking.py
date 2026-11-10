import logging

from mlflow.tracking import MlflowClient

from .schemas import ExperimentInput, Metrics
from .settings import settings

logger = logging.getLogger(__name__)


def log_run(experiment_id: str, config: ExperimentInput, metrics: Metrics) -> str | None:
    try:
        client = MlflowClient(tracking_uri=settings.tracking_uri)
        run = client.create_run(
            "0", tags={"mlflow.runName": config.name, "modelforge.id": experiment_id}
        )
        run_id = run.info.run_id
        for key, value in config.model_dump(exclude={"preprocessing"}).items():
            client.log_param(run_id, key, value)
        for key, value in metrics.best_parameters.items():
            client.log_param(run_id, f"optimized.{key}", value)
        for key, value in {**metrics.scores, **metrics.cross_validation}.items():
            client.log_metric(run_id, key, value)
        client.set_terminated(run_id)
        return run_id
    except Exception:
        logger.exception("MLflow tracking failed for experiment %s", experiment_id)
        return None
