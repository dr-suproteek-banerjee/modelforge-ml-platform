import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MF_EAGER", "true")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from sqlalchemy import create_engine

    from modelforge import api, database, registry, tracking
    from modelforge.storage import LocalArtifacts

    engine = create_engine(f"sqlite:///{tmp_path / 'workspace.db'}")
    database.Base.metadata.create_all(engine)
    monkeypatch.setattr(database.SessionLocal, "kw", {**database.SessionLocal.kw, "bind": engine})
    artifacts = LocalArtifacts(tmp_path / "artifacts")
    for module in (api, registry):
        monkeypatch.setattr(module, "store", artifacts)
    from modelforge import jobs

    monkeypatch.setattr(jobs, "store", artifacts)
    monkeypatch.setattr(tracking.settings, "tracking_uri", f"sqlite:///{tmp_path / 'tracking.db'}")
    with TestClient(api.app) as test:
        yield test
    engine.dispose()


@pytest.fixture()
def iris_bytes():
    return (SAMPLES / "iris.csv").read_bytes()


@pytest.fixture()
def regression_bytes():
    return (SAMPLES / "regression.csv").read_bytes()


@pytest.fixture()
def iris_batch():
    return (SAMPLES / "iris_shifted.csv").read_bytes()
