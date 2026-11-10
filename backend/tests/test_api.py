from sqlalchemy import select

PREFIX = "/api/v1"


def upload(client, data, name="Iris"):
    result = client.post(
        f"{PREFIX}/datasets", data={"name": name}, files={"file": ("iris.csv", data)}
    )
    assert result.status_code == 201, result.text
    return result.json()


def create(client, dataset_id, **overrides):
    return client.post(
        f"{PREFIX}/experiments",
        json={
            "name": "Iris test",
            "dataset_id": dataset_id,
            "target": "species",
            "task": "classification",
            "algorithm": "logistic",
            **overrides,
        },
    )


def test_api_end_to_end(client, iris_bytes, iris_batch):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/openapi.json").status_code == 200
    first = upload(client, iris_bytes)
    second = upload(client, iris_bytes)
    assert second["version"] == first["version"] + 1
    assert second["sha256"] == first["sha256"]
    profile = client.get(f"{PREFIX}/datasets/{first['id']}/profile").json()
    assert profile["rows"] == 150
    assert "species" in [column["name"] for column in profile["columns"]]
    assert create(client, first["id"], target="missing").status_code == 422
    draft = create(client, first["id"], trials=2)
    assert draft.status_code == 201, draft.text
    identifier = draft.json()["id"]
    path = f"{PREFIX}/experiments/{identifier}"
    assert client.get(f"{path}/metrics").status_code == 409
    run = client.post(f"{path}/train")
    assert run.status_code == 202
    assert run.json()["status"] == "completed", run.json()
    assert run.json()["mlflow_run"]
    assert client.post(f"{path}/train").status_code == 409
    assert client.get(f"{path}/metrics").json()["scores"]["accuracy"] >= 0.85
    models = client.get(f"{PREFIX}/models").json()
    model = models[0]
    assert model["metadata"]["dataset_version"] == 1
    endpoint = f"{PREFIX}/models/{model['id']}"
    sample = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    prediction = client.post(f"{endpoint}/predict", json={"rows": [sample]})
    assert prediction.status_code == 200, prediction.text
    assert prediction.json()["predictions"] == ["setosa"]
    assert 0.999 < sum(prediction.json()["probabilities"][0].values()) < 1.001
    assert (
        client.post(f"{endpoint}/predict", json={"rows": [{"petal_width": 0.2}]}).status_code == 422
    )
    assert (
        client.post(
            f"{endpoint}/predict", json={"rows": [{**sample, "sepal_length": True}]}
        ).status_code
        == 422
    )
    batch = client.post(f"{endpoint}/batch-predict", files={"file": ("data.csv", iris_bytes)})
    assert batch.status_code == 200, batch.text
    assert len(batch.json()["predictions"]) == 150
    assert batch.json()["performance"]["accuracy"] > 0.85
    drift = client.post(f"{endpoint}/drift", files={"file": ("batch.csv", iris_batch)})
    assert drift.status_code == 200, drift.text
    rows = {row["feature"]: row for row in drift.json()["features"]}
    assert rows["sepal_length"]["flagged"]
    assert not rows["petal_width"]["flagged"]
    assert client.get(f"{PREFIX}/activity").json()[0]["action"] == "drift_measured"


def test_cancel_and_retry_ignore_stale_tasks(client, iris_bytes, monkeypatch):
    from modelforge import jobs
    from modelforge.database import Experiment, SessionLocal

    submitted = []
    monkeypatch.setattr(jobs.train_job, "delay", lambda *args: submitted.append(args))
    identifier = create(client, upload(client, iris_bytes)["id"]).json()["id"]
    base = f"{PREFIX}/experiments/{identifier}"
    assert client.post(f"{base}/train").json()["status"] == "queued"
    assert client.post(f"{base}/cancel").json()["status"] == "cancelled"
    assert client.post(f"{base}/train").json()["status"] == "queued"
    jobs.train_job.run(*submitted[0])
    assert client.get(base).json()["status"] == "queued"
    assert client.post(f"{base}/cancel").status_code == 200
    jobs.train_job.run(*submitted[1])
    assert client.get(base).json()["status"] == "cancelled"
    with SessionLocal() as session:
        assert session.scalar(select(Experiment.job_id)) == submitted[1][1]
    assert client.get(f"{PREFIX}/models").json() == []


def test_reject_cross_site_writes(client):
    result = client.post(f"{PREFIX}/datasets", headers={"sec-fetch-site": "cross-site"})
    assert result.status_code == 403
