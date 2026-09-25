import os
os.environ["MODEL_BACKEND"] = "dummy"

from fastapi.testclient import TestClient
from app.main import app
from app.model import LABELS

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_all_labels():
    r = client.post("/predict", json={"texts": ["I am so happy today"]})
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert set(pred["scores"]) == set(LABELS)
    assert "joy" in pred["labels"]


def test_scores_are_probabilities():
    r = client.post("/predict", json={"texts": ["hello"]})
    for v in r.json()["predictions"][0]["scores"].values():
        assert 0.0 <= v <= 1.0


def test_batch_keeps_order():
    texts = ["first", "second", "third"]
    r = client.post("/predict", json={"texts": texts})
    assert [p["text"] for p in r.json()["predictions"]] == texts


def test_threshold_one_gives_no_labels():
    r = client.post("/predict", json={"texts": ["I love this"], "threshold": 1.0})
    assert r.json()["predictions"][0]["labels"] == []


def test_empty_input_rejected():
    assert client.post("/predict", json={"texts": []}).status_code == 422


def test_bad_threshold_rejected():
    assert client.post("/predict", json={"texts": ["x"], "threshold": 2}).status_code == 422
