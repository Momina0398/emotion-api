import os
os.environ.setdefault("MODEL_BACKEND", "dummy")

import pytest
from fastapi.testclient import TestClient
import app.main as main
from app.retrieval import HashEmbedder, Retriever

RECORDS = [
    {"text": "i am so happy and joyful today", "lang": "eng", "labels": {"joy": 1}},
    {"text": "happy happy day full of joy", "lang": "eng", "labels": {"joy": 1}},
    {"text": "this makes me so angry and furious", "lang": "eng", "labels": {"anger": 1}},
    {"text": "i am scared of the dark night", "lang": "eng", "labels": {"fear": 1}},
    {"text": "happy news from home", "lang": "hin", "labels": {"joy": 1, "surprise": 1}},
]


@pytest.fixture
def retriever():
    return Retriever.build(HashEmbedder(), RECORDS)


def test_nearest_is_most_similar(retriever):
    hits = retriever.search("so happy today", k=2)
    assert hits[0]["labels"].get("joy") == 1
    assert hits[0]["score"] >= hits[1]["score"]


def test_ranks_are_ordered(retriever):
    hits = retriever.search("angry", k=3)
    assert [h["rank"] for h in hits] == [1, 2, 3]


def test_language_filter(retriever):
    hits = retriever.search("happy", k=3, lang="hin")
    assert hits and all(h["lang"] == "hin" for h in hits)


def test_vote_between_zero_and_one(retriever):
    vote = Retriever.vote(retriever.search("happy joy", k=3))
    assert all(0.0 <= v <= 1.0 for v in vote.values())
    assert vote["joy"] > vote["anger"]


def test_save_and_load_roundtrip(retriever, tmp_path):
    retriever.save(tmp_path)
    loaded = Retriever.load(tmp_path, HashEmbedder())
    assert loaded.search("angry furious", k=1)[0]["text"] == RECORDS[2]["text"]


def test_similar_endpoint(retriever, monkeypatch):
    monkeypatch.setattr(main, "retriever", retriever)
    r = TestClient(main.app).post("/similar", json={"text": "so happy", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["neighbours"]) == 2 and "joy" in body["vote_labels"]


def test_similar_without_index_returns_503(monkeypatch):
    monkeypatch.setattr(main, "retriever", None)
    r = TestClient(main.app).post("/similar", json={"text": "hi"})
    assert r.status_code == 503
