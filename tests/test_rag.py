import os
os.environ.setdefault("MODEL_BACKEND", "dummy")

import pytest
from fastapi.testclient import TestClient
import app.main as main
from app.llm import DummyLLM, build_prompt, parse_emotions
from app.retrieval import HashEmbedder, Retriever

RECORDS = [
    {"text": "so much joy and happiness", "lang": "eng", "labels": {"joy": 1}},
    {"text": "pure anger at this news", "lang": "eng", "labels": {"anger": 1}},
]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "retriever", Retriever.build(HashEmbedder(), RECORDS))
    monkeypatch.setattr(main, "llm", DummyLLM())
    return TestClient(main.app)


def test_parse_clean_json():
    assert parse_emotions('{"emotions": ["joy", "anger"]}') == (["anger", "joy"], True)


def test_parse_json_inside_text():
    assert parse_emotions('Sure! {"emotions": ["fear"]} Hope that helps.') == (["fear"], True)


def test_parse_drops_invalid_labels():
    assert parse_emotions('{"emotions": ["joy", "happiness", "LOVE"]}') == (["joy"], True)


def test_parse_garbage_fails_safely():
    assert parse_emotions("I think it is joyful") == ([], False)


def test_prompt_includes_neighbours():
    p = build_prompt("target text", [{"text": "example one", "labels": {"joy": 1}}])
    assert "example one" in p and '["joy"]' in p and p.strip().endswith("target text")


def test_zero_shot_prompt_has_no_examples():
    assert "labelled examples" not in build_prompt("target text", [])


def test_rag_endpoint(client):
    r = client.post("/rag_predict", json={"text": "full of joy today", "k": 2})
    body = r.json()
    assert r.status_code == 200 and body["parsed_ok"] and "joy" in body["labels"]
    assert len(body["neighbours"]) == 2


def test_rag_without_retrieval(client):
    r = client.post("/rag_predict", json={"text": "anger", "use_retrieval": False})
    assert r.status_code == 200 and r.json()["neighbours"] == []


def test_rag_without_llm_returns_503(monkeypatch):
    monkeypatch.setattr(main, "llm", None)
    assert TestClient(main.app).post("/rag_predict", json={"text": "x"}).status_code == 503
