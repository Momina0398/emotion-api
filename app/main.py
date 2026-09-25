import time
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model import load_model, LABELS
from app.retrieval import load_retriever, Retriever

app = FastAPI(title="Multilingual Emotion Detection API", version="0.2.0")
model = load_model()
retriever = load_retriever()


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class Prediction(BaseModel):
    text: str
    scores: Dict[str, float]
    labels: List[str]


class PredictResponse(BaseModel):
    model: str
    latency_ms: float
    predictions: List[Prediction]


class SimilarRequest(BaseModel):
    text: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    lang: Optional[str] = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class Neighbour(BaseModel):
    rank: int
    score: float
    text: str
    lang: str
    labels: Dict[str, int]


class SimilarResponse(BaseModel):
    embedder: str
    latency_ms: float
    neighbours: List[Neighbour]
    vote: Dict[str, float]
    vote_labels: List[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": model.name, "labels": LABELS,
            "retriever": None if retriever is None else
            {"embedder": retriever.embedder.name, "size": retriever.index.ntotal}}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    scores = model.predict(req.texts)
    preds = [Prediction(text=t, scores=s,
                        labels=[k for k, v in s.items() if v >= req.threshold])
             for t, s in zip(req.texts, scores)]
    return PredictResponse(model=model.name,
                           latency_ms=round((time.perf_counter() - start) * 1000, 2),
                           predictions=preds)


@app.post("/similar", response_model=SimilarResponse)
def similar(req: SimilarRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="No vector index loaded. Build one and set RETRIEVER_BACKEND.")
    start = time.perf_counter()
    hits = retriever.search(req.text, k=req.k, lang=req.lang)
    vote = Retriever.vote(hits)
    return SimilarResponse(embedder=retriever.embedder.name,
                           latency_ms=round((time.perf_counter() - start) * 1000, 2),
                           neighbours=hits, vote=vote,
                           vote_labels=[e for e, v in vote.items() if v >= req.threshold])
