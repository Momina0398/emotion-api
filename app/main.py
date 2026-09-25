import time
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.model import load_model, LABELS

app = FastAPI(title="Multilingual Emotion Detection API", version="0.1.0")
model = load_model()


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


@app.get("/health")
def health():
    return {"status": "ok", "model": model.name, "labels": LABELS}


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
