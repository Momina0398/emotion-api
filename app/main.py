import asyncio
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from app.model import load_model, LABELS
from app.retrieval import load_retriever, Retriever
from app.llm import load_llm, build_prompt, parse_emotions, SYSTEM_PROMPT

app = FastAPI(title="Multilingual Emotion Detection API", version="0.3.0")
model = load_model()
retriever = load_retriever()
llm = load_llm()
gpu_lock = asyncio.Lock()


def ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


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


class RagRequest(BaseModel):
    text: str = Field(..., min_length=1)
    k: int = Field(5, ge=0, le=20)
    lang: Optional[str] = None
    use_retrieval: bool = True


class RagResponse(BaseModel):
    llm: str
    labels: List[str]
    parsed_ok: bool
    raw_output: str
    neighbours: List[Neighbour]
    retrieval_ms: float
    llm_ms: float
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "ok", "model": model.name, "labels": LABELS,
            "retriever": None if retriever is None else
            {"embedder": retriever.embedder.name, "size": retriever.index.ntotal},
            "llm": None if llm is None else llm.name}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    start = time.perf_counter()
    async with gpu_lock:
        scores = await run_in_threadpool(model.predict, req.texts)
    preds = [Prediction(text=t, scores=s,
                        labels=[k for k, v in s.items() if v >= req.threshold])
             for t, s in zip(req.texts, scores)]
    return PredictResponse(model=model.name, latency_ms=ms(start), predictions=preds)


@app.post("/similar", response_model=SimilarResponse)
async def similar(req: SimilarRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="No vector index loaded. Build one and set RETRIEVER_BACKEND.")
    start = time.perf_counter()
    hits = await run_in_threadpool(retriever.search, req.text, req.k, req.lang)
    vote = Retriever.vote(hits)
    return SimilarResponse(embedder=retriever.embedder.name, latency_ms=ms(start),
                           neighbours=hits, vote=vote,
                           vote_labels=[e for e, v in vote.items() if v >= req.threshold])


@app.post("/rag_predict", response_model=RagResponse)
async def rag_predict(req: RagRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="No LLM loaded. Set LLM_BACKEND.")
    if req.use_retrieval and req.k > 0 and retriever is None:
        raise HTTPException(status_code=503, detail="Retrieval requested but no vector index loaded.")
    start = time.perf_counter()
    hits = []
    if req.use_retrieval and req.k > 0:
        hits = await run_in_threadpool(retriever.search, req.text, req.k, req.lang)
    retrieval_ms = ms(start)
    t_llm = time.perf_counter()
    async with gpu_lock:
        raw = await run_in_threadpool(llm.generate, SYSTEM_PROMPT, build_prompt(req.text, hits))
    llm_ms = ms(t_llm)
    labels, ok = parse_emotions(raw)
    return RagResponse(llm=llm.name, labels=labels, parsed_ok=ok, raw_output=raw, neighbours=hits,
                       retrieval_ms=retrieval_ms, llm_ms=llm_ms, latency_ms=ms(start))
