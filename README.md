# Multilingual Emotion Detection API

A FastAPI service that serves a multi-label emotion classifier (anger, disgust, fear, joy, sadness, surprise)
fine-tuned during my MSc dissertation (CAST, BRIGHTER / SemEval-2025 Task 11, 9 languages).

## Run locally
```bash
pip install -r requirements.txt
MODEL_BACKEND=dummy uvicorn app.main:app --reload
# open http://localhost:8000/docs
```

## Run with Docker
```bash
docker build -t emotion-api .
docker run -p 8000:8000 emotion-api
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
CI (GitHub Actions) runs the tests and builds and smoke-tests the Docker image on every push.

## Roadmap
- [x] madx backend loads the real CAST Phase 3 adapters (xlm-roberta-large + lang + task + head)
- [ ] Publish adapter weights publicly and link them here
- [ ] Record latency (p50/p95) for CPU vs GPU and batch sizes
- [ ] Optional: RAG endpoint that retrieves labelled examples from a vector database (FAISS or Chroma) and uses them as few-shot context
- [ ] Optional: deploy to a free cloud tier (e.g. Hugging Face Spaces or Google Cloud Run)
