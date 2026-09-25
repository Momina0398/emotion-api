# Multilingual Emotion Detection API

A FastAPI service built on my MSc dissertation, *Multilingual Models Have Feelings Too?*
(BRIGHTER / SemEval-2025 Task 11, 9 languages). Six emotions, multi-label: anger, disgust, fear, joy, sadness, surprise.

## Endpoints
| Endpoint | What it does |
|---|---|
| `GET /health` | Service status, loaded model and vector index |
| `POST /predict` | Emotion scores from the MAD-X model (XLM-RoBERTa-large + language and task adapters) |
| `POST /similar` | Embeds the text, searches a FAISS index of labelled BRIGHTER examples, returns ranked neighbours and a retrieval-based emotion vote |

## Retrieval
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (normalised, cosine similarity)
- Vector index: FAISS `IndexFlatIP`, up to 3,000 training examples per language
- Evaluation on the dev split: see [`examples/retrieval_eval.json`](examples/retrieval_eval.json)
  (kNN macro-F1 and label precision@10 per language). Not comparable to the dissertation scores.
- The index is not committed; rebuild it with the part 2 notebook.

## Run locally
```bash
pip install -r requirements.txt
MODEL_BACKEND=dummy uvicorn app.main:app --reload     # open http://localhost:8000/docs
```
Real model: `MODEL_BACKEND=madx ADAPTER_PATH=/path/to/rus_track_a_C1`.
Retrieval: `RETRIEVER_BACKEND=st INDEX_DIR=index`.

## Docker
```bash
docker build -t emotion-api .
docker run -p 8000:8000 emotion-api
```

## Tests and CI
```bash
pip install -r requirements-dev.txt && pytest -q
```
GitHub Actions runs 14 tests (with mock model and mock embedder backends), then builds and smoke-tests the Docker image on every push.

## Roadmap
- [x] MAD-X model backend
- [x] Embeddings, FAISS vector search and retrieval evaluation
- [ ] Use retrieved examples as few-shot context for an LLM (full RAG)
- [ ] Deploy to a free cloud tier
