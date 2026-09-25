# Multilingual Emotion Detection API

A FastAPI service built on my MSc dissertation, *Multilingual Models Have Feelings Too?*
(BRIGHTER / SemEval-2025 Task 11, 9 languages). Six emotions, multi-label: anger, disgust, fear, joy, sadness, surprise.

Trained adapters: https://huggingface.co/MominaMahmood/All-Emotion-Adapters

## Endpoints (all async)
| Endpoint | What it does |
|---|---|
| `GET /health` | Service status, loaded model, vector index and LLM |
| `POST /predict` | Emotion scores from the MAD-X model (XLM-RoBERTa-large + language and task adapters) |
| `POST /similar` | Embeds the text, searches a FAISS index of labelled BRIGHTER examples, returns ranked neighbours and a retrieval-based vote |
| `POST /rag_predict` | RAG: retrieves similar labelled examples, adds them to the prompt of an open-source LLM, parses its JSON answer |

## How the RAG chain works
1. **Retrieve** the k most similar same-language examples (multilingual MiniLM embeddings, FAISS `IndexFlatIP`)
2. **Build the prompt** with those examples as few-shot context
3. **Generate** with `Qwen/Qwen2.5-1.5B-Instruct` (any Hugging Face chat model via `LLM_MODEL`), greedy decoding
4. **Parse** the JSON reply and keep only the 6 valid emotions; `parsed_ok` flags malformed output

Blocking model calls run in a thread pool with a GPU lock, so the event loop stays free.
The response reports `retrieval_ms` and `llm_ms` separately.

## Evaluation
- [`examples/retrieval_eval.json`](examples/retrieval_eval.json): kNN retrieval, macro-F1 and label precision@10 per language
- [`examples/rag_eval.json`](examples/rag_eval.json): zero-shot vs RAG prompting with the same LLM, 100 dev texts per language, macro-F1, JSON parse rate and latency

These use small samples and are not comparable to the dissertation's official-script scores.

## Run locally
```bash
pip install -r requirements.txt
MODEL_BACKEND=dummy uvicorn app.main:app --reload     # open http://localhost:8000/docs
```
| Variable | Values |
|---|---|
| `MODEL_BACKEND` | `dummy`, `madx` (+ `ADAPTER_PATH`) |
| `RETRIEVER_BACKEND` | `none`, `dummy`, `st` (+ `INDEX_DIR`) |
| `LLM_BACKEND` | `none`, `dummy`, `hf` (+ `LLM_MODEL`) |

## Docker
```bash
docker build -t emotion-api .
docker run -p 8000:8000 emotion-api
```

## Tests and CI
```bash
pip install -r requirements-dev.txt && pytest -q
```
GitHub Actions runs 23 tests with mock model, embedder and LLM backends, then builds and smoke-tests the Docker image on every push.

## Roadmap
- [x] MAD-X model backend
- [x] Embeddings, FAISS vector search and retrieval evaluation
- [x] RAG with an open-source LLM, structured JSON output, async endpoints
- [ ] Load adapters straight from the Hugging Face Hub
- [ ] Deploy to a free cloud tier
