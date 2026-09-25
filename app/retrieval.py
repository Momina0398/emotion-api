"""Embeddings + FAISS vector index + ranked retrieval."""
import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np

from app.model import LABELS

DEFAULT_ST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests. Not a real model."""
    name = "hash-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class STEmbedder:
    """Multilingual sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_ST_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 1000
                                 ).astype("float32")


class Retriever:
    def __init__(self, embedder, index, meta: List[Dict]):
        self.embedder, self.index, self.meta = embedder, index, meta

    @classmethod
    def build(cls, embedder, records: List[Dict]):
        """records: [{"text": str, "lang": str, "labels": {emotion: 0/1}}]"""
        import faiss
        vecs = embedder.embed([r["text"] for r in records])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(embedder, index, records)

    def save(self, directory: str):
        import faiss
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, f"{directory}/index.faiss")
        with open(f"{directory}/meta.jsonl", "w", encoding="utf-8") as f:
            for r in self.meta:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(f"{directory}/info.json", "w") as f:
            json.dump({"embedder": self.embedder.name, "size": len(self.meta)}, f)

    @classmethod
    def load(cls, directory: str, embedder):
        import faiss
        info = json.load(open(f"{directory}/info.json"))
        if info["embedder"] != embedder.name:
            raise RuntimeError(f"Index built with {info['embedder']}, not {embedder.name}")
        index = faiss.read_index(f"{directory}/index.faiss")
        meta = [json.loads(line) for line in open(f"{directory}/meta.jsonl", encoding="utf-8")]
        return cls(embedder, index, meta)

    def search(self, text: str, k: int = 5, lang: Optional[str] = None) -> List[Dict]:
        q = self.embedder.embed([text])
        # over-fetch when filtering by language, then keep the top k that match
        fetch = self.index.ntotal if lang else min(k, self.index.ntotal)
        scores, ids = self.index.search(q, fetch)
        hits = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            r = self.meta[i]
            if lang and r["lang"] != lang:
                continue
            hits.append({"rank": len(hits) + 1, "score": round(float(s), 4), **r})
            if len(hits) == k:
                break
        return hits

    @staticmethod
    def vote(hits: List[Dict]) -> Dict[str, float]:
        """Similarity-weighted share of neighbours carrying each emotion."""
        w = np.array([max(h["score"], 0.0) for h in hits])
        if not len(hits) or w.sum() == 0:
            return {e: 0.0 for e in LABELS}
        return {e: round(float(sum(wi * h["labels"].get(e, 0) for wi, h in zip(w, hits)) / w.sum()), 4)
                for e in LABELS}


def load_retriever():
    backend = os.getenv("RETRIEVER_BACKEND", "none")
    directory = os.getenv("INDEX_DIR", "index")
    if backend == "none" or not os.path.isdir(directory):
        return None
    embedder = HashEmbedder() if backend == "dummy" else STEmbedder(os.getenv("EMBED_MODEL", DEFAULT_ST_MODEL))
    return Retriever.load(directory, embedder)
