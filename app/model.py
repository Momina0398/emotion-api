"""Model loading and prediction. Heavy libraries are imported lazily."""
import os
from typing import List, Dict

LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


class DummyModel:
    """Deterministic fake model for tests and CI."""
    name = "dummy"

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        out = []
        for t in texts:
            joy = 0.9 if any(w in t.lower() for w in ["happy", "great", "love"]) else 0.1
            out.append({"anger": 0.1, "disgust": 0.05, "fear": 0.1,
                        "joy": joy, "sadness": 0.1, "surprise": 0.2})
        return out


class HFModel:
    """Wraps a fine-tuned multi-label Hugging Face classifier."""

    def __init__(self, path: str, max_length: int = 128):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.max_length = max_length
        self.name = os.path.basename(path.rstrip("/"))

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        enc = self.tok(texts, padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            probs = self.torch.sigmoid(self.model(**enc).logits).cpu().tolist()
        return [dict(zip(LABELS, [round(p, 4) for p in row])) for row in probs]


class MadXModel:
    """Loads a CAST Phase 3 MAD-X run: xlm-roberta-large + lang adapter + task adapter + head.

    Mirrors the inference code in Phase3_xlmr.ipynb. ADAPTER_PATH must hold the
    lang/, task/ and head/ sub-folders, e.g. .../adapters_xlmr_weighted/rus_track_a_C1
    """

    def __init__(self, adapter_path: str, base: str = "xlm-roberta-large", max_length: int = 256):
        import torch
        import adapters.composition as ac
        from adapters import XLMRobertaAdapterModel
        from transformers import AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(base)
        m = XLMRobertaAdapterModel.from_pretrained(base)
        m.load_adapter(f"{adapter_path}/lang", load_as="lang")
        m.load_adapter(f"{adapter_path}/task", load_as="emotion_task")
        m.load_head(f"{adapter_path}/head")
        m.set_active_adapters(ac.Stack("lang", "emotion_task"))
        self.model = m.to(self.device).eval()
        self.max_length = max_length
        self.name = "madx-" + os.path.basename(adapter_path.rstrip("/"))

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        enc = self.tok(texts, padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            probs = self.torch.sigmoid(self.model(**enc).logits).cpu().tolist()
        if len(probs[0]) != len(LABELS):
            raise RuntimeError(f"Head has {len(probs[0])} labels, expected {len(LABELS)}. "
                               "English runs drop 'disgust'; use a non-English run.")
        return [dict(zip(LABELS, [round(p, 4) for p in row])) for row in probs]


def load_model():
    backend = os.getenv("MODEL_BACKEND", "dummy")
    if backend == "dummy":
        return DummyModel()
    if backend == "madx":
        path = os.getenv("ADAPTER_PATH")
        if not path:
            raise RuntimeError("MODEL_BACKEND=madx needs ADAPTER_PATH")
        return MadXModel(path)
    if backend == "hf":
        path = os.getenv("MODEL_PATH")
        if not path:
            raise RuntimeError("MODEL_BACKEND=hf needs MODEL_PATH")
        return HFModel(path)
    raise RuntimeError(f"Unknown MODEL_BACKEND: {backend}")
