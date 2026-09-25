"""Prompt building, LLM backends and structured-output parsing for RAG."""
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from app.model import LABELS

DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an emotion classifier for short social media texts in many languages. "
    f"The possible emotions are: {', '.join(LABELS)}. "
    "A text can express several emotions, or none. "
    'Reply with JSON only, in exactly this format: {"emotions": ["joy"]}. '
    'Use {"emotions": []} if no emotion applies.'
)


def build_prompt(text: str, neighbours: Optional[List[Dict]] = None, max_chars: int = 300) -> str:
    parts = []
    if neighbours:
        parts.append("Here are labelled examples of similar texts:")
        for i, n in enumerate(neighbours, 1):
            labels = [e for e in LABELS if n["labels"].get(e)]
            parts.append(f"{i}. Text: {n['text'][:max_chars]}\n   Emotions: {json.dumps(labels)}")
        parts.append("")
    parts.append("Classify this text.")
    parts.append(f"Text: {text[:max_chars * 2]}")
    return "\n".join(parts)


def parse_emotions(raw: str) -> Tuple[List[str], bool]:
    """Return (valid emotions in label order, parsed_ok)."""
    match = re.search(r"\{.*?\}", raw, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            found = {str(e).strip().lower() for e in data.get("emotions", [])}
            return [e for e in LABELS if e in found], True
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return [], False


class DummyLLM:
    """Fake LLM for tests: 'detects' emotion names written in the text."""
    name = "dummy-llm"

    def generate(self, system: str, user: str) -> str:
        target = user.split("Text:")[-1].lower()
        return json.dumps({"emotions": [e for e in LABELS if e in target]})


class HFLLM:
    """Any Hugging Face chat model."""

    def __init__(self, model_name: str = DEFAULT_LLM, max_new_tokens: int = 48):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.name = model_name

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        n_prompt = enc["input_ids"].shape[1]
        return self.tok.decode(out[0, n_prompt:], skip_special_tokens=True)


def load_llm():
    backend = os.getenv("LLM_BACKEND", "none")
    if backend == "none":
        return None
    if backend == "dummy":
        return DummyLLM()
    if backend == "hf":
        return HFLLM(os.getenv("LLM_MODEL", DEFAULT_LLM))
    raise RuntimeError(f"Unknown LLM_BACKEND: {backend}")
