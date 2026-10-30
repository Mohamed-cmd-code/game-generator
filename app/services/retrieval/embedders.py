"""Embedding backends behind one small interface.

* SentenceTransformerEmbedder - real semantic embeddings (all-MiniLM-L6-v2, CPU).
* TfidfEmbedder - dependency-free fallback (word + character n-gram TF-IDF). Used
  in tests and CI (no model download), and automatically if the model cannot load.

Each backend carries its own calibrated relevance threshold, because cosine
scores are not comparable across embedding spaces (see scripts/eval_retrieval.py).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Protocol

import numpy as np

from app.services.text import singular, words

log = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "by", "from", "is", "are", "be", "it", "its", "this", "that", "my", "me", "i", "you", "your", "we", "our", "can", "could", "should", "would", "will", "want", "make", "create", "game", "games", "please", "some", "about", "into", "than", "less", "more", "very"]
)


class Embedder(Protocol):
    name: str
    default_threshold: float

    def fit(self, corpus: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised vectors, shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    default_threshold = 0.30

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.name = f"sentence-transformers/{model_name}"
        # Pin to CPU: by default SentenceTransformer grabs a GPU (CUDA/MPS) if one is
        # present; device="cpu" keeps inference off the GPU for portable, deterministic runs.
        self._model = SentenceTransformer(model_name, device="cpu")

    def fit(self, corpus: list[str]) -> None:  # pretrained: nothing to fit
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class TfidfEmbedder:
    """Sparse-as-dense TF-IDF over words and character 3-grams, fitted on the catalogue."""

    name = "tfidf-char-ngrams"
    default_threshold = 0.25

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0)

    @staticmethod
    def _features(text: str) -> list[str]:
        feats: list[str] = []
        for w in words(text):
            if w in _STOPWORDS:
                continue
            w = singular(w.replace("'", ""))
            feats.append(f"w:{w}")
            padded = f"<{w}>"
            feats.extend(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
        return feats

    def fit(self, corpus: list[str]) -> None:
        docs = [set(self._features(t)) for t in corpus]
        df = Counter(f for d in docs for f in d)
        self._vocab = {f: i for i, f in enumerate(sorted(df))}
        n = len(corpus)
        self._idf = np.array(
            [math.log((1 + n) / (1 + df[f])) + 1.0 for f in sorted(df)], dtype=np.float32
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for f, tf in Counter(self._features(text)).items():
                if (j := self._vocab.get(f)) is not None:
                    w = 2.0 if f.startswith("w:") else 1.0  # whole words weigh more
                    out[row, j] = w * (1 + math.log(tf)) * self._idf[j]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


def build_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "tfidf":
        return TfidfEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # ImportError, no network to download the model, ...
        if backend == "sentence-transformers":
            raise
        log.warning("Semantic embedder unavailable (%s); falling back to TF-IDF.", exc)
        return TfidfEmbedder()
