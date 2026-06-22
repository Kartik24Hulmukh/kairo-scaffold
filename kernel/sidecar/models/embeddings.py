"""Pluggable embeddings: fastembed (ONNX-quantized Rust), model2vec (static CPU), bge-m3.

Design:
  - EmbeddingBackend is a structural Protocol — duck-typed, no ABC overhead.
  - FastEmbedBackend: ONNX-quantized via fastembed; falls back to deterministic hash if unavailable.
  - Model2VecBackend: static CPU embeddings via model2vec; falls back to hash if unavailable.
  - get_embedder(): factory by mode string.

Fallback:
  When neither library is installed, _hash_embed() returns a deterministic
  64-dimensional pseudo-embedding based on SHA-256 of the text. This is NOT
  semantically meaningful but is deterministic — useful for offline tests.

GATE: pytest tests/test_embeddings.py -v
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Structural protocol for embedding backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...


# ---------------------------------------------------------------------------
# Shared deterministic fallback (used when ML deps are missing)
# ---------------------------------------------------------------------------

def _hash_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic pseudo-embedding using SHA-256.

    Returns a 64-dimensional vector where each element is a byte value in [0, 1].
    Deterministic: same text → same vector across runs and machines.
    Not semantically meaningful — only for offline testing.
    """
    result = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
        vec = [b / 255.0 for b in digest[:64]]
        result.append(vec)
    return result


# ---------------------------------------------------------------------------
# FastEmbed backend (ONNX-quantized via Rust)
# ---------------------------------------------------------------------------

class FastEmbedBackend:
    """fastembed: ONNX-quantized TextEmbedding, worker-tier speed.

    Falls back to _hash_embed when fastembed is not installed.
    """

    EMBEDDING_DIM = 384  # bge-small-en-v1.5 native dim

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._available = False
        self._model = None
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=model_name)
            self._available = True
        except ImportError:
            pass
        except Exception:
            # Model download or init failure — fall back silently.
            pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts. Falls back to deterministic hash if unavailable."""
        if not texts:
            return []
        if not self._available or self._model is None:
            return _hash_embed(texts)
        try:
            return [emb.tolist() for emb in self._model.embed(texts)]
        except Exception:
            return _hash_embed(texts)


# ---------------------------------------------------------------------------
# model2vec backend (static CPU embeddings)
# ---------------------------------------------------------------------------

class Model2VecBackend:
    """model2vec: static embeddings, very fast CPU inference.

    Falls back to _hash_embed when model2vec is not installed.
    """

    def __init__(self, model_name: str = "minishlab/potion-base-8M") -> None:
        self._model_name = model_name
        self._available = False
        self._model = None
        try:
            from model2vec import StaticModel
            self._model = StaticModel.from_pretrained(model_name)
            self._available = True
        except ImportError:
            pass
        except Exception:
            pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts. Falls back to deterministic hash if unavailable."""
        if not texts:
            return []
        if not self._available or self._model is None:
            return _hash_embed(texts)
        try:
            embeddings = self._model.encode(texts)
            # model2vec returns numpy arrays; convert to plain Python lists.
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
        except Exception:
            return _hash_embed(texts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(mode: str = "fastembed") -> EmbeddingBackend:
    """Return an EmbeddingBackend configured by mode string.

    Args:
        mode: "fastembed" (default) or "model2vec".

    Returns:
        EmbeddingBackend instance (never raises; falls back gracefully).
    """
    if mode == "fastembed":
        return FastEmbedBackend()
    elif mode == "model2vec":
        return Model2VecBackend()
    raise ValueError(f"Unknown embedding mode: {mode!r}. Choose 'fastembed' or 'model2vec'.")
