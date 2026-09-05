"""
Embedding generation, using Gemini's free-tier embedding model by default,
with a local/offline fallback (hashing-based bag-of-words vector) so the
project remains runnable if the Gemini quota is exhausted or the key is
missing during development/testing.

The fallback is NOT a production-quality embedding — it exists only so the
rest of the pipeline (chunk -> embed -> store -> retrieve) can be exercised
and tested without hitting a paid/quota-limited API.
"""
from __future__ import annotations

import hashlib
import logging
import math
from functools import lru_cache

from config.settings import settings

logger = logging.getLogger("sahulatai.embeddings")

_FALLBACK_DIM = 256


def _fallback_embed(text: str) -> list[float]:
    """Deterministic, dependency-free pseudo-embedding for offline/dev use."""
    vec = [0.0] * _FALLBACK_DIM
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % _FALLBACK_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@lru_cache(maxsize=1)
def _get_client():
    if not settings.gemini_api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=settings.gemini_api_key)
    except Exception as exc:  # pragma: no cover - network/environment dependent
        logger.warning("Gemini client unavailable, using local fallback embeddings: %s", exc)
        return None


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed a single string. Falls back to a local vector if Gemini is unavailable."""
    client = _get_client()
    if client is None:
        return _fallback_embed(text)
    try:
        response = client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=text,
            config={"task_type": task_type},
        )
        return list(response.embeddings[0].values)
    except Exception as exc:  # pragma: no cover - network/quota dependent
        logger.warning("Gemini embedding call failed (%s); using local fallback for this text.", exc)
        return _fallback_embed(text)


def embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    return [embed_text(t, task_type=task_type) for t in texts]
