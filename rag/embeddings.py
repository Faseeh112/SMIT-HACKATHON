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

_FALLBACK_DIM = 4096


def _fallback_embed(text: str) -> list[float]:
    """
    Deterministic, dependency-free pseudo-embedding for offline/dev use —
    a hashed bag-of-words vector. Two tuning choices matter a lot for
    retrieval quality here:

    1. Dimension (4096, not 256): with only 256 buckets, a knowledge base
       with many proper nouns (50+ city/campus names, 80+ course names)
       hashes multiple unrelated words into the same bucket constantly,
       burying the real signal under collision noise. 4096 buckets makes
       collisions rare enough that specific words like "Rawalpindi" or
       "cybersecurity" reliably land in their own slot.
    2. Log-scaled term frequency (log1p, not raw count): a long page (e.g.
       a "complete list of all campuses" page with hundreds of words) would
       otherwise dilute every individual word's weight after normalization,
       so it scores *lower* against a short query than a short, narrowly-
       focused page does — exactly backwards from what we want, since the
       complete list is usually the *best* answer to a broad question.
       Log-scaling compresses very high counts so long pages aren't
       penalized as heavily just for being long.
    """
    counts: dict[int, float] = {}
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % _FALLBACK_DIM
        counts[idx] = counts.get(idx, 0.0) + 1.0
    vec = [0.0] * _FALLBACK_DIM
    for idx, count in counts.items():
        vec[idx] = math.log1p(count)
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
