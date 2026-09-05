"""
Retriever: applies relevance thresholding, de-duplication, and trust-based
ranking on top of the raw vector_store.query() results.
"""
from __future__ import annotations

from config.settings import settings
from rag import vector_store

_TRUST_ORDER = {
    "challenge_document": 0,
    "official_website": 1,
    "official_pdf": 2,
    "secondary_source": 3,
}


def retrieve(question: str, top_k: int | None = None,
             min_score: float | None = None) -> list[dict]:
    top_k = top_k or settings.top_k
    min_score = settings.min_relevance_score if min_score is None else min_score

    raw_results = vector_store.query(question, top_k=top_k * 2)  # over-fetch, then filter/dedupe
    filtered = [r for r in raw_results if r["score"] >= min_score]

    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for r in filtered:
        if r["content_hash"] in seen_hashes:
            continue
        seen_hashes.add(r["content_hash"])
        deduped.append(r)

    deduped.sort(key=lambda r: (_TRUST_ORDER.get(r["source_type"], 9), -r["score"]))
    return deduped[:top_k]


def classify_retrieval(results: list[dict], strong_score: float | None = None) -> str:
    """
    Classify the retrieval outcome into one of:
    'answerable'  — at least one result clears the strong-relevance bar
    'partial'     — results exist but none are strongly relevant
    'not_found'   — no results at all
    """
    strong_score = settings.strong_relevance_score if strong_score is None else strong_score
    if not results:
        return "not_found"
    if any(r["score"] >= strong_score for r in results):
        return "answerable"
    return "partial"
