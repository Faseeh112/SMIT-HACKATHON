"""
Lightweight BM25 keyword scoring, used to hybridize with the local fallback
embeddings in vector_store.query().

Why this exists: the fallback embedder (rag/embeddings.py) is a hashed
bag-of-words vector with no semantic understanding, and this knowledge base
has many near-identical templated pages (every campus page, every course
page shares most of its boilerplate wording). That combination means a pure
cosine-similarity search can bury the one genuinely distinguishing document
(e.g. a "complete list of all campuses" page, or the one page that actually
names a specific course) under a pile of superficially-similar boilerplate
pages. BM25 fixes exactly this failure mode: common/boilerplate words that
appear in nearly every document get a very low IDF weight, while rare,
distinguishing words (a specific city name, a specific course name) get a
high weight — so an exact keyword match pulls its document to the top
regardless of how much boilerplate surrounds it.

This is intentionally simple (no external dependency, no persistent index —
the corpus here is only ~200 chunks, so recomputing document frequencies on
every call costs a few milliseconds, which is negligible next to the
network latency of an LLM call).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_K1 = 1.5
_B = 0.75

# Common filler words excluded from BM25 matching so they can't manufacture
# false relevance just by co-occurring in a mostly-unrelated sentence (e.g.
# "what is the capital of France" sharing "what"/"is"/"the"/"of" with half
# the knowledge base). Deliberately small and generic rather than a full
# stopword corpus — just enough to stop the most common function words from
# contributing IDF-weighted score.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "and", "or", "but", "if", "so", "of", "in", "on", "at", "to",
    "for", "with", "about", "as", "by", "from", "into", "over", "under",
    "this", "that", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "what", "which", "who", "whom", "how", "when", "where",
    "why", "do", "does", "did", "can", "could", "will", "would", "should",
    "shall", "may", "might", "not", "no", "yes", "please", "tell", "give",
    "hi", "hello", "ok", "okay", "kya", "hai", "hain", "ka", "ki", "ke",
    "ko", "sa", "se", "ma", "mein", "ha", "hn", "kn", "sy",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Return one BM25 score per document in `documents`, for `query`."""
    if not documents:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return [0.0] * len(documents)

    tokenized_docs = [_tokenize(d) for d in documents]
    doc_lens = [len(d) for d in tokenized_docs]
    avg_len = (sum(doc_lens) / len(doc_lens)) or 1.0
    n_docs = len(documents)

    # Document frequency per query term (how many docs contain it at least once).
    df: dict[str, int] = {}
    for term in set(query_terms):
        df[term] = sum(1 for doc in tokenized_docs if term in doc)

    idf: dict[str, float] = {}
    for term, freq in df.items():
        # Standard BM25 IDF with a +1 floor so it never goes negative for
        # very common terms.
        idf[term] = max(0.0, math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0))

    scores = []
    for doc_tokens, doc_len in zip(tokenized_docs, doc_lens):
        if not doc_tokens:
            scores.append(0.0)
            continue
        term_counts = Counter(doc_tokens)
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (_K1 + 1)
            denominator = tf + _K1 * (1 - _B + _B * (doc_len / avg_len))
            score += idf.get(term, 0.0) * (numerator / denominator)
        scores.append(score)
    return scores
