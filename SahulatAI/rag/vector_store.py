"""
Pure Python + numpy persistent vector store.

This replaces the previous ChromaDB-backed implementation. ChromaDB pulls in
`chroma-hnswlib`, a C++ extension that has no pre-built wheel for every
Python/Windows combination, so installing it can fail with:

    error: Microsoft Visual C++ 14.0 or greater is required.

This module gives the same public API (query, upsert_chunks, existing_hashes,
collection_stats, clear_collection) using only numpy for cosine-similarity
search and pickle for persistence. No native/compiled dependency, no wheel
build required. This is plenty fast for a knowledge base of a few thousand
chunks, which is the expected scale for this project.
"""
from __future__ import annotations

import logging
import pickle
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from config.settings import settings
from rag.chunker import Chunk
from rag.embeddings import embed_batch, embed_text

logger = logging.getLogger("sahulatai.vector_store")

STORE_FILENAME = "sahulatai_knowledge_base.pkl"
_lock = threading.Lock()


class _Store:
    """In-memory collection, persisted to a single pickle file on disk."""

    def __init__(self, path: Path):
        self.path = path
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    data = pickle.load(f)
                self.ids = data.get("ids", [])
                self.embeddings = data.get("embeddings", [])
                self.documents = data.get("documents", [])
                self.metadatas = data.get("metadatas", [])
            except Exception as exc:  # pragma: no cover - corrupted file edge case
                logger.warning("Could not load vector store at %s (%s); starting fresh.", self.path, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "embeddings": self.embeddings,
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                },
                f,
            )
        tmp_path.replace(self.path)

    def count(self) -> int:
        return len(self.ids)

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        existing_index = {id_: i for i, id_ in enumerate(self.ids)}
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            if id_ in existing_index:
                i = existing_index[id_]
                self.embeddings[i] = emb
                self.documents[i] = doc
                self.metadatas[i] = meta
            else:
                self.ids.append(id_)
                self.embeddings.append(emb)
                self.documents.append(doc)
                self.metadatas.append(meta)
        self.save()

    def delete_all(self) -> None:
        self.ids, self.embeddings, self.documents, self.metadatas = [], [], [], []
        if self.path.exists():
            self.path.unlink()


@lru_cache(maxsize=1)
def _get_store() -> _Store:
    return _Store(settings.chroma_db_dir / STORE_FILENAME)


def existing_hashes() -> set[str]:
    """Return the set of content_hash values already stored, for incremental ingestion."""
    store = _get_store()
    return {m.get("content_hash", "") for m in store.metadatas}


def upsert_chunks(chunks: list[Chunk]) -> int:
    """Embed and store chunks, skipping ones whose content_hash is already present."""
    if not chunks:
        return 0
    known = existing_hashes()
    new_chunks = [c for c in chunks if c.content_hash not in known]
    if not new_chunks:
        logger.info("All %d chunks already indexed (unchanged); skipping.", len(chunks))
        return 0

    store = _get_store()
    texts = [c.text for c in new_chunks]
    embeddings = embed_batch(texts)
    ids = [c.chunk_id for c in new_chunks]
    metadatas = [
        {
            "source_url": c.source_url,
            "source_title": c.source_title,
            "source_type": c.source_type,
            "domain": c.source_url.split("/")[2] if "://" in c.source_url else "local",
            "document_id": c.document_id,
            "chunk_id": c.chunk_id,
            "section": c.section,
            "crawl_date": c.crawl_date or "",
            "content_hash": c.content_hash,
            "filename": c.filename or "",
            "page_number": c.page_number or 0,
        }
        for c in new_chunks
    ]
    with _lock:
        store.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info("Indexed %d new chunks (%d were already present).", len(new_chunks), len(chunks) - len(new_chunks))
    return len(new_chunks)


def query(text: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k or settings.top_k
    store = _get_store()
    if store.count() == 0:
        return []

    query_embedding = np.asarray(embed_text(text, task_type="RETRIEVAL_QUERY"), dtype=np.float32)
    matrix = np.asarray(store.embeddings, dtype=np.float32)

    query_norm = np.linalg.norm(query_embedding) or 1.0
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms[matrix_norms == 0] = 1.0
    similarities = (matrix @ query_embedding) / (matrix_norms * query_norm)

    k = min(top_k, len(similarities))
    top_indices = np.argsort(-similarities)[:k]

    output = []
    for i in top_indices:
        i = int(i)
        output.append(
            {
                "text": store.documents[i],
                "score": round(float(max(0.0, similarities[i])), 4),
                **store.metadatas[i],
            }
        )
    return output


def collection_stats() -> dict:
    store = _get_store()
    total = store.count()
    if total == 0:
        return {"total_chunks": 0, "documents": 0, "official_sources": 0, "challenge_documents": 0}
    metas = store.metadatas
    doc_ids = {m.get("document_id") for m in metas}
    official = sum(1 for m in metas if m.get("source_type") in ("official_website", "official_pdf"))
    challenge = sum(1 for m in metas if m.get("source_type") == "challenge_document")
    return {
        "total_chunks": total,
        "documents": len(doc_ids),
        "official_sources": official,
        "challenge_documents": challenge,
    }


def clear_collection() -> None:
    store = _get_store()
    store.delete_all()
    _get_store.cache_clear()
    _get_store()
