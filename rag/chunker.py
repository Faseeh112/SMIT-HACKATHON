"""
Splits Document text into overlapping chunks sized for embedding + retrieval.
Chunking is character-based (simple, dependency-light, deterministic) with a
sentence-boundary snap so chunks don't cut mid-sentence where avoidable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import settings
from rag.loader import Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?۔])\s+")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_id: str
    source_url: str
    source_title: str
    source_type: str
    section: str
    crawl_date: str | None
    content_hash: str
    filename: str | None = None
    page_number: int | None = None


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENTENCE_SPLIT_RE.split(text)


def chunk_document(doc: Document, chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> list[Chunk]:
    """Greedy sentence-packing chunker with character-based size/overlap targets."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_index = 0

    def flush(section_hint: str) -> None:
        nonlocal current, current_len, chunk_index
        if not current:
            return
        text = " ".join(current).strip()
        if text:
            chunk_index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.document_id}-c{chunk_index}",
                    text=text,
                    document_id=doc.document_id,
                    source_url=doc.source_url,
                    source_title=doc.source_title,
                    source_type=doc.source_type,
                    section=section_hint,
                    crawl_date=doc.crawl_date,
                    content_hash=doc.content_hash,
                    filename=doc.filename,
                    page_number=doc.page_number,
                )
            )

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            flush(section_hint=current[0][:60])
            # carry overlap: keep trailing sentences whose combined length <= overlap
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = overlap_sentences
            current_len = overlap_len
        current.append(sentence)
        current_len += len(sentence)

    flush(section_hint=current[0][:60] if current else "")
    return chunks
