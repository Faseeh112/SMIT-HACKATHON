"""Tests for the chunking and loader stages of the RAG pipeline.

These tests avoid the vector store / embeddings (network- or
quota-dependent) and focus on the deterministic, offline parts:
loading and chunking text.
"""
from __future__ import annotations

from rag.chunker import chunk_document
from rag.loader import Document


def test_chunk_short_document_produces_single_chunk():
    doc = Document(
        text="Saylani offers free IT courses. Admission is open to ages 15 to 45.",
        source_url="file://sample.md",
        source_title="Sample",
        source_type="challenge_document",
        document_id="doc1",
    )
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert "Saylani" in chunks[0].text
    assert chunks[0].source_type == "challenge_document"


def test_chunk_long_document_splits_into_multiple_chunks():
    sentence = "This is a sentence about Saylani welfare programs and services. "
    long_text = sentence * 40  # ~2600 chars
    doc = Document(
        text=long_text,
        source_url="file://long.md",
        source_title="Long Doc",
        source_type="official_website",
        document_id="doc2",
    )
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=80)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 700  # allows some slack for sentence packing


def test_document_content_hash_is_deterministic():
    d1 = Document(text="hello world", source_url="u", source_title="t",
                   source_type="official_website", document_id="1")
    d2 = Document(text="hello world", source_url="u", source_title="t",
                   source_type="official_website", document_id="2")
    assert d1.content_hash == d2.content_hash


def test_chunk_ids_are_unique_within_a_document():
    sentence = "Another Saylani sentence for chunking. "
    doc = Document(
        text=sentence * 30,
        source_url="file://x.md",
        source_title="X",
        source_type="official_website",
        document_id="docX",
    )
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
