"""
rag/ingest.py

Reusable single-file ingestion path shared by the Streamlit "add a document"
uploader (chat tab) and the admin panel's document manager. This is the same
LOAD -> CHUNK -> EMBED -> STORE pipeline scripts/ingest_documents.py runs in
bulk, but for exactly one just-uploaded file, plus bookkeeping in the
`documents` table so the admin panel can list / open / download / remove it
later.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from config.settings import settings
from database import db
from rag.chunker import chunk_document
from rag.loader import load_challenge_text_file, load_pdf
from rag.vector_store import delete_by_document_id, upsert_chunks

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileType(Exception):
    pass


def _uploads_dir() -> Path:
    d = settings.knowledge_base_dir / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ingest_uploaded_file(file_bytes: bytes, filename: str, uploaded_by: str | None = None,
                          source_type: str = "user_upload") -> dict:
    """Save `file_bytes` under knowledge_base/uploads/, index it, and register
    it in the documents table. Returns {"document_id", "filename", "chunks"}.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"'{suffix or filename}' isn't supported yet — upload a PDF, .txt, or .md file."
        )

    # Keep the on-disk name stable but collision-free.
    digest = hashlib.sha256(file_bytes).hexdigest()[:12]
    safe_name = f"{Path(filename).stem}-{digest}{suffix}"
    dest_path = _uploads_dir() / safe_name
    dest_path.write_bytes(file_bytes)

    if suffix == ".pdf":
        documents = load_pdf(dest_path, source_type=source_type)
    else:
        documents = [load_challenge_text_file(dest_path)]
        # load_challenge_text_file tags everything "challenge_document"; retag
        # with the caller's source_type so uploads are distinguishable in stats.
        for d in documents:
            d.source_type = source_type

    if not documents:
        raise ValueError("No readable text found in that file.")

    base_document_id = documents[0].document_id.split("-p")[0]

    total_chunks = 0
    for doc in documents:
        chunks = chunk_document(doc)
        total_chunks += upsert_chunks(chunks)

    db.register_document(
        document_id=base_document_id,
        filename=filename,
        file_path=str(dest_path),
        source_type=source_type,
        chunks_created=total_chunks,
        uploaded_by=uploaded_by,
    )

    return {"document_id": base_document_id, "filename": filename, "chunks": total_chunks}


def remove_document(document_id: str) -> int:
    """Delete a document's chunks from the vector store, its file on disk, and its DB row."""
    removed_chunks = delete_by_document_id(document_id)
    row = db.get_document(document_id)
    if row:
        path = Path(row["file_path"])
        if path.exists():
            path.unlink()
        db.delete_document_record(document_id)
    return removed_chunks
