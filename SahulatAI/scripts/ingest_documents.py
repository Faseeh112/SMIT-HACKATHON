"""
scripts/ingest_documents.py

Runs the full pipeline: SOURCE -> LOAD -> CHUNK -> EMBED -> CHROMADB.

Ingests three source types:
  1. Crawled pages saved as JSON under knowledge_base/raw/ (from crawl_saylani.py)
  2. Challenge documents (plain text/markdown) placed under
     knowledge_base/challenge_docs/
  3. Official PDFs placed under knowledge_base/pdfs/

Incremental by design: rag/vector_store.upsert_chunks() skips chunks whose
content_hash already exists, so re-running this script after a small
website change only re-embeds what actually changed.

Usage:
    python scripts/ingest_documents.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from database import db  # noqa: E402
from rag.chunker import chunk_document  # noqa: E402
from rag.loader import load_challenge_text_file, load_crawled_page, load_pdf  # noqa: E402
from rag.vector_store import upsert_chunks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sahulatai.ingest")


def ingest_all() -> dict:
    db.init_db()
    run_id = db.start_ingestion_run(source_type="mixed")

    raw_dir = settings.knowledge_base_dir / "raw"
    challenge_dir = settings.knowledge_base_dir / "challenge_docs"
    pdf_dir = settings.knowledge_base_dir / "pdfs"
    challenge_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    documents = []

    for json_path in sorted(raw_dir.glob("*.json")):
        try:
            documents.append(load_crawled_page(json_path))
        except Exception as exc:
            logger.warning("Skipping %s: %s", json_path, exc)

    for txt_path in sorted(list(challenge_dir.glob("*.txt")) + list(challenge_dir.glob("*.md"))):
        try:
            documents.append(load_challenge_text_file(txt_path))
        except Exception as exc:
            logger.warning("Skipping %s: %s", txt_path, exc)

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            documents.extend(load_pdf(pdf_path))
        except Exception as exc:
            logger.warning("Skipping %s: %s", pdf_path, exc)

    logger.info("Loaded %d documents.", len(documents))

    total_chunks_created = 0
    total_chunks_indexed = 0
    for doc in documents:
        chunks = chunk_document(doc)
        total_chunks_created += len(chunks)
        indexed = upsert_chunks(chunks)
        total_chunks_indexed += indexed

    db.finish_ingestion_run(
        run_id,
        pages_crawled=0,
        documents_ingested=len(documents),
        chunks_created=total_chunks_indexed,
        status="completed",
        notes=f"{total_chunks_created} chunks considered, {total_chunks_indexed} newly indexed",
    )

    logger.info(
        "Ingestion complete. %d documents, %d chunks considered, %d newly indexed into ChromaDB.",
        len(documents), total_chunks_created, total_chunks_indexed,
    )
    return {
        "documents": len(documents),
        "chunks_considered": total_chunks_created,
        "chunks_indexed": total_chunks_indexed,
    }


if __name__ == "__main__":
    ingest_all()
