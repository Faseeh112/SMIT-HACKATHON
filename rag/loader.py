"""
Loaders turn raw sources (crawled HTML JSON, PDFs, challenge docs) into a
common Document representation ready for chunking.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - handled gracefully at call time
    fitz = None


@dataclass
class Document:
    text: str
    source_url: str
    source_title: str
    source_type: str  # challenge_document | official_website | official_pdf | secondary_source
    document_id: str
    crawl_date: Optional[str] = None
    filename: Optional[str] = None
    page_number: Optional[int] = None
    content_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_crawled_page(json_path: Path) -> Document:
    """Load a single crawled-page JSON file produced by scripts/crawl_saylani.py."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return Document(
        text=data["text"],
        source_url=data["source_url"],
        source_title=data.get("title") or data["source_url"],
        source_type=data.get("source_type", "official_website"),
        document_id=hashlib.sha256(data["source_url"].encode("utf-8")).hexdigest()[:16],
        crawl_date=data.get("crawl_date"),
    )


def load_pdf(pdf_path: Path, source_type: str = "official_pdf") -> list[Document]:
    """Extract text per page from a PDF. Requires PyMuPDF (fitz)."""
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install -r requirements.txt"
        )
    documents: list[Document] = []
    doc_id = hashlib.sha256(pdf_path.name.encode("utf-8")).hexdigest()[:16]
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            documents.append(
                Document(
                    text=text,
                    source_url=f"file://{pdf_path.name}",
                    source_title=pdf_path.stem,
                    source_type=source_type,
                    document_id=f"{doc_id}-p{page_number}",
                    filename=pdf_path.name,
                    page_number=page_number,
                )
            )
    return documents


def load_challenge_text_file(txt_path: Path) -> Document:
    """Load a plain-text or markdown challenge document."""
    text = txt_path.read_text(encoding="utf-8")
    doc_id = hashlib.sha256(txt_path.name.encode("utf-8")).hexdigest()[:16]
    return Document(
        text=text,
        source_url=f"file://{txt_path.name}",
        source_title=txt_path.stem,
        source_type="challenge_document",
        document_id=doc_id,
        filename=txt_path.name,
    )
