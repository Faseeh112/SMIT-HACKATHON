"""Tool: search_documents — the RAG retrieval tool exposed to the Grok agent."""
from __future__ import annotations

from rag.retriever import classify_retrieval, retrieve

TOOL_SCHEMA = {
    "name": "search_documents",
    "description": (
        "Search SahulatAI's local knowledge base (official Saylani website content, "
        "official PDFs, and challenge documents) for information relevant to the "
        "user's question. Always use this before answering any Saylani-specific "
        "informational question (courses, admissions, welfare programs, services)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, ideally the user's question or a paraphrase of it.",
            }
        },
        "required": ["query"],
    },
}


def run(query: str) -> dict:
    results = retrieve(query)
    status = classify_retrieval(results)
    return {
        "status": status,  # answerable | partial | not_found
        "results": [
            {
                "text": r["text"],
                "source_url": r["source_url"],
                "source_title": r["source_title"],
                "source_type": r["source_type"],
                "page_number": r.get("page_number") or None,
                "score": r["score"],
            }
            for r in results
        ],
    }
