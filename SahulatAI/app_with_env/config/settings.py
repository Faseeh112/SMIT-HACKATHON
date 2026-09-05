"""
Central configuration for SahulatAI.
Loads environment variables once and exposes typed settings.
Never hardcodes secrets; never prints the API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- Secrets (never logged/printed) ---
    # Groq (groq.com) — free, fast-inference API. NOTE: this is a different
    # company/product from xAI's "Grok" model, despite the similar name.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    # gemini_api_key is now optional: only used for higher-quality RAG
    # embeddings (xAI's Grok API has no embeddings endpoint). If left blank,
    # a local dependency-free fallback embedding is used automatically.
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    porcupine_access_key: str = field(default_factory=lambda: os.getenv("PORCUPINE_ACCESS_KEY", ""))

    # --- Models ---
    # Chat/agent model: Groq, via its OpenAI-compatible endpoint.
    # NOTE: Groq deprecated the old llama-3.3-70b-versatile / llama-3.1-8b-instant
    # models. openai/gpt-oss-120b is the current recommended general-purpose,
    # tool-calling-capable model as of Aug 2026. See https://console.groq.com/docs/models
    groq_chat_model: str = field(default_factory=lambda: os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b"))
    # Embedding model: optional, only used if gemini_api_key is set.
    # NOTE: gemini-1.5-flash and text-embedding-004 were the original defaults
    # here but have since been retired by Google; these current models are
    # used instead unless overridden via the environment.
    gemini_embed_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"))

    # --- Voice ---
    wake_word: str = field(default_factory=lambda: os.getenv("WAKE_WORD", "Hi Sahulat"))

    # --- RAG tuning ---
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 5))
    # NOTE: these thresholds are tuned for the local dependency-free fallback
    # embedding (rag/embeddings.py), which scores lower than a true semantic
    # embedding model would. If you set GEMINI_API_KEY for real embeddings,
    # you may want to raise both back up (e.g. 0.35 / 0.55) via .env.
    min_relevance_score: float = field(default_factory=lambda: _get_float("MIN_RELEVANCE_SCORE", 0.15))
    strong_relevance_score: float = field(default_factory=lambda: _get_float("STRONG_RELEVANCE_SCORE", 0.28))

    # --- Agent ---
    max_tool_calls: int = field(default_factory=lambda: _get_int("MAX_TOOL_CALLS", 5))

    # --- Paths ---
    chroma_db_dir: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("CHROMA_DB_DIR", "data/chroma_db"))
    sqlite_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("SQLITE_DB_PATH", "data/sahulatai.db"))
    knowledge_base_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge_base")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    sources_config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "sources.yaml")
    eligibility_rules_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "eligibility_rules.yaml")

    # --- Crawler ---
    crawl_max_depth: int = field(default_factory=lambda: _get_int("CRAWL_MAX_DEPTH", 2))
    crawl_max_pages: int = field(default_factory=lambda: _get_int("CRAWL_MAX_PAGES", 60))
    crawl_timeout_seconds: int = field(default_factory=lambda: _get_int("CRAWL_TIMEOUT_SECONDS", 10))
    crawl_user_agent: str = field(default_factory=lambda: os.getenv(
        "CRAWL_USER_AGENT", "SahulatAI-Bot/1.0 (+local hackathon project)"
    ))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems, empty if config looks OK."""
        problems = []
        if not self.groq_api_key:
            problems.append(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "free Groq API key (https://console.groq.com/keys)."
            )
        return problems


settings = Settings()

# Ensure required directories exist at import time (idempotent, cheap).
for _dir in (settings.chroma_db_dir, settings.logs_dir, settings.sqlite_db_path.parent):
    _dir.mkdir(parents=True, exist_ok=True)
