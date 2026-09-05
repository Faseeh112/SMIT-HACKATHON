"""
SQLite persistence layer for SahulatAI.

Tables: users, conversations, messages, user_memory, complaints,
callbacks, eligibility_rules (cache), ingestion_runs.

Uses raw sqlite3 with parameterized queries (no string-built SQL) so it
has zero extra runtime dependency beyond the standard library, and is
easy to inspect/debug for a hackathon demo.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from config.settings import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    preferred_language TEXT DEFAULT 'en',
    city TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS user_memory (
    user_id TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    memory_value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, memory_key)
);

CREATE TABLE IF NOT EXISTS complaints (
    complaint_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS callbacks (
    callback_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    callback_date TEXT NOT NULL,
    callback_time TEXT NOT NULL,
    contact TEXT NOT NULL,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'scheduled',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eligibility_rules (
    program_key TEXT PRIMARY KEY,
    rules_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    pages_crawled INTEGER DEFAULT 0,
    documents_ingested INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    notes TEXT
);

-- Tracks individually-managed documents (admin/user uploads) so the admin
-- panel can list, open/download, and remove them. Crawled website pages are
-- NOT rows here (they live only in the vector store) since there can be
-- hundreds of them; this table is for discrete files someone added.
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'user_upload',
    uploaded_by TEXT,
    chunks_created INTEGER DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.utcnow().isoformat()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.sqlite_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Create all tables if they do not exist. Safe to call every startup."""
    path = db_path or settings.sqlite_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Users / conversations / messages
# ---------------------------------------------------------------------------

def ensure_user(user_id: str, preferred_language: str = "en", city: Optional[str] = None) -> None:
    with get_connection() as conn:
        existing = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO users (user_id, preferred_language, city, created_at) VALUES (?, ?, ?, ?)",
                (user_id, preferred_language, city, _now()),
            )


def start_conversation(user_id: str) -> str:
    conversation_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, user_id, started_at) VALUES (?, ?, ?)",
            (conversation_id, user_id, _now()),
        )
    return conversation_id


def add_message(conversation_id: str, role: str, content: str) -> str:
    message_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, content, _now()),
        )
    return message_id


def get_recent_messages(conversation_id: str, limit: int = 20) -> list[sqlite3.Row]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
    return list(reversed(rows))


def get_all_messages(conversation_id: str) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()


def list_conversations(user_id: str, limit: int = 50) -> list[sqlite3.Row]:
    """Conversations for the sidebar history list, newest first, each with a
    title derived from its first user message (or 'New chat' if empty)."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT c.conversation_id, c.started_at,
                   (SELECT content FROM messages m
                     WHERE m.conversation_id = c.conversation_id AND m.role = 'user'
                     ORDER BY m.created_at ASC LIMIT 1) AS first_message
            FROM conversations c
            WHERE c.user_id = ?
            ORDER BY c.started_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()


# ---------------------------------------------------------------------------
# User memory (non-sensitive only — enforced by agent/memory.py, not here)
# ---------------------------------------------------------------------------

def set_user_memory(user_id: str, key: str, value: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO user_memory (user_id, memory_key, memory_value, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, memory_key) DO UPDATE SET memory_value=excluded.memory_value, "
            "updated_at=excluded.updated_at",
            (user_id, key, value, _now()),
        )


def get_user_memory(user_id: str) -> dict[str, str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT memory_key, memory_value FROM user_memory WHERE user_id = ?", (user_id,)
        ).fetchall()
    return {row["memory_key"]: row["memory_value"] for row in rows}


# ---------------------------------------------------------------------------
# Complaints
# ---------------------------------------------------------------------------

def create_complaint_record(user_id: str, category: str, description: str) -> str:
    complaint_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO complaints (complaint_id, user_id, category, description, status, created_at) "
            "VALUES (?, ?, ?, ?, 'open', ?)",
            (complaint_id, user_id, category, description, _now()),
        )
    return complaint_id


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def create_callback_record(user_id: str, date: str, time: str, contact: str, reason: str = "") -> str:
    callback_id = f"CB-{uuid.uuid4().hex[:8].upper()}"
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO callbacks (callback_id, user_id, callback_date, callback_time, contact, reason, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'scheduled', ?)",
            (callback_id, user_id, date, time, contact, reason, _now()),
        )
    return callback_id


# ---------------------------------------------------------------------------
# Ingestion run tracking (used by admin panel)
# ---------------------------------------------------------------------------

def start_ingestion_run(source_type: str) -> str:
    run_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ingestion_runs (run_id, source_type, started_at, status) VALUES (?, ?, ?, 'running')",
            (run_id, source_type, _now()),
        )
    return run_id


def finish_ingestion_run(run_id: str, pages_crawled: int, documents_ingested: int,
                          chunks_created: int, status: str = "completed", notes: str = "") -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE ingestion_runs SET pages_crawled=?, documents_ingested=?, chunks_created=?, "
            "finished_at=?, status=?, notes=? WHERE run_id=?",
            (pages_crawled, documents_ingested, chunks_created, _now(), status, notes, run_id),
        )


def get_last_ingestion_run() -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()


# ---------------------------------------------------------------------------
# Documents (admin/user-uploaded files — list, add, remove, open/download)
# ---------------------------------------------------------------------------

def register_document(document_id: str, filename: str, file_path: str, source_type: str,
                       chunks_created: int, uploaded_by: Optional[str] = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO documents (document_id, filename, file_path, source_type, "
            "uploaded_by, chunks_created, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(document_id) DO UPDATE SET chunks_created=excluded.chunks_created",
            (document_id, filename, file_path, source_type, uploaded_by, chunks_created, _now()),
        )


def list_documents() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()


def get_document(document_id: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()


def delete_document_record(document_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
