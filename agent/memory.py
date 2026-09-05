"""
Two layers of memory:

1. Conversation memory — recent turns of the current conversation, pulled
   from SQLite `messages` and given to Grok as context.
2. User memory — durable, cross-session, NON-SENSITIVE facts (preferred
   language, city if volunteered, stated preferences). Never stores API
   keys, passwords, tokens, or anything else sensitive.

Each user's memory is isolated by user_id — the primary key of every read
and write below.
"""
from __future__ import annotations

from database import db

# Keys we allow into long-term user memory. Anything else is dropped rather
# than silently stored, so the model can't be tricked into persisting
# sensitive data via a crafted "remember this" instruction.
_ALLOWED_MEMORY_KEYS = {"preferred_language", "city", "last_program_interest"}


def get_conversation_context(conversation_id: str, limit: int = 12) -> list[dict]:
    rows = db.get_recent_messages(conversation_id, limit=limit)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def remember(user_id: str, key: str, value: str) -> bool:
    """Store a non-sensitive fact. Returns False (no-op) for disallowed keys."""
    if key not in _ALLOWED_MEMORY_KEYS:
        return False
    if not value or not value.strip():
        return False
    db.set_user_memory(user_id=user_id, key=key, value=value.strip())
    return True


def recall_all(user_id: str) -> dict[str, str]:
    return db.get_user_memory(user_id)


def format_memory_for_prompt(user_id: str) -> str:
    memory = recall_all(user_id)
    if not memory:
        return "No prior information remembered about this user."
    lines = [f"- {k}: {v}" for k, v in memory.items()]
    return "Known (non-sensitive) facts about this user:\n" + "\n".join(lines)
