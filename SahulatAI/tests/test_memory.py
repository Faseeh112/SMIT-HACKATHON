"""Tests for agent memory: conversation context + isolated, non-sensitive user memory."""
from __future__ import annotations

from agent import memory


def test_remember_allowed_key_succeeds(temp_db):
    temp_db.ensure_user("u1")
    ok = memory.remember("u1", "preferred_language", "urdu")
    assert ok is True
    assert memory.recall_all("u1")["preferred_language"] == "urdu"


def test_remember_disallowed_key_is_dropped(temp_db):
    temp_db.ensure_user("u1")
    ok = memory.remember("u1", "password", "secret123")
    assert ok is False
    assert "password" not in memory.recall_all("u1")


def test_memory_is_isolated_per_user(temp_db):
    temp_db.ensure_user("u1")
    temp_db.ensure_user("u2")
    memory.remember("u1", "city", "Karachi")
    memory.remember("u2", "city", "Lahore")
    assert memory.recall_all("u1")["city"] == "Karachi"
    assert memory.recall_all("u2")["city"] == "Lahore"


def test_conversation_context_round_trip(temp_db):
    temp_db.ensure_user("u1")
    conv_id = temp_db.start_conversation("u1")
    temp_db.add_message(conv_id, "user", "Hello")
    temp_db.add_message(conv_id, "assistant", "Hi there!")
    context = memory.get_conversation_context(conv_id)
    assert context[0]["role"] == "user"
    assert context[0]["content"] == "Hello"
    assert context[1]["role"] == "assistant"


def test_format_memory_for_prompt_handles_empty(temp_db):
    temp_db.ensure_user("new_user")
    text = memory.format_memory_for_prompt("new_user")
    assert "No prior information" in text
