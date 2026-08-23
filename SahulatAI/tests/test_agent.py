"""
Tests for the agent orchestration loop using a fake LLM client, so these
tests run with zero network calls and zero Grok quota usage.
"""
from __future__ import annotations

from agent.agent import FunctionCall, run_turn


class FakeLLM:
    """Scripts a fixed sequence of (text_or_None, calls) responses."""

    def __init__(self, script):
        self._script = list(script)

    def generate(self, system_prompt, history, tool_schemas):
        return self._script.pop(0)

    def generate_with_tool_result(self, system_prompt, history, tool_schemas, call, result):
        return self._script.pop(0)


def test_agent_plain_answer_no_tools(temp_db):
    temp_db.ensure_user("u1")
    conv_id = temp_db.start_conversation("u1")
    llm = FakeLLM([("Hello! How can I help?", [])])

    result = run_turn(user_id="u1", conversation_id=conv_id, user_message="Hi", llm=llm)

    assert result.answer_text == "Hello! How can I help?"
    assert result.tool_calls_used == 0
    assert result.action_trace.entries == []


def test_agent_single_tool_call_flow(temp_db, monkeypatch):
    temp_db.ensure_user("u1")
    conv_id = temp_db.start_conversation("u1")

    fake_result = {"status": "answerable", "results": [{"text": "x", "source_url": "u",
                                                          "source_title": "t", "source_type": "official_website",
                                                          "score": 0.9}]}
    monkeypatch.setattr(
        "agent.agent.execute_tool",
        lambda name, args, user_id: fake_result,
    )

    llm = FakeLLM([
        (None, [FunctionCall(name="search_documents", args={"query": "courses"})]),
        ("Here is what I found about courses.", []),
    ])

    result = run_turn(user_id="u1", conversation_id=conv_id, user_message="What courses exist?", llm=llm)

    assert result.answer_text == "Here is what I found about courses."
    assert result.tool_calls_used == 1
    assert "Searched knowledge base" in result.action_trace.entries
    assert len(result.sources) == 1


def test_agent_respects_max_tool_calls(temp_db, monkeypatch):
    temp_db.ensure_user("u1")
    conv_id = temp_db.start_conversation("u1")

    monkeypatch.setattr("agent.agent.execute_tool", lambda name, args, user_id: {"status": "not_found", "results": []})

    from config.settings import settings
    original_max = settings.max_tool_calls
    object.__setattr__(settings, "max_tool_calls", 2)
    try:
        # LLM keeps requesting more tool calls forever — agent must stop at the limit.
        script = [(None, [FunctionCall(name="search_documents", args={"query": "x"})])] * 5
        llm = FakeLLM(script)
        result = run_turn(user_id="u1", conversation_id=conv_id, user_message="loop please", llm=llm)
        assert result.tool_calls_used == 2
    finally:
        object.__setattr__(settings, "max_tool_calls", original_max)
