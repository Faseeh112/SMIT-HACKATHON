"""
Core agent loop: sends the conversation + tool schemas to Grok (xAI), executes
any tool calls Grok requests, feeds results back, and repeats until Grok
returns a final text answer or MAX_TOOL_CALLS is reached.

Tool routing is entirely model-driven (Grok function calling) — there is
no keyword-based `if "complaint" in text` branching anywhere in this file.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from agent import memory
from agent.state import ActionTrace, TurnResult
from agent.tool_registry import all_schemas, execute_tool
from config.settings import settings

logger = logging.getLogger("sahulatai.agent")

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"

_TOOL_ACTION_LABELS = {
    "search_documents": "Searched knowledge base",
    "check_eligibility": "Checked eligibility",
    "create_complaint": "Filed a complaint",
    "schedule_callback": "Scheduled a callback",
}


def _load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class FunctionCall:
    name: str
    args: dict
    call_id: Optional[str] = None  # provider-assigned id, needed to report the result back


class LLMClient(Protocol):
    """Interface the agent needs from an LLM backend — lets tests inject a fake."""

    def generate(self, system_prompt: str, history: list[dict], tool_schemas: list[dict]
                 ) -> tuple[Optional[str], list[FunctionCall]]:
        """Return (final_text_or_None, function_calls). Exactly one should be non-empty."""
        ...

    def generate_with_tool_result(self, system_prompt: str, history: list[dict],
                                   tool_schemas: list[dict], call: FunctionCall, result: dict
                                   ) -> tuple[Optional[str], list[FunctionCall]]:
        ...


class GrokClient:
    """Real Groq-backed implementation, via the OpenAI-compatible endpoint.

    Groq (https://groq.com) offers a free, fast-inference API that is
    OpenAI-compatible (https://api.groq.com/openai/v1), so this uses the
    standard `openai` client pointed at Groq's base URL rather than a
    Groq-specific SDK — this keeps the request/response shapes (messages,
    tools, tool_calls) identical to the widely-documented OpenAI format.
    (Note: Groq is a different company from xAI's "Grok" model.)
    """

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
        self._model = settings.groq_chat_model

    def _build_tools(self, tool_schemas: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["parameters"],
                },
            }
            for s in tool_schemas
        ]

    def _to_messages(self, system_prompt: str, history: list[dict]) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        return messages

    def _parse_response(self, response) -> tuple[Optional[str], list[FunctionCall]]:
        message = response.choices[0].message
        if message.tool_calls:
            calls = [
                FunctionCall(
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments or "{}"),
                    call_id=tc.id,
                )
                for tc in message.tool_calls
            ]
            return None, calls
        return (message.content or "").strip(), []

    def generate(self, system_prompt, history, tool_schemas):
        messages = self._to_messages(system_prompt, history)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=self._build_tools(tool_schemas),
        )
        return self._parse_response(response)

    def generate_with_tool_result(self, system_prompt, history, tool_schemas, call, result):
        messages = self._to_messages(system_prompt, history)
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.args)},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": json.dumps(result),
            }
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=self._build_tools(tool_schemas),
        )
        return self._parse_response(response)


def _get_default_llm() -> LLMClient:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your free "
            "Groq API key from https://console.groq.com/keys."
        )
    return GrokClient()


def run_turn(user_id: str, conversation_id: str, user_message: str,
             llm: Optional[LLMClient] = None) -> TurnResult:
    """Run one full agent turn: history in, grounded answer out."""
    llm = llm or _get_default_llm()
    system_prompt = _load_system_prompt() + "\n\n" + memory.format_memory_for_prompt(user_id)

    history = memory.get_conversation_context(conversation_id)
    history.append({"role": "user", "content": user_message})

    trace = ActionTrace()
    sources: list[dict] = []
    tool_calls_used = 0
    schemas = all_schemas()

    text, calls = llm.generate(system_prompt, history, schemas)

    while calls and tool_calls_used < settings.max_tool_calls:
        call = calls[0]  # process sequentially, one at a time, for a clear action trace
        tool_calls_used += 1
        result = execute_tool(call.name, call.args, user_id=user_id)
        label = _TOOL_ACTION_LABELS.get(call.name, call.name)
        trace.log(label)

        if call.name == "search_documents" and isinstance(result, dict):
            sources.extend(result.get("results", []))

        text, calls = llm.generate_with_tool_result(system_prompt, history, schemas, call, result)

    if calls and tool_calls_used >= settings.max_tool_calls:
        logger.warning("MAX_TOOL_CALLS reached for conversation %s", conversation_id)
        text = text or (
            "I've done what I can for now, but this needs a bit more help from our "
            "staff to fully resolve. Would you like me to schedule a callback?"
        )

    final_text = text or "I'm sorry, I couldn't generate a response. Please try again."

    return TurnResult(
        answer_text=final_text,
        action_trace=trace,
        sources=sources,
        tool_calls_used=tool_calls_used,
    )
