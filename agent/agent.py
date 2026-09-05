"""
Core agent loop: sends the conversation + tool schemas to Grok (xAI) / Groq, executes
any tool calls requested, feeds results back, and repeats until the model
returns a final text answer or MAX_TOOL_CALLS is reached.

Tool routing is entirely model-driven (function calling) — there is
no keyword-based `if "complaint" in text` branching anywhere in this file.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

from openai import OpenAI

from agent import memory
from agent.state import ActionTrace, TurnResult
from agent.tool_registry import all_schemas, execute_tool
from config.settings import settings
from rag.retriever import classify_retrieval, retrieve

logger = logging.getLogger("sahulatai.agent")

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"

_TOOL_ACTION_LABELS = {
    "search_documents": "Searched knowledge base",
    "check_eligibility": "Checked eligibility",
    "create_complaint": "Filed a complaint",
    "schedule_callback": "Scheduled a callback",
}


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    # Cached: this file doesn't change while the app is running, and
    # re-reading it from disk on every single message (every Streamlit
    # rerun) was pure wasted latency. Restart the app to pick up prompt
    # edits.
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
                                  tool_schemas: list[dict], call: FunctionCall,
                                  result: dict | list) -> tuple[Optional[str], list[FunctionCall]]:
        """Return (final_text_or_None, function_calls) after feeding back a tool result."""
        ...


class GrokClient:
    """Client for Groq / OpenAI-compatible LLM endpoint."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        self._model = model or settings.groq_chat_model
        api_key = api_key or settings.groq_api_key
        # Groq endpoint is OpenAI-compatible (https://api.groq.com/openai/v1)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.groq.com/openai/v1",
        )

    def _to_messages(self, system_prompt: str, history: list[dict]) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        return messages

    def _build_tools(self, tool_schemas: list[dict]) -> list[dict] | None:
        if not tool_schemas:
            return None
        return [{"type": "function", "function": schema} for schema in tool_schemas]

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

    def _create_completion(self, messages: list[dict], tools: list[dict] | None):
        try:
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            if "rate limit" in err_str or "429" in err_str:
                for fallback_model in ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
                    if fallback_model != self._model:
                        try:
                            logger.warning(
                                "Primary model %s rate limited (429), falling back to %s",
                                self._model,
                                fallback_model,
                            )
                            return self._client.chat.completions.create(
                                model=fallback_model,
                                messages=messages,
                                tools=tools,
                            )
                        except Exception:
                            continue
            raise

    def generate(self, system_prompt: str, history: list[dict], tool_schemas: list[dict]
                 ) -> tuple[Optional[str], list[FunctionCall]]:
        messages = self._to_messages(system_prompt, history)
        response = self._create_completion(
            messages=messages,
            tools=self._build_tools(tool_schemas),
        )
        return self._parse_response(response)

    def generate_with_tool_result(self, system_prompt: str, history: list[dict],
                                  tool_schemas: list[dict], call: FunctionCall,
                                  result: dict | list) -> tuple[Optional[str], list[FunctionCall]]:
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
        response = self._create_completion(
            messages=messages,
            tools=self._build_tools(tool_schemas),
        )
        return self._parse_response(response)


@lru_cache(maxsize=1)
def _get_default_llm() -> LLMClient:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your free "
            "Groq API key from https://console.groq.com/keys."
        )
    return GrokClient()


def _format_auto_retrieval_context(status: str, results: list[dict]) -> str:
    """
    Render a pre-fetched knowledge-base lookup as a block of context text.

    This is used to avoid a wasted network round trip: search_documents is
    a local, in-memory numpy lookup (no network call, a few milliseconds),
    so it's cheap to run automatically against the user's own message
    before ever calling the LLM. Handing the LLM these results already
    formatted like a completed search_documents call means most simple
    informational questions only need ONE Groq call instead of two
    (decide-to-search, then answer) — this is the main source of the
    "processing" delay users see, since each Groq call is a network round
    trip. The agent can still call search_documents itself mid-turn for a
    different/follow-up query (e.g. checking a second course by name);
    this pre-fetch only covers the user's latest message as typed.
    """
    payload = {
        "status": status,
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
    return (
        "AUTOMATIC KNOWLEDGE BASE LOOKUP (already run for the user's latest "
        "message below, equivalent to having called search_documents on it "
        "yourself — no need to call it again for this same question; only "
        "call it if you need to look up something else this turn):\n"
        + json.dumps(payload, ensure_ascii=False)
    )


_MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]*)\]\((?:https?://|mailto:)[^\)\s]+\)")
_RAW_URL_RE = re.compile(r"(?:https?://|www\.)\S+")


def _strip_links(text: str) -> str:
    """
    Defense-in-depth backstop: even though the system prompt tells the
    model never to include URLs/markdown links (sources are shown in a
    separate UI panel), the model has occasionally slipped and glued a
    raw URL together with a broken markdown-link fragment (e.g.
    "https://x.com/enroll[ by](https://x.com/enroll%E2%80%AFby)"), which
    renders as garbled text. Strip both patterns unconditionally so a
    prompt-compliance miss can never reach the user as broken text.
    """
    if not text:
        return text
    # Markdown links: keep the human-readable label, drop the URL.
    text = _MARKDOWN_LINK_RE.sub(lambda m: m.group(1).strip(), text)
    # Any remaining bare URLs.
    text = _RAW_URL_RE.sub("", text)
    # Tidy up artifacts left behind (double spaces, dangling brackets/dashes).
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def run_turn(user_id: str, conversation_id: str, user_message: str,
             llm: Optional[LLMClient] = None) -> TurnResult:
    """Run one full agent turn: history in, grounded answer out."""
    llm = llm or _get_default_llm()

    # Cheap, local, no network call — run it before ever hitting the LLM so
    # the common case (a plain informational question) needs only one Groq
    # round trip instead of two. See _format_auto_retrieval_context above.
    auto_results = retrieve(user_message)
    auto_status = classify_retrieval(auto_results)

    system_prompt = (
        _load_system_prompt()
        + "\n\n" + memory.format_memory_for_prompt(user_id)
        + "\n\n" + _format_auto_retrieval_context(auto_status, auto_results)
    )

    history = memory.get_conversation_context(conversation_id)
    history.append({"role": "user", "content": user_message})

    trace = ActionTrace()
    sources: list[dict] = []
    tool_calls_used = 0
    schemas = all_schemas()

    if auto_results:
        trace.log("Searched knowledge base")
        sources.extend(auto_results)

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
    final_text = _strip_links(final_text)

    # Ensure response never outputs in Hindi or Urdu script
    try:
        from voice.speech_to_text import to_roman_urdu
        final_text = to_roman_urdu(final_text)
    except Exception:
        pass

    return TurnResult(
        answer_text=final_text,
        action_trace=trace,
        sources=sources,
        tool_calls_used=tool_calls_used,
    )
