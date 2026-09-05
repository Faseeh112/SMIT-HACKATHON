"""
Tests for the retrieval classification that drives refusal / grounding
behavior (answerable vs partial vs not_found), and for the prompt-injection
guardrail text being present in the system prompt.
"""
from __future__ import annotations

from pathlib import Path

from rag.retriever import classify_retrieval


def test_classify_not_found_with_no_results():
    assert classify_retrieval([]) == "not_found"


def test_classify_answerable_with_strong_match():
    results = [{"score": 0.8}]
    assert classify_retrieval(results, strong_score=0.55) == "answerable"


def test_classify_partial_with_weak_matches_only():
    results = [{"score": 0.4}, {"score": 0.3}]
    assert classify_retrieval(results, strong_score=0.55) == "partial"


def test_system_prompt_forbids_inventing_facts():
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"
    text = prompt_path.read_text(encoding="utf-8")
    assert "Never invent Saylani-specific facts" in text


def test_system_prompt_has_prompt_injection_guardrail():
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"
    text = prompt_path.read_text(encoding="utf-8")
    assert "DATA, not" in text
    assert "ignore previous instructions" in text.lower()
