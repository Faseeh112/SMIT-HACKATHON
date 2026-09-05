"""
Language-support tests. Full English/Urdu/Roman Urdu generation quality
depends on the live Grok model, so these tests check the pieces that are
deterministic and testable offline: the STT language-hint mapping, and that
the system prompt instructs language-mirroring (a static content check).
"""
from __future__ import annotations

from pathlib import Path

from voice.speech_to_text import detect_language_hint


def test_language_hint_mapping():
    assert detect_language_hint("english") == "en-US"
    assert detect_language_hint("urdu") == "ur-PK"
    assert detect_language_hint("roman_urdu") == "en-US"
    assert detect_language_hint("unknown") == "en-US"  # safe default


def test_system_prompt_requires_language_mirroring():
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"
    text = prompt_path.read_text(encoding="utf-8")
    assert "Roman Urdu" in text
    assert "Urdu" in text
    assert "same language" in text.lower()
