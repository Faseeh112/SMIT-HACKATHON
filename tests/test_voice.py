"""
Voice-related tests that don't require a real microphone, audio hardware,
or network access — they test the pure-logic pieces: wake-word fuzzy
matching, and STT error handling for empty audio.
"""
from __future__ import annotations

import pytest

from voice.speech_to_text import TranscriptionError, transcribe_audio_bytes
from voice.wake_word import fuzzy_matches_wake_word


def test_wake_word_exact_match():
    assert fuzzy_matches_wake_word("hi sahulat", "Hi Sahulat") is True


def test_wake_word_close_match():
    assert fuzzy_matches_wake_word("hai sahulaat", "Hi Sahulat") is True


def test_wake_word_no_match():
    assert fuzzy_matches_wake_word("what is the weather today", "Hi Sahulat") is False


def test_wake_word_empty_transcript():
    assert fuzzy_matches_wake_word("", "Hi Sahulat") is False


def test_wake_word_phrase_embedded_in_longer_sentence():
    assert fuzzy_matches_wake_word("hi sahulat what courses do you have", "Hi Sahulat") is True


def test_transcribe_empty_audio_raises():
    with pytest.raises(TranscriptionError):
        transcribe_audio_bytes(b"")
