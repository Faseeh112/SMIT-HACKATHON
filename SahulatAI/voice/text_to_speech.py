"""
Text-to-speech using pyttsx3 — fully offline, free, no API key.

pyttsx3 speaks synchronously through the local OS speech engine (SAPI5 on
Windows). Since Streamlit re-runs the whole script on each interaction, we
create a fresh engine per call rather than caching a long-lived engine
object (pyttsx3 engines are not safe to reuse across Streamlit re-runs).
"""
from __future__ import annotations

import logging

import pyttsx3

logger = logging.getLogger("sahulatai.tts")


class SpeechError(Exception):
    pass


def list_voices() -> list[dict]:
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.stop()
    return [{"id": v.id, "name": v.name, "languages": getattr(v, "languages", [])} for v in voices]


def find_urdu_capable_voice(voices: list[dict]) -> str | None:
    for v in voices:
        name = v["name"].lower()
        if "urdu" in name or "ur-pk" in name or "hindi" in name:
            return v["id"]
    return None


def speak(text: str, voice_id: str | None = None, rate: int = 175, volume: float = 1.0) -> bool:
    """
    Speak `text` aloud using pyttsx3. Returns True on success, False if no
    usable voice/engine is available (caller should fall back to text-only
    and show a warning, per spec — never crash the app).
    """
    if not text or not text.strip():
        return False
    try:
        engine = pyttsx3.init()
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.setProperty("rate", rate)
        engine.setProperty("volume", max(0.0, min(1.0, volume)))
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception as exc:
        logger.warning("TTS failed (%s). Falling back to text-only response.", exc)
        return False
