"""
Text-to-speech supporting Urdu and English.

Priority order (fastest first):
  1. Windows SAPI via win32com (instant, ~30ms)
  2. pyttsx3 offline engine (fast, ~100ms)
  3. gTTS online (slower, 1-3s network call) — fallback only
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import threading
from typing import Callable

logger = logging.getLogger("sahulatai.tts")


class SpeechError(Exception):
    pass


def _clean_text_for_speech(text: str) -> str:
    """Strip markdown links, symbols, headers, bullet points and emojis for clear speech."""
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[`*#_~|]", "", text)
    text = re.sub(r"\n+", ". ", text)
    return text.strip()


def detect_script_language(text: str) -> str:
    """Returns 'ur' if text contains Urdu/Arabic unicode block, else 'en'."""
    if re.search(r"[\u0600-\u06FF]", text):
        return "ur"
    return "en"


def _speak_sapi(text: str, rate: int = 175, volume: float = 1.0) -> bool:
    """Instant offline TTS via Windows SAPI (fastest, ~30ms latency)."""
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Volume = int(max(0.0, min(1.0, volume)) * 100)
        # SAPI rate is roughly -10 to +10; default rate 175 maps to 1
        sapi_rate = int((rate - 175) / 25)
        speaker.Rate = max(-10, min(10, sapi_rate))
        speaker.Speak(text)
        return True
    except Exception as exc:
        logger.debug("SAPI failed: %s", exc)
        return False


def _speak_pyttsx3(text: str, voice_id: str | None = None, rate: int = 175, volume: float = 1.0) -> bool:
    """Offline TTS via pyttsx3 (~100ms latency)."""
    try:
        import pyttsx3

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
        logger.warning("pyttsx3 failed: %s", exc)
        return False


def _speak_gtts(text: str) -> bool:
    """Online TTS via gTTS (1-3s network latency). Last resort fallback."""
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        import winsound

        lang = detect_script_language(text)
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        sound = AudioSegment.from_file(mp3_fp, format="mp3")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        sound.export(wav_path, format="wav")
        try:
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
        return True
    except Exception as exc:
        logger.warning("gTTS playback failed (%s).", exc)
        return False


def speak(
    text: str,
    voice_id: str | None = None,
    rate: int = 175,
    volume: float = 1.0,
    on_start: Callable[[], None] | None = None,
    on_done: Callable[[], None] | None = None,
) -> bool:
    """
    Speak `text` aloud. Tries engines in order of speed:
    1. Windows SAPI (instant)
    2. pyttsx3 (fast offline)
    3. gTTS (slow online fallback)
    """
    cleaned = _clean_text_for_speech(text)
    if not cleaned:
        return False

    if on_start:
        try:
            on_start()
        except Exception:
            pass

    try:
        # 1. SAPI — instant (~30ms)
        if _speak_sapi(cleaned, rate=rate, volume=volume):
            return True
        # 2. pyttsx3 — fast offline
        if _speak_pyttsx3(cleaned, voice_id=voice_id, rate=rate, volume=volume):
            return True
        # 3. gTTS — slow online fallback
        return _speak_gtts(cleaned)
    finally:
        if on_done:
            try:
                on_done()
            except Exception:
                pass


def speak_async(
    text: str,
    voice_id: str | None = None,
    rate: int = 175,
    volume: float = 1.0,
    on_start: Callable[[], None] | None = None,
    on_done: Callable[[], None] | None = None,
) -> threading.Thread:
    """Fire-and-forget speech playback on a background daemon thread."""
    thread = threading.Thread(
        target=speak,
        args=(text,),
        kwargs={
            "voice_id": voice_id,
            "rate": rate,
            "volume": volume,
            "on_start": on_start,
            "on_done": on_done,
        },
        daemon=True,
    )
    thread.start()
    return thread
