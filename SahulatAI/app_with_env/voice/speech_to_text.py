"""
Speech-to-text with multiple engine support:

  1. **groq** (DEFAULT) — Groq's Whisper-large-v3-turbo via the OpenAI-
     compatible audio transcription endpoint. Fast, accurate, free-tier,
     and handles noisy audio / accents / multilingual input far better
     than Google's free Web Speech API.  Requires GROQ_API_KEY (already
     set for the chat agent).

  2. **google_free** — SpeechRecognition's built-in free Google Web Speech
     endpoint.  No API key, but unreliable (frequent UnknownValueError
     even on clear audio).  Kept as a fallback.

  3. **sphinx** — fully offline PocketSphinx (requires extra install).

Audio format handling:
  streamlit-mic-recorder v0.0.8 can return audio bytes that are NOT a
  proper WAV container even when format="wav" is set.  We use pydub to
  normalise *any* input into a clean 16 kHz mono 16-bit WAV before
  sending it to any engine.
"""
from __future__ import annotations

import io
import logging
import os

import speech_recognition as sr
from pydub import AudioSegment

logger = logging.getLogger("sahulatai.stt")

# ── Thresholds ────────────────────────────────────────────────────────────
_MIN_DURATION_SEC = 0.5  # ignore clips shorter than 500 ms


class TranscriptionError(Exception):
    pass


# ── Audio normalisation ──────────────────────────────────────────────────
def _normalize_audio_to_wav(audio_bytes: bytes) -> bytes:
    """
    Accept audio bytes in *any* format pydub/ffmpeg can decode (wav, webm,
    ogg, mp3, raw PCM …) and return a canonical 16 kHz, mono, 16-bit PCM
    WAV that every STT engine handles reliably.
    """
    try:
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as exc:
        raise TranscriptionError(
            f"Could not decode the recorded audio ({exc}). "
            "Please make sure ffmpeg is installed and try again."
        ) from exc

    # Normalise to 16 kHz mono 16-bit — the sweet spot for speech APIs.
    segment = (
        segment
        .set_frame_rate(16000)
        .set_channels(1)
        .set_sample_width(2)  # 16-bit
    )

    duration_sec = len(segment) / 1000.0
    if duration_sec < _MIN_DURATION_SEC:
        raise TranscriptionError(
            f"Recording too short ({duration_sec:.1f}s). "
            "Please hold the button and speak for at least one second."
        )

    wav_io = io.BytesIO()
    segment.export(wav_io, format="wav")
    return wav_io.getvalue()


# ── Groq Whisper engine ──────────────────────────────────────────────────
def _transcribe_groq(wav_bytes: bytes, language: str) -> str:
    """Transcribe via Groq's Whisper-large-v3-turbo (OpenAI-compatible)."""
    from openai import OpenAI  # already a project dependency

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise TranscriptionError(
            "GROQ_API_KEY is not set. Cannot use Groq Whisper for speech-to-text."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # Map BCP-47 codes to ISO 639-1 for Whisper
    lang_map = {"en-US": "en", "ur-PK": "ur"}
    whisper_lang = lang_map.get(language, language.split("-")[0])

    try:
        transcription = client.audio.transcriptions.create(
            file=("recording.wav", io.BytesIO(wav_bytes)),
            model="whisper-large-v3-turbo",
            language=whisper_lang,
            response_format="text",
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Groq Whisper transcription failed: {exc}"
        ) from exc

    text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
    if not text:
        raise TranscriptionError(
            "Sorry, I couldn't understand that. Please speak clearly and try again."
        )
    return text


# ── Google free engine (fallback) ────────────────────────────────────────
def _transcribe_google_free(wav_bytes: bytes, language: str) -> str:
    """Transcribe via Google's free Web Speech endpoint (no API key)."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)
    except Exception as exc:
        raise TranscriptionError(
            f"Failed to process audio for Google STT ({exc})."
        ) from exc

    try:
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError as exc:
        raise TranscriptionError(
            "Sorry, I couldn't understand that. Please speak clearly and try again."
        ) from exc
    except sr.RequestError as exc:
        raise TranscriptionError(
            "Speech recognition service is unavailable (network or quota issue)."
        ) from exc


# ── Sphinx engine (offline) ──────────────────────────────────────────────
def _transcribe_sphinx(wav_bytes: bytes, language: str) -> str:
    """Transcribe via PocketSphinx (fully offline)."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_sphinx(audio_data, language=language)


# ── Public API ───────────────────────────────────────────────────────────
def transcribe_audio_bytes(audio_bytes: bytes, sample_rate: int = 16000,
                            sample_width: int = 2, language: str = "en-US",
                            engine: str = "groq") -> str:
    """
    Transcribe audio bytes to text.

    `audio_bytes` can be in *any* browser-native format (wav, webm/opus,
    ogg …). The function normalises the data into a clean WAV internally
    via pydub before passing it to the chosen STT engine.

    engine:
      - "groq"       : (default) Groq Whisper — fast, accurate, free tier.
      - "google_free": Google Web Speech — no API key, but unreliable.
      - "sphinx"     : fully offline PocketSphinx.
    """
    if not audio_bytes:
        raise TranscriptionError("No audio captured. Please try again.")

    # ── Step 1: Normalise to clean WAV ──────────────────────────────────
    wav_bytes = _normalize_audio_to_wav(audio_bytes)
    logger.info("Normalised audio: %d raw bytes → %d WAV bytes", len(audio_bytes), len(wav_bytes))

    # ── Step 2: Transcribe ──────────────────────────────────────────────
    try:
        if engine == "sphinx":
            return _transcribe_sphinx(wav_bytes, language)
        elif engine == "google_free":
            return _transcribe_google_free(wav_bytes, language)
        else:
            # Default: Groq Whisper
            return _transcribe_groq(wav_bytes, language)
    except TranscriptionError:
        raise
    except Exception as exc:  # pragma: no cover — defensive
        raise TranscriptionError(f"Unexpected transcription error: {exc}") from exc


def detect_language_hint(user_selected_language: str) -> str:
    """Map SahulatAI's language options to SpeechRecognition BCP-47 codes."""
    mapping = {
        "english": "en-US",
        "urdu": "ur-PK",
        "roman_urdu": "en-US",  # Roman Urdu is spoken with Urdu phonetics but
                                 # transcribed in Latin script; en-US tends to
                                 # produce readable Roman-Urdu-ish text in practice.
    }
    return mapping.get(user_selected_language.lower(), "en-US")

