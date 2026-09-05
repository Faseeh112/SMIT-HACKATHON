# """
# Speech-to-text with multiple engine support:

#   1. **groq** (DEFAULT) — Groq's Whisper-large-v3-turbo via the OpenAI-
#      compatible audio transcription endpoint. Fast, accurate, free-tier,
#      and handles noisy audio / accents / multilingual input far better
#      than Google's free Web Speech API.  Requires GROQ_API_KEY (already
#      set for the chat agent).

#   2. **google_free** — SpeechRecognition's built-in free Google Web Speech
#      endpoint.  No API key, but unreliable (frequent UnknownValueError
#      even on clear audio).  Kept as a fallback.

#   3. **sphinx** — fully offline PocketSphinx (requires extra install).

# Audio format handling:
#   streamlit-mic-recorder v0.0.8 can return audio bytes that are NOT a
#   proper WAV container even when format="wav" is set.  We use pydub to
#   normalise *any* input into a clean 16 kHz mono 16-bit WAV before
#   sending it to any engine.
# """
# from __future__ import annotations

# import io
# import logging
# import os

# import speech_recognition as sr
# from pydub import AudioSegment

# logger = logging.getLogger("sahulatai.stt")

# # ── Thresholds ────────────────────────────────────────────────────────────
# _MIN_DURATION_SEC = 0.5  # ignore clips shorter than 500 ms


# class TranscriptionError(Exception):
#     pass


# # ── Audio normalisation ──────────────────────────────────────────────────
# def _normalize_audio_to_wav(audio_bytes: bytes) -> bytes:
#     """
#     Accept audio bytes in *any* format pydub/ffmpeg can decode (wav, webm,
#     ogg, mp3, raw PCM …) and return a canonical 16 kHz, mono, 16-bit PCM
#     WAV that every STT engine handles reliably.
#     """
#     try:
#         segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
#     except Exception as exc:
#         raise TranscriptionError(
#             f"Could not decode the recorded audio ({exc}). "
#             "Please make sure ffmpeg is installed and try again."
#         ) from exc

#     # Normalise to 16 kHz mono 16-bit — the sweet spot for speech APIs.
#     segment = (
#         segment
#         .set_frame_rate(16000)
#         .set_channels(1)
#         .set_sample_width(2)  # 16-bit
#     )

#     duration_sec = len(segment) / 1000.0
#     if duration_sec < _MIN_DURATION_SEC:
#         raise TranscriptionError(
#             f"Recording too short ({duration_sec:.1f}s). "
#             "Please hold the button and speak for at least one second."
#         )

#     wav_io = io.BytesIO()
#     segment.export(wav_io, format="wav")
#     return wav_io.getvalue()


# # ── Groq Whisper engine ──────────────────────────────────────────────────
# def _transcribe_groq(wav_bytes: bytes, language: str) -> str:
#     """Transcribe via Groq's Whisper-large-v3-turbo (OpenAI-compatible)."""
#     from openai import OpenAI  # already a project dependency

#     api_key = os.getenv("GROQ_API_KEY", "")
#     if not api_key:
#         raise TranscriptionError(
#             "GROQ_API_KEY is not set. Cannot use Groq Whisper for speech-to-text."
#         )

#     client = OpenAI(
#         api_key=api_key,
#         base_url="https://api.groq.com/openai/v1",
#     )

#     # Map BCP-47 codes to ISO 639-1 for Whisper
#     lang_map = {"en-US": "en", "ur-PK": "ur"}
#     whisper_lang = lang_map.get(language, language.split("-")[0])

#     try:
#         transcription = client.audio.transcriptions.create(
#             file=("recording.wav", io.BytesIO(wav_bytes)),
#             model="whisper-large-v3-turbo",
#             language=whisper_lang,
#             response_format="text",
#         )
#     except Exception as exc:
#         raise TranscriptionError(
#             f"Groq Whisper transcription failed: {exc}"
#         ) from exc

#     text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
#     if not text:
#         raise TranscriptionError(
#             "Sorry, I couldn't understand that. Please speak clearly and try again."
#         )
#     return text


# # ── Google free engine (fallback) ────────────────────────────────────────
# def _transcribe_google_free(wav_bytes: bytes, language: str) -> str:
#     """Transcribe via Google's free Web Speech endpoint (no API key)."""
#     recognizer = sr.Recognizer()
#     try:
#         with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
#             audio_data = recognizer.record(source)
#     except Exception as exc:
#         raise TranscriptionError(
#             f"Failed to process audio for Google STT ({exc})."
#         ) from exc

#     try:
#         return recognizer.recognize_google(audio_data, language=language)
#     except sr.UnknownValueError as exc:
#         raise TranscriptionError(
#             "Sorry, I couldn't understand that. Please speak clearly and try again."
#         ) from exc
#     except sr.RequestError as exc:
#         raise TranscriptionError(
#             "Speech recognition service is unavailable (network or quota issue)."
#         ) from exc


# # ── Sphinx engine (offline) ──────────────────────────────────────────────
# def _transcribe_sphinx(wav_bytes: bytes, language: str) -> str:
#     """Transcribe via PocketSphinx (fully offline)."""
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
#         audio_data = recognizer.record(source)
#     return recognizer.recognize_sphinx(audio_data, language=language)


# # ── Public API ───────────────────────────────────────────────────────────
# def transcribe_audio_bytes(audio_bytes: bytes, sample_rate: int = 16000,
#                             sample_width: int = 2, language: str = "en-US",
#                             engine: str = "groq") -> str:
#     """
#     Transcribe audio bytes to text.

#     `audio_bytes` can be in *any* browser-native format (wav, webm/opus,
#     ogg …). The function normalises the data into a clean WAV internally
#     via pydub before passing it to the chosen STT engine.

#     engine:
#       - "groq"       : (default) Groq Whisper — fast, accurate, free tier.
#       - "google_free": Google Web Speech — no API key, but unreliable.
#       - "sphinx"     : fully offline PocketSphinx.
#     """
#     if not audio_bytes:
#         raise TranscriptionError("No audio captured. Please try again.")

#     # ── Step 1: Normalise to clean WAV ──────────────────────────────────
#     wav_bytes = _normalize_audio_to_wav(audio_bytes)
#     logger.info("Normalised audio: %d raw bytes → %d WAV bytes", len(audio_bytes), len(wav_bytes))

#     # ── Step 2: Transcribe ──────────────────────────────────────────────
#     try:
#         if engine == "sphinx":
#             return _transcribe_sphinx(wav_bytes, language)
#         elif engine == "google_free":
#             return _transcribe_google_free(wav_bytes, language)
#         else:
#             # Default: Groq Whisper
#             return _transcribe_groq(wav_bytes, language)
#     except TranscriptionError:
#         raise
#     except Exception as exc:  # pragma: no cover — defensive
#         raise TranscriptionError(f"Unexpected transcription error: {exc}") from exc


# def detect_language_hint(user_selected_language: str) -> str:
#     """Map SahulatAI's language options to SpeechRecognition BCP-47 codes."""
#     mapping = {
#         "english": "en-US",
#         "urdu": "ur-PK",
#         "roman_urdu": "en-US",  # Roman Urdu is spoken with Urdu phonetics but
#                                  # transcribed in Latin script; en-US tends to
#                                  # produce readable Roman-Urdu-ish text in practice.
#     }
#     return mapping.get(user_selected_language.lower(), "en-US")

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
import re
from typing import Optional

import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger("sahulatai.stt")

# ── Thresholds ────────────────────────────────────────────────────────────
_MIN_DURATION_SEC = 0.5  # ignore clips shorter than 500 ms

# Whisper (and Groq's hosted copy of it) is well known to *hallucinate* a
# plausible-sounding phrase — most commonly "Thank you." — when it's fed
# silence, background noise, or near-silent audio instead of real speech.
# Two independent guards against that:
#   1. Reject clips that have no genuinely loud chunk anywhere in them
#      (cheap, and stops the hallucination from being generated at all).
#   2. When the API does return per-segment confidence (verbose_json), drop
#      any segment Whisper itself flagged as "probably not speech".
#
# Guard #1 deliberately looks for the loudest *chunk*, not the clip's
# average loudness — a click-to-talk recording naturally has quiet padding
# before/after the words (time between pressing the button and speaking),
# which drags the average dBFS down even when the speech itself is clear.
_SILENCE_CHUNK_MS = 200          # scan in 200ms windows
_SILENCE_THRESH_DBFS = -38.0     # reject faint background room hiss / breathing
_MIN_SPEECH_MS = 200             # need at least 200ms non-silent audio total
_NO_SPEECH_PROB_THRESHOLD = 0.85 # only drop if Whisper is highly confident (>85%) it's silence
_AVG_LOGPROB_THRESHOLD = -2.0    # allow proper nouns/Urdu transliterations without false rejection

_KNOWN_HALLUCINATIONS = {
    "thank you", "thank you.", "thank you!", "thank you very much.", "thank you thank you",
    "thank you. thank you.", "thanks for watching", "thanks for watching.", "thanks", "you",
    "bye", "bye.", "subscribe", "please subscribe", "subtitles by", "subtitles",
    "silence", "you're welcome", "watching", "amara.org",
    "ありがとうございました", "ご視聴ありがとうございました", "ありがとう"
}


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

    # Guard #1: if there's no genuinely loud chunk anywhere in the clip,
    # don't send it to Whisper at all — this is what causes it to
    # hallucinate text like "Thank you." instead of correctly saying
    # "I heard nothing".
    nonsilent_ranges = detect_nonsilent(
        segment, min_silence_len=_SILENCE_CHUNK_MS, silence_thresh=_SILENCE_THRESH_DBFS
    )
    speech_ms = sum(end - start for start, end in nonsilent_ranges)
    if speech_ms < _MIN_SPEECH_MS:
        raise TranscriptionError(
            "No speech detected — the recording was too quiet. "
            "Please speak closer to the microphone and try again."
        )

    wav_io = io.BytesIO()
    segment.export(wav_io, format="wav")
    return wav_io.getvalue()


from config.settings import settings


URDU_MAP = {
    'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's',
    'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ذ': 'z',
    'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's',
    'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
    'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'o',
    'ہ': 'h', 'ۂ': 'h', 'ۃ': 'h', 'ھ': 'h', 'ء': '', 'ی': 'i', 'ے': 'e',
    'ئ': 'i', 'ؤ': 'o', 'لا': 'la', '؟': '?'
}

DEVA_MAP = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ँ': 'n', '्': '',
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', '़': '', '।': '.', '॥': '.'
}

SCRIPT_WORD_MAP = {
    'سیلانی': 'Saylani', 'سہولت': 'Sahulat', 'پاکستان': 'Pakistan', 'کورس': 'course',
    'کورسز': 'courses', 'داخلہ': 'admission', 'داخلے': 'admissions', 'کون': 'kon',
    'کونسا': 'konsa', 'کونسے': 'konse', 'کونسی': 'konsi', 'کیا': 'kya', 'کب': 'kab',
    'کہاں': 'kahan', 'کیسے': 'kaise', 'کتنی': 'kitni', 'کتنا': 'kitna', 'فیس': 'fees',
    'سروسز': 'services', 'فراہم': 'provide', 'پروائی': 'provide', 'پروائیڈ': 'provide',
    'کرتا': 'karta', 'کرتے': 'karte', 'کرتی': 'karti', 'ہے': 'hai', 'ہیں': 'hain',
    'میں': 'mein', 'سے': 'se', 'کا': 'ka', 'کی': 'ki', 'کے': 'ke', 'کو': 'ko', 'پر': 'par',
    'اور': 'aur', 'سلام': 'salam', 'ہائی': 'Hi', 'ہیلو': 'Hello',
    # Hindi common vocabulary
    'सायलानी': 'Saylani', 'सेलानी': 'Saylani', 'सहूलत': 'Sahulat', 'सहूल': 'Sahulat',
    'पुलत': 'Sahulat', 'पाकिस्तान': 'Pakistan', 'पाकस्तान': 'Pakistan', 'पलस्तान': 'Pakistan',
    'पंकस्तान': 'Pakistan', 'कोर्स': 'course', 'एडमिशन': 'admission', 'कौन': 'kaun',
    'क्या': 'kya', 'कब': 'kab', 'कहाँ': 'kahan', 'कैसे': 'kaise', 'कितनी': 'kitni',
    'फीस': 'fees', 'सर्विसेज': 'services', 'सर्विस': 'service', 'सर्शिस': 'service',
    'प्रवाइड': 'provide', 'प्रवादा': 'provide', 'प्रवाइद': 'provide', 'करता': 'karta',
    'करती': 'karti', 'है': 'hai', 'हैं': 'hain', 'में': 'mein', 'से': 'se', 'का': 'ka',
    'کی': 'ki', 'के': 'ke', 'को': 'ko', 'पर': 'par', 'اور': 'aur', 'नमस्ते': 'Namaste',
    'हाई': 'Hi', 'हाय': 'Hi', 'आए': 'Hi', 'हेलो': 'Hello'
}


def to_roman_urdu(text: str, client: Optional[object] = None) -> str:
    """Instantly convert any Devanagari (Hindi) or Arabic-script Urdu text to clean Roman Urdu."""
    if not text:
        return text

    # Strip Whisper special tokens like <|hi|>, <|en|>
    text = re.sub(r"<\|[^|>]*\|>", " ", text).strip()

    if not re.search(r"[\u0900-\u097F\u0600-\u06FF]", text):
        return text

    words = text.split()
    converted_words = []
    for w in words:
        clean_w = re.sub(r"[^\w\u0600-\u06FF\u0900-\u097F]", "", w)
        if clean_w in SCRIPT_WORD_MAP:
            converted_words.append(SCRIPT_WORD_MAP[clean_w])
        elif any("\u0600" <= c <= "\u06FF" for c in w):
            cw = "".join(URDU_MAP.get(c, c) for c in w)
            converted_words.append(cw)
        elif any("\u0900" <= c <= "\u097F" for c in w):
            cw = "".join(DEVA_MAP.get(c, c) for c in w)
            converted_words.append(cw)
        else:
            converted_words.append(w)

    res = " ".join(converted_words)
    res = re.sub(r"\s+", " ", res).strip()
    return res


def _transcribe_groq(wav_bytes: bytes, language: str | None = None) -> str:
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

    # Map BCP-47 / names to ISO 639-1 for Whisper; if None or auto, Whisper auto-detects English / Urdu
    whisper_lang = None
    if language and language.lower() not in ("auto", "none", ""):
        lang_map = {"en-us": "en", "ur-pk": "ur", "urdu": "ur", "english": "en", "roman_urdu": "en", "en": "en", "ur": "ur"}
        whisper_lang = lang_map.get(language.lower(), language.split("-")[0])

    create_kwargs = {
        "file": ("recording.wav", io.BytesIO(wav_bytes)),
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "prompt": "Roman Urdu, Urdu, English, Saylani, SMIT, admissions, dakhla, fees, courses, timing, contact, eligibility.",
    }
    if whisper_lang:
        create_kwargs["language"] = whisper_lang

    try:
        transcription = client.audio.transcriptions.create(**create_kwargs)
    except Exception as exc:
        raise TranscriptionError(
            f"Groq Whisper transcription failed: {exc}"
        ) from exc

    segments = getattr(transcription, "segments", None)
    if segments:
        kept_words = []
        for seg in segments:
            # segments can come back as objects or plain dicts depending on
            # SDK version — read both the same way.
            get = (lambda k, d=None: getattr(seg, k, d)) if not isinstance(seg, dict) else seg.get
            no_speech_prob = get("no_speech_prob", 0.0) or 0.0
            avg_logprob = get("avg_logprob", 0.0) or 0.0
            seg_text = (get("text", "") or "").strip()
            if no_speech_prob > _NO_SPEECH_PROB_THRESHOLD and avg_logprob < _AVG_LOGPROB_THRESHOLD:
                logger.info(
                    "Dropped likely-hallucinated STT segment %r (no_speech_prob=%.2f, avg_logprob=%.2f)",
                    seg_text, no_speech_prob, avg_logprob,
                )
                continue
            if seg_text:
                kept_words.append(seg_text)
        text = " ".join(kept_words).strip()
    else:
        # Fallback for SDK/response shapes without segment-level detail.
        raw_text = getattr(transcription, "text", None)
        text = (raw_text or str(transcription)).strip()

    if not text:
        raise TranscriptionError(
            "Sorry, I couldn't understand that. Please speak clearly and try again."
        )

    # Filter out Whisper silence/music hallucinations
    cleaned_lower = text.lower().strip(" .!?,;:-—_")
    if cleaned_lower in _KNOWN_HALLUCINATIONS or cleaned_lower in {"thank you", "thank you thank you", "thanks", "you", "subscribe"}:
        logger.debug("Filtered out Whisper hallucination: %r", text)
        raise TranscriptionError("Silence / non-speech hallucination dropped.")

    # Convert any Hindi (Devanagari) or Urdu script into Roman Urdu
    return to_roman_urdu(text, client=client)


# ── Google free engine (fallback) ────────────────────────────────────────
def _transcribe_google_free(wav_bytes: bytes, language: str = "en-US") -> str:
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
        return recognizer.recognize_google(audio_data, language=language or "en-US")
    except sr.UnknownValueError as exc:
        raise TranscriptionError(
            "Sorry, I couldn't understand that. Please speak clearly and try again."
        ) from exc
    except sr.RequestError as exc:
        raise TranscriptionError(
            "Speech recognition service is unavailable (network or quota issue)."
        ) from exc


# ── Sphinx engine (offline) ──────────────────────────────────────────────
def _transcribe_sphinx(wav_bytes: bytes, language: str = "en-US") -> str:
    """Transcribe via PocketSphinx (fully offline)."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_sphinx(audio_data, language=language)


# ── Public API ───────────────────────────────────────────────────────────
def transcribe_audio_bytes(audio_bytes: bytes, sample_rate: int = 16000,
                            sample_width: int = 2, language: str | None = None,
                            engine: str = "groq") -> str:
    """
    Transcribe audio bytes to text. Supports English and Urdu voice input.

    `audio_bytes` can be in *any* browser-native format (wav, webm/opus,
    ogg …). The function normalises the data into a clean WAV internally
    via pydub before passing it to the chosen STT engine.

    engine:
      - "groq"       : (default) Groq Whisper — fast, accurate, auto-detects Urdu/English.
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
            return _transcribe_sphinx(wav_bytes, language or "en-US")
        elif engine == "google_free":
            return _transcribe_google_free(wav_bytes, language or "en-US")
        else:
            # Default: Groq Whisper (auto-detects English or Urdu if language is None)
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