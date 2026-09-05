"""
Hands-free wake-word listening — Siri-style flow:

  1. Background thread continuously listens via SpeechRecognition.
  2. User says "Hi Sahulat" → system "arms" and waits for the command.
  3. User speaks command → command goes to queue → agent processes it.
  4. Agent replies with TTS → mic is muted during playback.
  5. After TTS finishes → system goes back to "waiting for Hi Sahulat".
  6. Nothing else triggers until user says "Hi Sahulat" again.

The listener object is created once via st.cache_resource in app.py
so it survives Streamlit reruns; only start()/stop() are called per toggle.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import speech_recognition as sr

from config.settings import settings
from voice.speech_to_text import TranscriptionError, transcribe_audio_bytes
from voice.wake_word import fuzzy_matches_wake_word

logger = logging.getLogger("sahulatai.hands_free")

# ── Tuning constants ──────────────────────────────────────────────────────
_ARM_WINDOW_SECONDS = 8.0   # seconds to wait for user's question after bare wake word
_POST_COMMAND_COOLDOWN = 2.0 # seconds to ignore audio after command queued (prevents TTS echo)


@dataclass
class HandsFreeState:
    """Shared, thread-safe state between the background listener thread and
    the main Streamlit thread. One instance per user session (cached)."""
    heard_queue: "queue.Queue[str]" = field(default_factory=queue.Queue)
    status_queue: "queue.Queue[str]" = field(default_factory=queue.Queue)
    active: bool = False
    _recognizer: sr.Recognizer = field(default_factory=sr.Recognizer)
    _stop_fn: Optional[Callable[..., None]] = None
    _armed_until: float = 0.0
    _muted_until: float = 0.0
    _is_speaking: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mute(self, seconds: float) -> None:
        """Silence the listener for *seconds*."""
        with self._lock:
            self._muted_until = max(self._muted_until, time.monotonic() + seconds)
        logger.info("Hands-free muted for %.1fs", seconds)

    def unmute(self) -> None:
        with self._lock:
            self._muted_until = 0.0
            self._is_speaking = False
        logger.info("Hands-free unmuted")

    def set_speaking(self, speaking: bool, cooldown: float = 2.0) -> None:
        """Called when TTS starts or finishes speaking."""
        with self._lock:
            self._is_speaking = speaking
            if speaking:
                self._muted_until = time.monotonic() + 300.0  # stay muted while speaking
            else:
                self._muted_until = time.monotonic() + cooldown  # cooldown after done
        logger.info("Hands-free set_speaking(%s, cooldown=%.1f)", speaking, cooldown)

    @property
    def is_muted(self) -> bool:
        with self._lock:
            return self._is_speaking or time.monotonic() < self._muted_until

    @property
    def is_armed(self) -> bool:
        with self._lock:
            return time.monotonic() < self._armed_until


def _contains_wake_word(transcript: str) -> bool:
    """Fast check: does the transcript contain the wake word?"""
    if not transcript:
        return False
    return fuzzy_matches_wake_word(transcript, settings.wake_word, threshold=0.70)


def _extract_command_after_wake_word(transcript: str) -> Optional[str]:
    """
    If the transcript starts with the wake word followed by a command,
    return the command part. If it's just the wake word alone, return "".
    If no wake word found, return None.

    Examples:
      "Hi Sahulat what is SMIT" → "what is SMIT"
      "Hi Sahulat"              → ""
      "random speech"           → None
    """
    if not transcript or not transcript.strip():
        return None

    raw = transcript.strip()
    raw_lower = raw.lower()

    # Known wake word prefixes to check
    wake_word_lower = settings.wake_word.lower()
    prefixes = [wake_word_lower]
    # Add common variations
    for prefix in ["hi sahulat", "hey sahulat", "hai sahulat", "hello sahulat",
                   "salam sahulat", "hi sahlat", "hi sahoolat", "high sahulat",
                   "aye sahulat"]:
        if prefix not in prefixes:
            prefixes.append(prefix)

    # Check if transcript starts with any wake word prefix
    for prefix in prefixes:
        if raw_lower.startswith(prefix):
            remainder = raw[len(prefix):].strip(" ,.:;!?-—")
            # Strip trailing "ai" if it's just that
            if remainder.lower() in ("ai", "a i", ""):
                return ""
            if remainder.lower().startswith("ai "):
                remainder = remainder[3:].strip()
            return remainder if remainder else ""

    # Check if the whole transcript IS the wake word (fuzzy)
    if fuzzy_matches_wake_word(raw, settings.wake_word, threshold=0.72):
        return ""

    return None


def _make_callback(state: HandsFreeState):
    def _callback(_recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
        # ── Guard: skip while muted (TTS playback / cooldown) ──
        if state.is_muted:
            logger.debug("Hands-free: dropping audio (muted)")
            return

        try:
            transcript = transcribe_audio_bytes(audio.get_wav_data(), engine="groq")
            logger.info("Hands-free heard: %r", transcript)
        except TranscriptionError as exc:
            logger.debug("Hands-free dropped non-speech: %s", exc)
            return
        except Exception:
            logger.exception("Hands-free transcription failed")
            return

        # ── State 1: Armed (waiting for user's question after wake word) ──
        with state._lock:
            armed = time.monotonic() < state._armed_until
            if armed:
                state._armed_until = 0.0  # disarm immediately

        if armed:
            cmd = transcript.strip()
            if cmd:
                logger.info("Hands-free: armed command accepted: %r", cmd)
                state.heard_queue.put(cmd)
                state.mute(_POST_COMMAND_COOLDOWN)
            else:
                logger.debug("Hands-free: armed but got empty transcript, re-arming")
                with state._lock:
                    state._armed_until = time.monotonic() + _ARM_WINDOW_SECONDS
            return

        # ── State 2: Not armed — check for wake word ──
        result = _extract_command_after_wake_word(transcript)

        if result is None:
            # No wake word detected — completely ignore this phrase
            logger.debug("Hands-free: no wake word in: %r", transcript)
            return

        if result:
            # Wake word + command in one breath: "Hi Sahulat what is SMIT"
            logger.info("Hands-free: one-shot command: %r", result)
            state.heard_queue.put(result)
            state.mute(_POST_COMMAND_COOLDOWN)
        else:
            # Just the wake word alone: arm and wait for next phrase
            logger.info("Hands-free: wake word detected, arming for %ds", int(_ARM_WINDOW_SECONDS))
            with state._lock:
                state._armed_until = time.monotonic() + _ARM_WINDOW_SECONDS
            state.status_queue.put("__heard_wake_word__")

    return _callback


def start(state: HandsFreeState) -> None:
    """Start listening in the background. Safe to call if already active."""
    if state.active:
        return
    try:
        mic = sr.Microphone()
    except Exception as exc:
        state.status_queue.put(f"__error__:No microphone available on this device ({exc}).")
        return

    try:
        with mic as source:
            state._recognizer.adjust_for_ambient_noise(source, duration=0.5)

        # Energy threshold: require clear voice, not whispers
        state._recognizer.energy_threshold = max(300, state._recognizer.energy_threshold)
        state._recognizer.dynamic_energy_threshold = True
        state._recognizer.dynamic_energy_adjustment_damping = 0.15
        state._recognizer.dynamic_energy_ratio = 1.5

        # Faster end-of-speech detection
        state._recognizer.pause_threshold = 0.6      # 0.6s silence = user stopped talking
        state._recognizer.non_speaking_duration = 0.3 # min silence before phrase starts

        # No phrase time limit — let user speak as long as they want
        stop_fn = state._recognizer.listen_in_background(
            mic, _make_callback(state), phrase_time_limit=None
        )
    except Exception as exc:
        state.status_queue.put(f"__error__:Could not start the microphone listener ({exc}).")
        return

    state._stop_fn = stop_fn
    state.active = True
    logger.info("Hands-free started (wake word: %r).", settings.wake_word)


def stop(state: HandsFreeState) -> None:
    """Stop the background listener. Safe to call if already stopped."""
    if state._stop_fn is not None:
        try:
            state._stop_fn(wait_for_stop=False)
        except Exception:
            logger.exception("Error stopping hands-free listener")
        state._stop_fn = None
    state.active = False
    with state._lock:
        state._armed_until = 0.0
        state._is_speaking = False
        state._muted_until = 0.0
    logger.info("Hands-free listening stopped.")
