"""
Wake-word detection for hands-free mode.

Two strategies, chosen automatically:

1. Porcupine (pvporcupine) — local, lightweight, purpose-built wake-word
   engine. Requires a free Picovoice access key (PORCUPINE_ACCESS_KEY).
   Runs entirely on-device once initialized; does not call any cloud API
   while listening.

2. Fallback fuzzy-match detector — if no Porcupine key is configured, we
   transcribe short audio windows locally-ish via the free STT endpoint
   only when the audio contains speech energy above a simple threshold,
   and fuzzy-match the transcript against the configured wake word. This
   avoids sending continuous audio to any cloud service: nothing is sent
   until a burst of speech is detected locally.

Neither path streams continuous microphone audio to Grok — wake-word
detection never touches the LLM.
"""
from __future__ import annotations

import difflib
import logging

from config.settings import settings

logger = logging.getLogger("sahulatai.wake_word")


def is_porcupine_available() -> bool:
    return bool(settings.porcupine_access_key)


def fuzzy_matches_wake_word(transcript: str, wake_word: str | None = None,
                             threshold: float = 0.72) -> bool:
    """
    Local, dependency-free fallback: compare a short transcript against the
    configured wake word using sequence similarity. Used only after local
    energy-based voice-activity detection has already fired (see app.py),
    so this is not a continuous cloud call.
    """
    wake_word = (wake_word or settings.wake_word).strip().lower()
    transcript = (transcript or "").strip().lower()
    if not transcript:
        return False
    ratio = difflib.SequenceMatcher(None, transcript, wake_word).ratio()
    return ratio >= threshold or wake_word in transcript


class PorcupineWakeWordDetector:
    """Thin wrapper around pvporcupine for a locally-built custom keyword,
    or the closest built-in keyword as a stand-in for a hackathon demo."""

    def __init__(self, keyword_paths: list[str] | None = None):
        import pvporcupine

        if not settings.porcupine_access_key:
            raise RuntimeError("PORCUPINE_ACCESS_KEY is not set.")
        # NOTE: a truly custom "Hi Sahulat" keyword requires training a
        # .ppn file at https://console.picovoice.ai (free tier). Until that
        # file is generated and placed under config/wake_word/, this falls
        # back to a stock keyword ("porcupine") purely so the pipeline is
        # exercisable end-to-end; app.py prefers the fuzzy fallback unless
        # a real keyword_paths value is supplied.
        self._porcupine = pvporcupine.create(
            access_key=settings.porcupine_access_key,
            keyword_paths=keyword_paths,
            keywords=None if keyword_paths else ["porcupine"],
        )

    @property
    def frame_length(self) -> int:
        return self._porcupine.frame_length

    @property
    def sample_rate(self) -> int:
        return self._porcupine.sample_rate

    def process(self, pcm_frame) -> bool:
        """Returns True if the wake word was detected in this audio frame."""
        return self._porcupine.process(pcm_frame) >= 0

    def close(self) -> None:
        self._porcupine.delete()
