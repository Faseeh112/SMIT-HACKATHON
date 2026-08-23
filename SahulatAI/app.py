"""
SahulatAI — Community Helpdesk Agent
Streamlit frontend: text chat, click-to-talk microphone, and hands-free
wake-word mode, backed by a Grok function-calling agent grounded on a
local ChromaDB knowledge base.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging
import hashlib
import uuid

import streamlit as st

from agent.agent import run_turn
from config.settings import settings
from database import db
from voice.speech_to_text import TranscriptionError, transcribe_audio_bytes
from voice.text_to_speech import speak
from voice.wake_word import fuzzy_matches_wake_word

st.set_page_config(page_title="SahulatAI — Community Helpdesk", page_icon="💬", layout="centered")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.logs_dir / "app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("sahulatai.app")


# ---------------------------------------------------------------------------
# Cached, reusable resources (created once per process, not per message)
# ---------------------------------------------------------------------------
@st.cache_resource
def _init_database():
    db.init_db()
    return True


_init_database()


# ---------------------------------------------------------------------------
# Session state (per-user conversation — NOT cached globally)
# ---------------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    db.ensure_user(st.session_state.user_id)
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = db.start_conversation(st.session_state.user_id)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}
if "voice_state" not in st.session_state:
    st.session_state.voice_state = "idle"  # idle | listening | processing | speaking | error
if "hands_free" not in st.session_state:
    st.session_state.hands_free = False
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_action_trace" not in st.session_state:
    st.session_state.last_action_trace = ""
if "_last_processed_audio_id" not in st.session_state:
    st.session_state._last_processed_audio_id = None
if "_mic_key_counter" not in st.session_state:
    st.session_state._mic_key_counter = 0


def handle_user_message(text: str) -> None:
    """Send a message through the agent and update session state."""
    st.session_state.chat_history.append({"role": "user", "content": text})
    db.add_message(st.session_state.conversation_id, "user", text)

    st.session_state.voice_state = "processing"
    try:
        result = run_turn(
            user_id=st.session_state.user_id,
            conversation_id=st.session_state.conversation_id,
            user_message=text,
        )
    except RuntimeError as exc:
        st.session_state.voice_state = "error"
        st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ {exc}"})
        return
    except Exception:
        # Log the real traceback so failures are diagnosable from logs/ instead
        # of only showing the generic message below to the user.
        logger.exception("Unhandled error while processing user message: %r", text)
        st.session_state.voice_state = "error"
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "Sorry, something went wrong on my end. Please try again."}
        )
        return

    st.session_state.chat_history.append({"role": "assistant", "content": result.answer_text})
    db.add_message(st.session_state.conversation_id, "assistant", result.answer_text)
    st.session_state.last_sources = result.sources
    st.session_state.last_action_trace = result.action_trace.render()
    st.session_state.voice_state = "idle"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("SahulatAI")
st.caption("Community Helpdesk Assistant — courses, admissions, welfare & more")

problems = settings.validate()
if problems:
    for p in problems:
        st.warning(p)

status_labels = {
    "idle": f"🟢 Idle — say '{settings.wake_word}' to start (hands-free mode)",
    "listening": "🎙️ Listening...",
    "processing": "⏳ Processing your question...",
    "speaking": "🔊 Speaking...",
    "error": "⚠️ Sorry, I couldn't understand that. Please try again.",
}
st.info(status_labels.get(st.session_state.voice_state, ""))

tab_chat, tab_admin = st.tabs(["💬 Helpdesk", "🛠️ Admin"])

with tab_chat:
    # --- New Chat button to clear old messages ---
    if st.session_state.chat_history:
        if st.button("🗑️ New Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.last_sources = []
            st.session_state.last_action_trace = ""
            st.session_state._last_processed_audio_id = None
            st.session_state.conversation_id = db.start_conversation(st.session_state.user_id)
            st.rerun()

    # --- Chat history ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.last_sources:
        with st.expander("📚 Sources for the last answer"):
            for s in st.session_state.last_sources:
                page = f" (page {s['page_number']})" if s.get("page_number") else ""
                st.markdown(f"- **{s['source_title']}**{page} — `{s['source_url']}` (score {s['score']})")

    if st.session_state.last_action_trace:
        with st.expander("🔍 What the agent did"):
            st.text(st.session_state.last_action_trace)

    st.divider()

    # --- Mode A: Text input ---
    text_input = st.chat_input("Type your message (English / Urdu / Roman Urdu)...")
    if text_input:
        handle_user_message(text_input)
        st.rerun()

    # --- Mode B: Click-to-talk ---
    st.subheader("🎤 Voice input (click-to-talk)")
    try:
        from streamlit_mic_recorder import mic_recorder

        audio = mic_recorder(start_prompt="🎤 START", stop_prompt="⏹ STOP", format="wav",
                             key=f"click_to_talk_{st.session_state._mic_key_counter}")
        if audio and audio.get("bytes"):
            # Deduplicate: skip if we already processed this exact recording
            audio_id = hashlib.md5(audio["bytes"]).hexdigest()
            if audio_id != st.session_state._last_processed_audio_id:
                st.session_state._last_processed_audio_id = audio_id
                st.session_state._mic_key_counter += 1  # fresh widget on next rerun
                st.session_state.voice_state = "listening"
                try:
                    transcript = transcribe_audio_bytes(audio["bytes"], sample_rate=audio.get("sample_rate", 16000))
                    st.markdown(f"**You said:** _{transcript}_")
                    handle_user_message(transcript)
                    speak(st.session_state.chat_history[-1]["content"])
                    st.session_state.voice_state = "idle"
                    st.rerun()
                except TranscriptionError as exc:
                    st.session_state.voice_state = "error"
                    st.error(str(exc))
    except ImportError:
        st.caption(
            "Install `streamlit-mic-recorder` (already in requirements.txt) to enable "
            "click-to-talk. Falling back to text mode above."
        )



with tab_admin:
    st.subheader("Knowledge base status")
    from rag.vector_store import clear_collection, collection_stats

    stats = collection_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total chunks", stats["total_chunks"])
    col2.metric("Documents", stats["documents"])
    col3.metric("Official sources", stats["official_sources"])
    col4.metric("Challenge docs", stats["challenge_documents"])

    last_run = db.get_last_ingestion_run()
    if last_run:
        st.write(f"**Last ingestion run:** {last_run['started_at']} → {last_run['status']}")
        st.write(f"Documents ingested: {last_run['documents_ingested']}, chunks: {last_run['chunks_created']}")
    else:
        st.write("No ingestion runs recorded yet.")

    st.divider()
    st.subheader("Actions")
    st.caption(
        "Crawling and ingestion are run as scripts (scripts/crawl_saylani.py, "
        "scripts/ingest_documents.py) so long-running jobs don't block the Streamlit UI."
    )
    if st.button("🔄 Refresh stats"):
        st.rerun()

    st.divider()
    st.subheader("⚠️ Destructive actions")
    confirm = st.checkbox("I understand this will delete the entire vector index.")
    if st.button("🗑️ Clear & rebuild index", disabled=not confirm):
        clear_collection()
        st.success("Index cleared. Run `python scripts/ingest_documents.py` to rebuild it.")
