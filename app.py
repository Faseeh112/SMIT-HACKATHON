"""
SahulatAI — Community Helpdesk Agent
Streamlit frontend: text chat, click-to-talk microphone, and hands-free
wake-word mode, backed by a Grok function-calling agent grounded on a
local ChromaDB knowledge base.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

import streamlit as st

from agent.agent import run_turn
from config.settings import settings
from database import db
from rag.ingest import UnsupportedFileType, ingest_uploaded_file, remove_document
from voice import hands_free
from voice.speech_to_text import TranscriptionError, transcribe_audio_bytes
from voice.text_to_speech import speak_async

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


@st.cache_resource
def _get_hands_free_state():
    from voice.hands_free import HandsFreeState
    return HandsFreeState()


hands_free_state = _get_hands_free_state()


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
    st.session_state.voice_state = "idle"  # idle | waiting | listening | processing | speaking | error
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
if "pending_response" not in st.session_state:
    st.session_state.pending_response = False
if "pending_speak" not in st.session_state:
    st.session_state.pending_speak = False


def enqueue_user_message(text: str, speak_reply: bool = False) -> None:
    """Show the user's message immediately and flag that a reply is owed."""
    st.session_state.chat_history.append({"role": "user", "content": text})
    db.add_message(st.session_state.conversation_id, "user", text)
    st.session_state.voice_state = "processing"
    st.session_state.pending_response = True
    st.session_state.pending_speak = speak_reply


def generate_pending_response() -> None:
    """Run the agent for the last queued user message and append the reply."""
    text = st.session_state.chat_history[-1]["content"]
    try:
        result = run_turn(
            user_id=st.session_state.user_id,
            conversation_id=st.session_state.conversation_id,
            user_message=text,
        )
    except RuntimeError as exc:
        st.session_state.voice_state = "error"
        st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ {exc}"})
        st.session_state.pending_response = False
        st.session_state.pending_speak = False
        return
    except Exception:
        logger.exception("Unhandled error while processing user message: %r", text)
        st.session_state.voice_state = "error"
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "Sorry, something went wrong on my end. Please try again."}
        )
        st.session_state.pending_response = False
        st.session_state.pending_speak = False
        return

    st.session_state.chat_history.append({"role": "assistant", "content": result.answer_text})
    db.add_message(st.session_state.conversation_id, "assistant", result.answer_text)
    st.session_state.last_sources = result.sources
    st.session_state.last_action_trace = result.action_trace.render()
    st.session_state.voice_state = "idle"

    if st.session_state.pending_speak:
        if hands_free_state.active:
            hands_free_state.set_speaking(True)
            speak_async(
                result.answer_text,
                on_done=lambda: hands_free_state.set_speaking(False, cooldown=2.0),
            )
        else:
            speak_async(result.answer_text)

    st.session_state.pending_response = False
    st.session_state.pending_speak = False


def _load_conversation(conversation_id: str) -> None:
    """Switch to an existing conversation and load its messages into the chat."""
    st.session_state.conversation_id = conversation_id
    rows = db.get_all_messages(conversation_id)
    st.session_state.chat_history = [{"role": r["role"], "content": r["content"]} for r in rows if r["role"] != "tool"]
    st.session_state.last_sources = []
    st.session_state.last_action_trace = ""
    st.session_state._last_processed_audio_id = None


def _start_new_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    st.session_state.last_action_trace = ""
    st.session_state._last_processed_audio_id = None
    st.session_state.conversation_id = db.start_conversation(st.session_state.user_id)


# ---------------------------------------------------------------------------
# Sidebar — new chat + conversation history (Claude/ChatGPT-style)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💬 SahulatAI")
    if st.button("🆕  New chat", use_container_width=True, key="new_chat_sidebar"):
        _start_new_chat()
        st.rerun()

    st.divider()
    st.caption("History")
    conversations = db.list_conversations(st.session_state.user_id)
    if not conversations:
        st.caption("No history")
    for conv in conversations:
        title = (conv["first_message"] or "New chat").strip()
        title = (title[:38] + "…") if len(title) > 38 else title
        is_active = conv["conversation_id"] == st.session_state.conversation_id
        if st.button(
            ("🟢 " if is_active else "") + (title or "New chat"),
            key=f"conv_{conv['conversation_id']}",
            use_container_width=True,
            disabled=is_active,
        ):
            _load_conversation(conv["conversation_id"])
            st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("SahulatAI")
st.caption("Community helpdesk (courses, admissions, welfare)")

problems = settings.validate()
if problems:
    for p in problems:
        st.warning(p)

status_labels = {
    "idle": "🟢 Idle",
    "waiting": f"🟢 Waiting for \"{settings.wake_word}\"...",
    "listening": "🎙️ Listening (Speak your question)",
    "processing": "⏳ Processing answer...",
    "speaking": "🔊 Speaking answer...",
    "error": "⚠️ Didn't catch that",
}

if hands_free_state.active:
    if hands_free_state.is_muted:
        st.session_state.voice_state = "speaking"
    elif hands_free_state.is_armed:
        st.session_state.voice_state = "listening"
    elif st.session_state.voice_state not in ("processing", "speaking"):
        st.session_state.voice_state = "waiting"

st.info(status_labels.get(st.session_state.voice_state, "🟢 Idle"))

tab_chat, tab_admin = st.tabs(["💬 Helpdesk", "🛠️ Admin"])

with tab_chat:
    # --- Chat history ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # If a user message was just queued, show thinking spinner and generate reply
    if st.session_state.pending_response:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                generate_pending_response()
        st.rerun()

    if st.session_state.last_sources:
        with st.expander("📚 Sources"):
            for s in st.session_state.last_sources:
                page = f" (p.{s['page_number']})" if s.get("page_number") else ""
                st.markdown(f"- **{s['source_title']}**{page} — `{s['source_url']}` ({s['score']})")

    if st.session_state.last_action_trace:
        with st.expander("🔍 Actions taken"):
            st.text(st.session_state.last_action_trace)

    st.divider()

    # --- Mode A: Text input ---
    text_input = st.chat_input("Type your message (English / Urdu / Roman Urdu)...")
    if text_input:
        enqueue_user_message(text_input)
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
                    enqueue_user_message(transcript, speak_reply=True)
                    st.rerun()
                except TranscriptionError as exc:
                    st.session_state.voice_state = "error"
                    st.error(str(exc))
    except ImportError:
        st.caption("streamlit-mic-recorder not installed (voice disabled)")

    # --- Mode C: Hands-free wake word ---
    st.subheader("🙌 Hands-free (wake word)")
    hf_on = st.toggle(
        f"Listen for \"{settings.wake_word}\"",
        value=st.session_state.hands_free,
        key="hands_free_toggle",
        help=f"Say '{settings.wake_word}' clearly to activate, then ask your question. SahulatAI answers and returns to waiting for '{settings.wake_word}'.",
    )
    if hf_on != st.session_state.hands_free:
        st.session_state.hands_free = hf_on
        if hf_on:
            hands_free.start(hands_free_state)
        else:
            hands_free.stop(hands_free_state)
        st.rerun()

    if st.session_state.hands_free:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=800, key="hands_free_poll")
        except ImportError:
            st.caption("streamlit-autorefresh not installed (auto-detect disabled)")

        # Surface listener errors (e.g. no microphone) and turn the toggle back off.
        while not hands_free_state.status_queue.empty():
            status_msg = hands_free_state.status_queue.get_nowait()
            if status_msg.startswith("__error__:"):
                st.session_state.hands_free = False
                hands_free.stop(hands_free_state)
                st.error(status_msg.split(":", 1)[1])
            elif status_msg == "__heard_wake_word__":
                st.toast(f"🎙️ Heard \"{settings.wake_word}\"! Speak your question now...")

        if hands_free_state.active:
            if hands_free_state.is_armed:
                st.caption(f"🎙️ **Listening...** (Wake word heard — ask your question now)")
            elif hands_free_state.is_muted:
                st.caption("🔊 **Speaking answer...**")
            else:
                st.caption(f"🟢 **Waiting for '{settings.wake_word}'** (Say '{settings.wake_word}' clearly to start)")

        # Drain one finished command per rerun and answer it like a typed message.
        if not hands_free_state.heard_queue.empty():
            heard_text = hands_free_state.heard_queue.get_nowait()
            enqueue_user_message(heard_text, speak_reply=True)
            st.rerun()

    st.divider()
    with st.expander("📎 Add document"):
        st.caption("PDF / TXT / MD (indexed instantly)")
        uploaded = st.file_uploader(
            "Upload a document", type=["pdf", "txt", "md"], key="chat_doc_uploader", label_visibility="collapsed"
        )
        if uploaded is not None:
            upload_id = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("_last_uploaded_id") != upload_id:
                try:
                    result = ingest_uploaded_file(
                        uploaded.getvalue(), uploaded.name, uploaded_by=st.session_state.user_id
                    )
                    st.session_state._last_uploaded_id = upload_id
                    st.success(f"Added **{result['filename']}** ({result['chunks']} chunks)")
                except UnsupportedFileType as exc:
                    st.error(str(exc))
                except Exception as exc:
                    logger.exception("Document upload failed: %r", uploaded.name)
                    st.error(f"Couldn't index that file: {exc}")


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
        st.write(f"**Last run:** {last_run['started_at']} ({last_run['status']})")
        st.write(f"Docs: {last_run['documents_ingested']} · Chunks: {last_run['chunks_created']}")
    else:
        st.write("No ingestion runs yet")

    st.divider()
    st.subheader("Actions")
    st.caption("Via scripts/crawl_saylani.py, scripts/ingest_documents.py")
    if st.button("🔄 Refresh stats"):
        st.rerun()

    st.divider()
    st.subheader("📄 Documents")
    st.caption("Uploaded files (download / remove)")

    with st.expander("➕ Add document"):
        admin_uploaded = st.file_uploader(
            "Upload a document", type=["pdf", "txt", "md"], key="admin_doc_uploader", label_visibility="collapsed"
        )
        if admin_uploaded is not None:
            admin_upload_id = f"{admin_uploaded.name}:{admin_uploaded.size}"
            if st.session_state.get("_last_admin_uploaded_id") != admin_upload_id:
                try:
                    result = ingest_uploaded_file(
                        admin_uploaded.getvalue(), admin_uploaded.name, uploaded_by="admin", source_type="admin_upload"
                    )
                    st.session_state._last_admin_uploaded_id = admin_upload_id
                    st.success(f"Added **{result['filename']}** ({result['chunks']} chunks)")
                    st.rerun()
                except UnsupportedFileType as exc:
                    st.error(str(exc))
                except Exception as exc:
                    logger.exception("Admin document upload failed: %r", admin_uploaded.name)
                    st.error(f"Couldn't index that file: {exc}")

    documents = db.list_documents()
    if not documents:
        st.caption("No documents yet")
    else:
        for doc in documents:
            doc_path = Path(doc["file_path"])
            cols = st.columns([4, 2, 2, 2, 2])
            cols[0].write(f"**{doc['filename']}**")
            cols[1].caption(doc["source_type"])
            cols[2].caption(f"{doc['chunks_created']} chunks")
            if doc_path.exists():
                cols[3].download_button(
                    "⬇️", data=doc_path.read_bytes(), file_name=doc["filename"],
                    key=f"dl_{doc['document_id']}", use_container_width=True,
                )
            else:
                cols[3].caption("missing")
            if cols[4].button("🗑️", key=f"rm_{doc['document_id']}", use_container_width=True):
                removed = remove_document(doc["document_id"])
                st.success(f"Removed {doc['filename']} ({removed} chunks)")
                st.rerun()
            if doc_path.exists() and doc_path.suffix.lower() in (".txt", ".md"):
                with st.expander(f"👁️ {doc['filename']}"):
                    st.text(doc_path.read_text(encoding="utf-8", errors="replace")[:4000])

    st.divider()
    st.subheader("⚠️ Destructive actions")
    confirm = st.checkbox("Confirm (deletes entire index)")
    if st.button("🗑️ Clear & rebuild index", disabled=not confirm):
        clear_collection()
        st.success("Index cleared (rebuild via ingest_documents.py)")