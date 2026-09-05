# SahulatAI — Community Helpdesk Agent

A local, cost-conscious AI helpdesk for Saylani Welfare International.
Grounded RAG over a local knowledge base, Gemini-driven tool use, SQLite
storage, and a Streamlit UI with text, click-to-talk, and hands-free
wake-word voice modes.

**No phone number. No WhatsApp. No Twilio. No paid APIs required to run it.**

---

## 1. Architecture

```
                    USER
                      |
          +-----------+------------+
          |           |            |
        TEXT      CLICK MIC     WAKE WORD
          |           |            |
          |          STT          STT
          |           |            |
          +-----------+------------+
                      |
                      v
                SAHULATAI AGENT  (agent/agent.py)
                      |
          +-----------+-----------+
          |           |           |
        GEMINI       RAG        TOOLS
          |           |           |
          |        ChromaDB     SQLite
          |           |           |
          +-----------+-----------+
                      |
                      v
                  FINAL ANSWER
                      |
             +--------+--------+
             |                 |
            TEXT             pyttsx3 (voice)
```

Knowledge flow is always: **USER → local ChromaDB → retrieved trusted
sources → Gemini → grounded answer.** Gemini never answers
Saylani-specific questions from its own general knowledge — it must call
`search_documents` first, and it must not fill in a gap the knowledge
base doesn't cover.

### Project layout

```
SahulatAI/
├── app.py                    # Streamlit frontend (3 input modes + admin tab)
├── agent/                    # Gemini function-calling loop, memory, tool registry
├── rag/                      # loader, chunker, embeddings, ChromaDB store, retriever
├── tools/                    # search_documents, create_complaint, check_eligibility, schedule_callback
├── voice/                    # STT (free), TTS (pyttsx3), wake-word detection
├── database/                 # SQLite schema + access layer
├── knowledge_base/           # raw crawled pages, challenge docs, PDFs
├── config/                   # settings.py, sources.yaml, eligibility_rules.yaml
├── scripts/                  # crawl_saylani.py, ingest_documents.py, run_evaluation.py
├── tests/                    # pytest suite + evaluation_questions.json
├── prompts/system_prompt.txt # the agent's grounding + security rules
└── data/                     # chroma_db/ (vector index), sahulatai.db (SQLite)
```

---

## 2. Installation (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The same commands work on macOS/Linux with `python3.12 -m venv .venv` and
`source .venv/bin/activate`.

> This project was built and its non-network logic tested in a Linux
> sandbox against Python 3.12.3, installing the exact `requirements.txt`
> above with zero dependency conflicts. See **"What was actually
> validated" near the end of this README** for exactly what that did and
> didn't cover.

## 3. Environment variables

```powershell
copy .env.example .env
```

Edit `.env` and set:

```
GEMINI_API_KEY=your_key_here
```

Get a free key from **Google AI Studio**: https://aistudio.google.com/app/apikey

Everything else in `.env.example` has a sensible default (chunk size,
top-k, wake word, etc.) — you only need to set `GEMINI_API_KEY` to get
started.

**Never commit your real `.env`.** Only `.env.example` (placeholders) is
tracked.

## 4. Gemini setup

- Model used for chat + tool calling: `gemini-1.5-flash` (fast, free-tier
  friendly). Change via `GEMINI_CHAT_MODEL` in `.env` if needed.
- Model used for embeddings: `text-embedding-004`. Change via
  `GEMINI_EMBED_MODEL`.
- If Gemini embedding calls fail (no key, quota exhausted, offline dev),
  `rag/embeddings.py` automatically falls back to a local deterministic
  hashing-based embedding so the rest of the pipeline keeps working —
  this fallback is for **development/testing only**, not for a real demo
  answer quality bar.

## 5. Speech-to-Text (STT) setup

Default: `SpeechRecognition`'s free Google Web Speech endpoint — **no API
key, no billing**. Subject to informal rate limits; fine for a demo.

For a fully offline alternative, install `pocketsphinx` yourself
(`pip install pocketsphinx`) and pass `engine="sphinx"` to
`voice/speech_to_text.py::transcribe_audio_bytes`. Not included in
`requirements.txt` by default to keep the install lean.

## 6. Text-to-Speech (TTS) setup

Default and only TTS engine: **pyttsx3** (offline, free, uses the local
OS speech engine — SAPI5 on Windows). No cloud calls, no API key.

If no Urdu-capable voice is installed on the machine, the UI shows a
warning and keeps the text answer available — it never crashes.

## 7. Knowledge-base setup

Two ways to add knowledge:

**A. Crawl the official Saylani website**

```powershell
python scripts/crawl_saylani.py
```

Reads trusted URLs from `config/sources.yaml`, stays within
`allowed_domains`, respects `robots.txt`, limits depth/pages, and saves
cleaned page text + metadata as JSON under `knowledge_base/raw/`.

**B. Add challenge documents / PDFs**

Drop `.txt`/`.md` files into `knowledge_base/challenge_docs/` and PDFs
into `knowledge_base/pdfs/`. Two sample challenge documents are already
included so the project is demoable without a live crawl.

**Then ingest everything into ChromaDB:**

```powershell
python scripts/ingest_documents.py
```

This is incremental — re-running it after a small site change only
re-embeds chunks whose content actually changed (tracked via SHA-256
content hashes), so it won't burn embedding quota unnecessarily.

## 8. ChromaDB

Persistent, local, embedded vector database stored at `data/chroma_db/`.
No server to run, no account, no cost. Every chunk's metadata (source
URL, title, type, document id, chunk id, section, crawl date, content
hash, and PDF filename/page number where relevant) is stored alongside
the vector for citation.

## 9. Running the app

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`. Three ways to talk to SahulatAI:

- **Text** — normal chat box.
- **Click-to-talk** — press START, speak, press STOP; you'll see "You
  said: ..." before the answer.
- **Hands-free** — toggle it on, say the wake word (default
  `"Hi Sahulat"`, configurable via `WAKE_WORD` in `.env`) followed by
  your question.

An **Admin** tab shows knowledge-base stats (chunk/document/source
counts, last ingestion run) and a confirm-gated "clear & rebuild index"
action.

## 10. Tools

| Tool | Purpose | Storage |
|---|---|---|
| `search_documents` | RAG lookup against ChromaDB | — |
| `create_complaint` | File a complaint | SQLite `complaints` |
| `check_eligibility` | Check program eligibility against `config/eligibility_rules.yaml` | — |
| `schedule_callback` | Book a staff callback (validates date/time/contact) | SQLite `callbacks` |

Gemini decides which tool(s) to call via native function calling — there
is no keyword-based `if "complaint" in text` routing anywhere in the
code. A single request can chain multiple tools
(e.g. "check eligibility, then book me a callback").
`MAX_TOOL_CALLS` (default 5) caps tool calls per turn to prevent loops.

## 11. Memory

- **Conversation memory** — recent turns of the current conversation
  (SQLite `messages`), fed back to Gemini as context.
- **User memory** — durable, cross-session, and strictly allow-listed to
  non-sensitive fields (`preferred_language`, `city`,
  `last_program_interest`). Anything else is silently dropped rather
  than stored — see `agent/memory.py::_ALLOWED_MEMORY_KEYS`. Each user's
  memory is isolated by `user_id`.

## 12. Testing

```powershell
pytest tests/ -v
```

34 tests across `test_rag.py`, `test_tools.py`, `test_memory.py`,
`test_agent.py`, `test_language.py`, `test_refusal.py`, `test_voice.py`.
All of them run offline (no Gemini calls, no microphone, no network) by
using a fake LLM client for agent-loop tests and the local fallback
embedder for RAG tests, so the suite is fast and doesn't burn API quota.

## 13. Evaluation

```powershell
python scripts/run_evaluation.py
```

Runs the 15-question set in `tests/evaluation_questions.json` (5
answerable, 5 unanswerable, 5 tool-required) against the **live** agent
— this one does make real Gemini calls and needs `GEMINI_API_KEY` plus
network access. It records each answer, tool trace, response time, and
any errors to `logs/evaluation_results.json`, and prints a summary.

It deliberately does **not** self-report a fabricated accuracy score —
grading whether an answer was actually correct/refused/tool-appropriate
needs a human (or a separate LLM-as-judge pass) to read the transcripts,
so the script hands you the raw material for that rather than inventing
a number.

## 14. Troubleshooting

| Symptom | Likely cause |
|---|---|
| "GEMINI_API_KEY is not set" warning in the UI | Copy `.env.example` to `.env` and fill in your key |
| Retrieval always returns "not_found" | Run `python scripts/ingest_documents.py` — the index is probably empty |
| Click-to-talk does nothing | Browser microphone permission not granted, or `streamlit-mic-recorder` not installed |
| No sound from TTS | No voice installed on the OS, or pyttsx3 can't reach the OS speech engine — the app falls back to text-only and shows a warning |
| Crawler saves 0 pages | Check `config/sources.yaml` URLs are reachable and within `allowed_domains`; check `robots.txt` isn't blocking the path |
| `pip install` fails building `chroma-hnswlib` with "Microsoft Visual C++ 14.0 or greater is required" | No prebuilt wheel matched your Python/Windows combo, so pip tried to compile from source. Fix, in order: (1) `python -m pip install --upgrade pip setuptools wheel` then retry — an outdated pip often misses a valid wheel; (2) confirm you're on 64-bit Python 3.12 (`python -c "import platform; print(platform.python_version(), platform.architecture())"`), 32-bit Python has no matching wheel; (3) if using conda, `conda install -c conda-forge hnswlib` then `pip install -r requirements.txt`; (4) as a last resort, install "Build Tools for Visual Studio" with the "Desktop development with C++" workload from https://visualstudio.microsoft.com/visual-cpp-build-tools/ |

## 15. Limitations

- The local fallback embedding (used only when `GEMINI_API_KEY` is
  unavailable) is a simple hashing-based vector for pipeline testing —
  it is **not** semantically meaningful and should never be used for a
  real demo; always set a real Gemini key for actual retrieval quality.
- The free Google Web Speech STT endpoint is unofficial/rate-limited —
  fine for a hackathon demo, not for production-scale voice traffic.
- Hands-free wake-word mode in this Streamlit build listens for a fixed
  recording window per click rather than a truly always-on background
  listener (Streamlit's request/response model doesn't support a
  persistent background microphone loop cleanly); `voice/wake_word.py`
  includes both a Porcupine-based and a fuzzy-match fallback detector
  that a native desktop/background process could drive continuously.
- Roman Urdu STT quality depends on the underlying recognizer's handling
  of Urdu-accented English; there is no dedicated Roman Urdu acoustic
  model in the free tier.
## 16. What was actually validated (read this before trusting any "it works" claim)

Built in a sandboxed environment whose network access is restricted to
package registries (PyPI, npm, GitHub) — it cannot reach
`saylaniwelfare.com` or Google's Gemini API. Given that constraint, here
is exactly what was and wasn't verified before this ZIP was produced:

**Actually run and confirmed working:**
- Clean Python 3.12.3 venv, `pip install -r requirements.txt` — succeeded
  with zero conflicts.
- Every `.py` file parses (no syntax errors) and every core module
  imports cleanly (`config`, `database`, `rag`, `tools`, `agent`,
  `voice`).
- `pytest tests/` — 34/34 passing (chunking, tool logic, memory
  isolation/allow-listing, the agent tool-calling loop via a fake LLM,
  wake-word matching, refusal classification, prompt-safety content
  checks).
- `scripts/ingest_documents.py` — ran end-to-end against the two sample
  challenge documents included in `knowledge_base/challenge_docs/`,
  using the local fallback embedder: loaded documents, chunked them,
  and indexed real chunks into a persistent ChromaDB collection.
- `streamlit run app.py` — booted headlessly and served the UI over
  HTTP with no server-side errors.

**Written, structurally sound, but NOT executed against the live
service in this environment (needs your own `GEMINI_API_KEY` + network
access to saylaniwelfare.com):**
- `scripts/crawl_saylani.py` against the real Saylani website.
- Any real Gemini chat/function-calling/embedding call
  (`agent/agent.py`'s `GeminiClient`, real `rag/embeddings.py` calls).
- `scripts/run_evaluation.py`'s live 15-question run.
- Real microphone capture and real pyttsx3 audio output (no audio
  hardware in this sandbox) — the STT/TTS *code paths* are exercised by
  `tests/test_voice.py` with synthetic inputs, but not a real mic/speaker.

Before you demo this, run `python scripts/crawl_saylani.py`, then
`python scripts/ingest_documents.py` with a real `GEMINI_API_KEY` set,
then `python scripts/run_evaluation.py`, and read through the actual
answers it produces — don't take "it should work" as "it does work"
until you've seen that output yourself.
