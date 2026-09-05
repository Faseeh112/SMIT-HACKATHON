# SahulatAI — Final Project Report

## 1. Implemented features

- Streamlit UI with three input modes: text chat, click-to-talk
  microphone, hands-free wake-word toggle.
- Gemini-driven agent with native function calling (no keyword routing)
  over four tools: `search_documents`, `create_complaint`,
  `check_eligibility`, `schedule_callback`.
- Local RAG pipeline: crawler → loader → chunker → embeddings →
  persistent ChromaDB → relevance-thresholded, trust-ranked retriever.
- SQLite persistence: users, conversations, messages, user_memory,
  complaints, callbacks, eligibility_rules reference, ingestion_runs.
- Two-tier memory (conversation + allow-listed, isolated user memory).
- Multi-language support (English / Urdu / Roman Urdu) via
  language-mirroring instructions in the system prompt.
- Free/local voice stack: SpeechRecognition (free Google Web Speech
  endpoint, no key) for STT, pyttsx3 (fully offline) for TTS, and a
  dual-strategy wake-word detector (Porcupine if a free key is
  configured, otherwise a dependency-free fuzzy-match fallback).
- Admin panel: knowledge-base stats, last ingestion run, confirm-gated
  destructive "clear index" action.
- Security: documents/user text treated as data, not instructions;
  explicit prompt-injection guardrail language in the system prompt;
  per-user memory isolation; no keys ever logged or stored in the repo.
- Test suite (34 tests) and a 15-question evaluation harness with
  honest (non-fabricated) result reporting.

## 2–6. Level status

This build doesn't map to a specific numbered "Level 1–5" rubric that
was defined elsewhere in the challenge brief, so rather than invent
scoring against an unseen rubric, here's what's complete vs. partial
against the requirements actually given in this conversation:

- **Core grounded Q&A over a local knowledge base:** complete
  (retrieval, relevance thresholding, source citation, three-way
  answerable/partial/not-found handling, out-of-scope refusal).
- **Action tools (complaints, eligibility, callbacks):** complete, with
  real SQLite persistence and input validation.
- **Multi-step / chained tool calls in one turn:** implemented in the
  agent loop (sequential tool execution up to `MAX_TOOL_CALLS`); not
  exercised against a live Gemini model in this environment (see
  README §16).
- **Voice (STT/TTS/wake word):** code complete and unit-testable
  offline; not exercised against real audio hardware in this
  environment.
- **Crawler + incremental ingestion:** code complete, incremental via
  content hashing; not run against the live Saylani site in this
  environment (no network access to it from this sandbox).
- **Evaluation:** harness complete; live 15-question run not executed
  here (requires a real Gemini key + network).

## 7. Voice implementation status

- STT: implemented via `SpeechRecognition` (free Google Web Speech
  endpoint by default; optional offline `sphinx` engine). Error paths
  (empty audio, unclear audio, network/quota failure) are handled and
  unit-tested.
- TTS: implemented via `pyttsx3` (fully offline). Falls back to
  text-only with a UI warning if no voice/engine is available; never
  crashes the app (tested with a forced-failure mock is straightforward
  to add but not included in the current suite).
- Wake word: implemented with a Porcupine path (needs a free Picovoice
  key + a custom-trained `.ppn` keyword file for a literal "Hi Sahulat"
  detection) and a local fuzzy-match fallback that requires no extra
  key or dependency. Neither path streams continuous audio to Gemini.

## 8. Knowledge sources

- Official website: `saylaniwelfare.com` and subpages, via
  `config/sources.yaml` (not yet crawled in this environment — see
  README §16).
- Challenge documents: 2 sample markdown documents included in
  `knowledge_base/challenge_docs/` (`saylani_overview.md`,
  `eligibility_notes.md`) so the project is demoable without a live
  crawl. Replace/add real challenge documents here for the actual
  submission.
- Official PDFs: pipeline ready (`rag/loader.py::load_pdf`,
  `knowledge_base/pdfs/`), none included by default.

## 9–11. Document / chunk / ChromaDB configuration

From the one real ingestion run performed in this environment (sample
docs only): **2 documents → 3 chunks → 3 chunks indexed** into a
persistent ChromaDB collection at `data/chroma_db/`
(`sahulatai_knowledge_base`, cosine distance space).
`CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`, `TOP_K=5`,
`MIN_RELEVANCE_SCORE=0.35` (all configurable via `.env`). These numbers
will change once the real website crawl and real challenge documents
are ingested — re-run `scripts/ingest_documents.py` and check the Admin
tab for current counts.

## 12. Tools

`search_documents`, `create_complaint`, `check_eligibility`,
`schedule_callback` — see README §10 for the summary table. All four
have unit tests (`tests/test_tools.py`, plus `search_documents` covered
indirectly through `tests/test_refusal.py`'s classification tests).

## 13. Memory

Conversation memory (SQLite `messages`, isolated by `conversation_id`)
and allow-listed user memory (SQLite `user_memory`, isolated by
`user_id`, restricted to `preferred_language`, `city`,
`last_program_interest`). Verified in `tests/test_memory.py`, including
that disallowed keys (e.g. a hypothetical "password") are silently
dropped rather than stored.

## 14. STT implementation

See §7. Free-tier/no-key by default; offline `sphinx` fallback
available as an opt-in extra install.

## 15. TTS implementation

See §7. pyttsx3, fully offline, no API key, no billing.

## 16. Wake-word implementation

See §7. Default phrase: `"Hi Sahulat"`, configurable via `WAKE_WORD` in
`.env`.

## 17. Evaluation results

The evaluation harness (`scripts/run_evaluation.py`) and its 15-question
dataset (`tests/evaluation_questions.json`) are complete and were
**not** run against the live Gemini model in this environment (no
network access to Google's API here). Running it requires a real
`GEMINI_API_KEY`; it writes full transcripts, tool traces, response
times, and errors to `logs/evaluation_results.json` and does not
self-report a fabricated accuracy percentage — see README §13 for why.

What **was** run in this environment: the automated pytest suite (34
tests, all offline, all passing) and one real pass of
`scripts/ingest_documents.py` against the sample challenge documents,
confirming the chunk → embed → store pipeline works end-to-end.

## 18. Known limitations

See README §15 for the full list (fallback-embedding quality, free STT
rate limits, Streamlit's request/response model constraining "always
listening" wake-word behavior, Roman Urdu STT accuracy). The single
most important one: **this report and ZIP were produced without access
to the live Saylani website or the Gemini API**, so the knowledge base
currently contains only two illustrative sample documents, and no
answer quality has been verified against a real model. Treat this as a
complete, tested *scaffold* — the last mile (real crawl, real Gemini
key, real evaluation run, real mic/speaker test) is yours to run and
verify before a live demo.

## 19. Exact run command

```
streamlit run app.py
```

## 20. Exact ingestion commands

```
python scripts/crawl_saylani.py
python scripts/ingest_documents.py
```

## 21. Final ZIP path

`SahulatAI_Final.zip` (project root, alongside `app.py`). Excludes
`.venv/`, `.env`, the local `data/` runtime artifacts (chroma_db
contents, sqlite db), `__pycache__`, and `.pytest_cache` — includes
source, `requirements.txt`, `README.md`, this report, `.env.example`,
tests, scripts, prompts, config, and the sample knowledge base.
