"""
scripts/run_evaluation.py

Runs the 15-question evaluation set (5 answerable, 5 unanswerable/refusal,
5 tool-required) against the LIVE agent (real Grok calls — requires
GEMINI_API_KEY and network access) and reports accuracy + timing.

This script makes real API calls and therefore is NOT run automatically by
`pytest`. Run it manually once your .env is configured and the knowledge
base has been ingested:

    python scripts/run_evaluation.py

It does not fabricate results — if GEMINI_API_KEY is missing or a call
fails, that question is recorded as a failure, not skipped silently.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import run_turn  # noqa: E402
from config.settings import settings  # noqa: E402
from database import db  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "tests" / "evaluation_questions.json"


def _run_one(user_id: str, conv_id: str, question: str) -> dict:
    start = time.time()
    try:
        result = run_turn(user_id=user_id, conversation_id=conv_id, user_message=question)
        elapsed = time.time() - start
        return {
            "question": question,
            "answer": result.answer_text,
            "tools_used": result.action_trace.entries,
            "response_time_seconds": round(elapsed, 2),
            "error": None,
        }
    except Exception as exc:
        elapsed = time.time() - start
        return {
            "question": question,
            "answer": None,
            "tools_used": [],
            "response_time_seconds": round(elapsed, 2),
            "error": str(exc),
        }


def main() -> None:
    problems = settings.validate()
    if problems:
        print("Cannot run evaluation:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    db.init_db()
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    user_id = f"eval-{uuid.uuid4().hex[:8]}"
    db.ensure_user(user_id)
    conv_id = db.start_conversation(user_id)

    all_results = {"answerable": [], "unanswerable": [], "tool_required": []}
    failures = 0
    total_time = 0.0
    total_questions = 0

    for category, items in data.items():
        for item in items:
            r = _run_one(user_id, conv_id, item["question"])
            all_results[category].append({**item, **r})
            total_questions += 1
            total_time += r["response_time_seconds"]
            if r["error"]:
                failures += 1

    print("\n=== SahulatAI Evaluation Results ===")
    print(f"Total questions: {total_questions}")
    print(f"Failures (exceptions/errors): {failures}")
    print(f"Average response time: {round(total_time / max(total_questions, 1), 2)}s\n")

    for category, items in all_results.items():
        print(f"--- {category} ---")
        for r in items:
            status = "ERROR" if r["error"] else "OK"
            print(f"  [{status}] ({r['response_time_seconds']}s) {r['question']}")
            if r["error"]:
                print(f"      error: {r['error']}")
            else:
                print(f"      answer: {r['answer'][:150]}")
                if r["tools_used"]:
                    print(f"      tools:  {r['tools_used']}")
        print()

    out_path = Path(__file__).resolve().parent.parent / "logs" / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Full results written to {out_path}")

    print(
        "\nNOTE: 'accuracy' for answerable/unanswerable/tool-required categories "
        "requires a human (or a second LLM-as-judge pass, not included here to "
        "avoid extra API cost) to grade each answer against its expected outcome. "
        "This script reports raw answers, tool traces, and timing/failure counts "
        "so that grading can be done honestly rather than self-reported."
    )


if __name__ == "__main__":
    main()
