"""
Tool: check_eligibility — evaluates eligibility against rules loaded from
config/eligibility_rules.yaml. Rules are DATA, never hardcoded if/else logic
for program-specific facts, so staff can update programs without a code change.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import yaml

from config.settings import settings

TOOL_SCHEMA = {
    "name": "check_eligibility",
    "description": (
        "Check whether a user is eligible for a Saylani program (course, welfare "
        "support, etc.) based on age, city, income, and the program name. Use this "
        "when the user asks 'am I eligible' or wants to know if they qualify."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "program": {"type": "string", "description": "Program key or name, e.g. 'saylani_it_courses'."},
            "age": {"type": "integer", "description": "User's age in years."},
            "city": {"type": "string", "description": "User's city."},
            "income": {"type": "integer", "description": "Monthly household income (PKR), if relevant."},
        },
        "required": ["program"],
    },
}


@lru_cache(maxsize=1)
def _load_rules() -> dict:
    if not settings.eligibility_rules_path.exists():
        return {"programs": {}}
    with open(settings.eligibility_rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"programs": {}}


def _match_program_key(program: str, programs: dict) -> Optional[str]:
    program_lower = program.strip().lower().replace(" ", "_")
    if program_lower in programs:
        return program_lower
    for key, cfg in programs.items():
        if program.strip().lower() in cfg.get("display_name", "").lower():
            return key
    return None


def run(program: str, age: Optional[int] = None, city: Optional[str] = None,
        income: Optional[int] = None) -> dict:
    rules_data = _load_rules()
    programs = rules_data.get("programs", {})
    key = _match_program_key(program, programs)
    if key is None:
        return {
            "status": "insufficient_information",
            "message": f"Unknown program '{program}'. Known programs: {', '.join(programs.keys())}",
        }

    cfg = programs[key]
    rules = cfg.get("rules", {})
    required = cfg.get("required_fields", [])

    provided = {"age": age, "city": city, "income": income}
    missing = [f for f in required if provided.get(f) in (None, "")]
    if missing:
        return {
            "status": "insufficient_information",
            "program": cfg.get("display_name", key),
            "missing_fields": missing,
        }

    reasons = []
    if rules.get("min_age") is not None and age is not None and age < rules["min_age"]:
        reasons.append(f"minimum age is {rules['min_age']}")
    if rules.get("max_age") is not None and age is not None and age > rules["max_age"]:
        reasons.append(f"maximum age is {rules['max_age']}")

    allowed_cities = rules.get("allowed_cities") or []
    if allowed_cities and "any" not in [c.lower() for c in allowed_cities]:
        if city and city.strip().lower() not in [c.lower() for c in allowed_cities]:
            reasons.append(f"program is currently limited to: {', '.join(allowed_cities)}")

    if rules.get("max_income") is not None and income is not None and income > rules["max_income"]:
        reasons.append(f"household income must be at or below {rules['max_income']}")

    if reasons:
        return {
            "status": "not_eligible",
            "program": cfg.get("display_name", key),
            "reasons": reasons,
        }

    return {
        "status": "eligible",
        "program": cfg.get("display_name", key),
    }
