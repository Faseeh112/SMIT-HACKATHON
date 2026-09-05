"""Tool: schedule_callback — validates and stores a callback request."""
from __future__ import annotations

import re
from datetime import datetime

from database import db

TOOL_SCHEMA = {
    "name": "schedule_callback",
    "description": (
        "Schedule a staff callback for the user at a requested date/time. Use this "
        "when the user asks to be called back or wants staff to reach out."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
            "time": {"type": "string", "description": "Time in HH:MM 24-hour format."},
            "contact": {"type": "string", "description": "Phone number or email to reach the user."},
            "reason": {"type": "string", "description": "Brief reason for the callback."},
        },
        "required": ["date", "time", "contact"],
    },
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_valid_contact(contact: str) -> bool:
    contact = contact.strip()
    if _EMAIL_RE.match(contact):
        return True
    digits_only = re.sub(r"[\s\-()]", "", contact)
    return bool(re.match(r"^\+?\d{7,15}$", digits_only))


def run(user_id: str, date: str, time: str, contact: str, reason: str = "") -> dict:
    errors = []
    if not _DATE_RE.match(date or ""):
        errors.append("date must be in YYYY-MM-DD format")
    else:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            errors.append("date is not a real calendar date")
    if not _TIME_RE.match(time or ""):
        errors.append("time must be in HH:MM 24-hour format")
    if not contact or not _is_valid_contact(contact):
        errors.append("contact must be a valid phone number or email")

    if errors:
        return {"status": "error", "message": "; ".join(errors)}

    try:
        callback_id = db.create_callback_record(
            user_id=user_id, date=date, time=time, contact=contact.strip(), reason=reason
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "error", "message": f"Could not save callback: {exc}"}

    return {"status": "success", "callback_id": callback_id}
