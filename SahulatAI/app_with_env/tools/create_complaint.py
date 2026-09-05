"""Tool: create_complaint — logs a user complaint to SQLite."""
from __future__ import annotations

from database import db

TOOL_SCHEMA = {
    "name": "create_complaint",
    "description": (
        "File a complaint on behalf of the user. Use this when the user explicitly "
        "wants to report a problem, complain about a service, or escalate an issue."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Short category, e.g. 'admission', 'course', 'staff', 'facility', 'other'.",
            },
            "description": {
                "type": "string",
                "description": "The user's complaint in their own words.",
            },
        },
        "required": ["category", "description"],
    },
}


def run(user_id: str, category: str, description: str) -> dict:
    if not category.strip() or not description.strip():
        return {"status": "error", "message": "Category and description are both required."}
    try:
        complaint_id = db.create_complaint_record(user_id=user_id, category=category, description=description)
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "error", "message": f"Could not save complaint: {exc}"}
    return {"status": "success", "complaint_id": complaint_id}
