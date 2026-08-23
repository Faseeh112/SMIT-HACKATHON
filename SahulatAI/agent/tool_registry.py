"""
Registers all tools with their Grok function-calling schema and a callable
executor. The agent never uses keyword-based routing — Grok decides which
tool(s), if any, to call based on these schemas.
"""
from __future__ import annotations

from typing import Callable

from tools import check_eligibility, create_complaint, schedule_callback, search_documents

# Tools that need the current user_id injected automatically (not exposed to
# the model as a parameter, since the model should never see/set user_id).
_USER_SCOPED_TOOLS = {"create_complaint", "schedule_callback"}

_REGISTRY: dict[str, dict] = {
    "search_documents": {"schema": search_documents.TOOL_SCHEMA, "fn": search_documents.run},
    "create_complaint": {"schema": create_complaint.TOOL_SCHEMA, "fn": create_complaint.run},
    "check_eligibility": {"schema": check_eligibility.TOOL_SCHEMA, "fn": check_eligibility.run},
    "schedule_callback": {"schema": schedule_callback.TOOL_SCHEMA, "fn": schedule_callback.run},
}


def all_schemas() -> list[dict]:
    return [entry["schema"] for entry in _REGISTRY.values()]


def get_executor(name: str) -> Callable | None:
    entry = _REGISTRY.get(name)
    return entry["fn"] if entry else None


def is_user_scoped(name: str) -> bool:
    return name in _USER_SCOPED_TOOLS


def execute_tool(name: str, args: dict, user_id: str) -> dict:
    fn = get_executor(name)
    if fn is None:
        return {"status": "error", "message": f"Unknown tool '{name}'."}
    call_args = dict(args)
    if is_user_scoped(name):
        call_args["user_id"] = user_id
    try:
        return fn(**call_args)
    except TypeError as exc:
        return {"status": "error", "message": f"Invalid arguments for {name}: {exc}"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "error", "message": f"Tool '{name}' failed: {exc}"}
