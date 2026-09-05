"""Lightweight state objects passed through a single agent turn."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionTrace:
    """A safe, user-visible record of what the agent did (no chain-of-thought)."""
    entries: list[str] = field(default_factory=list)

    def log(self, label: str) -> None:
        self.entries.append(label)

    def render(self) -> str:
        if not self.entries:
            return ""
        return "Agent actions:\n" + "\n".join(f"✓ {e}" for e in self.entries)


@dataclass
class TurnResult:
    answer_text: str
    action_trace: ActionTrace
    sources: list[dict] = field(default_factory=list)
    tool_calls_used: int = 0
    language: str = "en"
