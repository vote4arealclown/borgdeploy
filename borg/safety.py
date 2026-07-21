"""Safety confirmation gates and audit helpers."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from borg.config import settings
from borg.db import Database, db
from borg.events import EventLog, event_log


class ActionKind(str, Enum):
    SELF_MODIFY = "self_modify"
    RESOURCE_HEAVY = "resource_heavy"
    CLONE = "clone"
    DELETE = "delete"
    FORECAST = "forecast"
    INGEST = "ingest"
    ASSIMILATE = "assimilate"
    IMAGE_GENERATION = "image_generation"


class SafetyGate:
    """Decides whether an action needs user confirmation before execution."""

    def __init__(
        self,
        database: Database = db,
        events: EventLog = event_log,
    ) -> None:
        self.db = database
        self.events = events
        self.required = set(settings.require_confirmation_for)
        # Runtime approvals granted via the API override the static config.
        self._approved: set[str] = set()

    @staticmethod
    def _key(action: ActionKind | str) -> str:
        if isinstance(action, ActionKind):
            return action.value
        return str(action)

    def needs_confirmation(self, action: ActionKind | str) -> bool:
        action_str = self._key(action)
        return action_str in self.required and action_str not in self._approved

    def check(self, action: ActionKind | str, detail: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Return a decision record; if confirmation is required, status is 'needs_confirm'."""
        action_str = self._key(action)
        requires = self.needs_confirmation(action)
        decision = {
            "action": action_str,
            "requires_confirmation": requires,
            "approved": not requires,
            "detail": detail or {},
        }
        self.events.emit(
            f"Safety check '{action_str}': {'needs_confirm' if requires else 'approved'}",
            category="safety",
            phase="idle",
            metadata=decision,
        )
        self.db.audit("safety", "check", decision)
        return decision

    def approve(self, action: ActionKind | str, actor: str = "user", detail: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        action_str = self._key(action)
        self._approved.add(action_str)
        record = {"action": action_str, "actor": actor, "approved": True, "detail": detail or {}}
        self.events.emit(f"Safety approval granted for {action_str} by {actor}", category="safety", phase="idle")
        self.db.audit(actor, f"approve:{action_str}", record)
        return record

    def reject(self, action: ActionKind | str, actor: str = "user", reason: str = "") -> dict[str, Any]:
        action_str = self._key(action)
        self._approved.discard(action_str)
        record = {"action": action_str, "actor": actor, "approved": False, "reason": reason}
        self.events.emit(f"Safety rejection for {action_str} by {actor}: {reason}", category="safety", phase="idle", level="WARN")
        self.db.audit(actor, f"reject:{action_str}", record)
        return record


safety = SafetyGate()
