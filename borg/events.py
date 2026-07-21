"""Central event log for the Borg colony."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from borg.db import Database, db


class EventLog:
    """Structured event stream used by the dashboard and visual simulation."""

    def __init__(self, database: Database = db) -> None:
        self.db = database

    def emit(
        self,
        message: str,
        category: str = "general",
        level: str = "INFO",
        phase: Optional[str] = None,
        symbol: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        return self.db.insert_event(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "category": category,
                "phase": phase,
                "message": message,
                "symbol": symbol,
                "metadata": metadata,
            }
        )

    def recent(
        self,
        limit: int = 100,
        category: Optional[str] = None,
        after_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        return self.db.recent_events(limit=limit, category=category, after_id=after_id)


event_log = EventLog()
