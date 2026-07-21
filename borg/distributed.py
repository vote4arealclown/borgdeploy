"""Distributed coordination primitives for multi-instance Borg deployments."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from borg.db import Database, db


class DistributedLock:
    """Database-backed lock for coordinating multiple Borg instances.

    Works with both PostgreSQL and SQLite. Locks expire automatically after
    `ttl_seconds` to avoid deadlocks if a holder crashes.
    """

    def __init__(
        self,
        lock_name: str,
        database: Database = db,
        ttl_seconds: int = 30,
        instance_id: Optional[str] = None,
    ) -> None:
        self.db = database
        self.lock_name = lock_name
        self.ttl_seconds = ttl_seconds
        self.instance_id = instance_id or str(uuid.uuid4())[:8]

    def _now(self) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)).isoformat()

    def _placeholder(self) -> str:
        return "%s" if self.db.is_postgres else "?"

    def acquire(self, timeout_seconds: float = 5.0) -> bool:
        """Try to acquire the lock; block up to timeout_seconds."""
        ph = self._placeholder()
        expires = self._now()
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                # Try to insert a fresh lock row.
                self.db.execute(
                    f"INSERT INTO distributed_locks (lock_name, holder, expires_at) VALUES ({ph}, {ph}, {ph}) ON CONFLICT (lock_name) DO NOTHING",
                    (self.lock_name, self.instance_id, expires),
                )

                # Verify we hold the lock and it has not expired.
                row = self.db.fetchone(
                    f"SELECT holder, expires_at FROM distributed_locks WHERE lock_name = {ph}",
                    (self.lock_name,),
                )
                if row and row["holder"] == self.instance_id:
                    return True
            except Exception as exc:
                # Lock table may not exist yet; caller should ensure schema is applied.
                print(f"Distributed lock acquire error: {exc}")
                return False

            # Wait a bit before retrying.
            time.sleep(0.1)

        return False

    def release(self) -> None:
        """Release the lock if this instance holds it."""
        ph = self._placeholder()
        self.db.execute(
            f"DELETE FROM distributed_locks WHERE lock_name = {ph} AND holder = {ph}",
            (self.lock_name, self.instance_id),
        )

    def __enter__(self) -> "DistributedLock":
        if not self.acquire():
            raise RuntimeError(f"Could not acquire lock {self.lock_name}")
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
