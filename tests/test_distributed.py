"""Tests for distributed locks."""
from __future__ import annotations

import pytest

from borg.distributed import DistributedLock
from borg.db import Database


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'distributed.db'}"
    database = Database(db_url)
    yield database


def test_lock_acquire_and_release(fresh_db) -> None:
    """A lock can be acquired and released."""
    lock = DistributedLock("test_lock", database=fresh_db, ttl_seconds=5, instance_id="a")
    assert lock.acquire(timeout_seconds=1.0) is True
    lock.release()


def test_lock_excludes_other_holders(fresh_db) -> None:
    """Two different instance IDs cannot hold the same lock simultaneously."""
    lock_a = DistributedLock("exclusive", database=fresh_db, ttl_seconds=5, instance_id="a")
    lock_b = DistributedLock("exclusive", database=fresh_db, ttl_seconds=5, instance_id="b")

    assert lock_a.acquire(timeout_seconds=1.0) is True
    assert lock_b.acquire(timeout_seconds=0.5) is False

    lock_a.release()
    assert lock_b.acquire(timeout_seconds=1.0) is True
    lock_b.release()


def test_lock_context_manager(fresh_db) -> None:
    """The lock works as a context manager."""
    with DistributedLock("ctx", database=fresh_db, ttl_seconds=5, instance_id="a"):
        other = DistributedLock("ctx", database=fresh_db, ttl_seconds=5, instance_id="b")
        assert other.acquire(timeout_seconds=0.2) is False

    # After exit, another holder can acquire.
    assert other.acquire(timeout_seconds=1.0) is True
    other.release()
