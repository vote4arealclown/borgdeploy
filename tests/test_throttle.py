"""Tests for adaptive resource throttling."""
from __future__ import annotations

import pytest

from borg.monitor import SystemMonitor
from borg.config import settings


def test_adaptive_sleep_increases_under_load(monkeypatch) -> None:
    """Adaptive sleep should grow when CPU/RAM exceed thresholds."""
    monitor = SystemMonitor()
    monitor._current_sleep = settings.loop_min_interval_seconds

    # Mock high CPU and RAM.
    import psutil

    original_cpu_percent = psutil.cpu_percent
    original_virtual_memory = psutil.virtual_memory

    class FakeMem:
        percent = 95.0
        used = 3 * 1024 * 1024 * 1024  # 3 GB
        total = 4 * 1024 * 1024 * 1024  # 4 GB

    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 95.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeMem())

    sleep1 = monitor.compute_adaptive_sleep()
    sleep2 = monitor.compute_adaptive_sleep()
    sleep3 = monitor.compute_adaptive_sleep()

    assert sleep2 > sleep1
    assert sleep3 > sleep2
    assert sleep3 <= settings.loop_max_interval_seconds

    monkeypatch.setattr(psutil, "cpu_percent", original_cpu_percent)
    monkeypatch.setattr(psutil, "virtual_memory", original_virtual_memory)


def test_adaptive_sleep_decreases_when_idle(monkeypatch) -> None:
    """Adaptive sleep should shrink when CPU/RAM are low."""
    monitor = SystemMonitor()
    monitor._current_sleep = settings.loop_max_interval_seconds

    import psutil

    class FakeMem:
        percent = 10.0
        used = 512 * 1024 * 1024
        total = 4 * 1024 * 1024 * 1024

    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 5.0)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeMem())

    sleep1 = monitor.compute_adaptive_sleep()
    sleep2 = monitor.compute_adaptive_sleep()

    assert sleep2 < sleep1
    assert sleep2 >= settings.loop_min_interval_seconds


@pytest.mark.asyncio
async def test_sleep_adaptive_non_negative() -> None:
    """Adaptive sleep should always be a non-negative number."""
    monitor = SystemMonitor()
    duration = monitor.compute_adaptive_sleep()
    assert duration >= 0
