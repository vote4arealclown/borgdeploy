"""System resource monitor with soft safety limits."""
from __future__ import annotations


import asyncio
from typing import Optional

import psutil

from borg.config import settings
from borg.schemas import SystemStatus


class SystemMonitor:
    """Collect CPU / memory telemetry and decide whether to throttle work."""

    def __init__(self) -> None:
        # Prime psutil's CPU counter so later interval=None calls are non-blocking.
        psutil.cpu_percent(interval=0.1)
        self._current_sleep = float(settings.brain_interval_seconds)

    def status(self) -> SystemStatus:
        mem = psutil.virtual_memory()
        return SystemStatus(
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_used_mb=mem.used / (1024 * 1024),
            memory_total_mb=mem.total / (1024 * 1024),
            db_path=str(settings.sqlite_path) if not settings.is_postgres else settings.database_url,
            ollama_reachable=False,
            active_symbols=list(settings.symbol_list),
            last_cycle_at=None,
        )

    def should_throttle(self) -> bool:
        """Return True if the system is under sustained load."""
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            if cpu > settings.cpu_soft_limit_pct and mem.percent > settings.ram_soft_limit_pct:
                return True
            if mem.used / (1024 * 1024) > settings.memory_limit_mb:
                return True
        except Exception:
            return False
        return False

    def adaptive_interval(self, base: int) -> int:
        """Stretch interval when load is high."""
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            load = max(cpu / 100.0, mem.percent / 100.0)
            if load > 0.8:
                return min(settings.loop_max_interval_seconds, int(base * 1.5))
            if load < 0.3:
                return max(settings.loop_min_interval_seconds, int(base * 0.8))
        except Exception:
            pass
        return base

    def compute_adaptive_sleep(
        self,
        base: Optional[float] = None,
        cpu_threshold: Optional[float] = None,
        ram_threshold: Optional[float] = None,
    ) -> float:
        """Return the next sleep duration based on current resource usage.

        Exponential backoff when CPU or RAM is high; exponential recovery when low.
        """
        base = base if base is not None else settings.brain_interval_seconds
        cpu_threshold = cpu_threshold if cpu_threshold is not None else settings.cpu_soft_limit_pct
        ram_threshold = ram_threshold if ram_threshold is not None else settings.ram_soft_limit_pct

        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            if cpu > cpu_threshold or mem.percent > ram_threshold:
                self._current_sleep = min(self._current_sleep * 1.5, settings.loop_max_interval_seconds)
            else:
                self._current_sleep = max(self._current_sleep / 1.2, settings.loop_min_interval_seconds)
        except Exception:
            self._current_sleep = float(base)
        return self._current_sleep

    async def sleep_adaptive(self, base: Optional[float] = None) -> None:
        """Sleep for an adaptive duration based on resource usage."""
        sleep_time = self.compute_adaptive_sleep(base)
        await asyncio.sleep(sleep_time)


monitor = SystemMonitor()
