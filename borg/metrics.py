"""Prometheus instrumentation for Borg."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Gauge, Histogram


# Counters
brain_cycles_total = Counter("borg_brain_cycles_total", "Total brain cycles completed", ["status"])
forecasts_generated = Counter(
    "borg_forecasts_generated_total", "Forecasts generated", ["symbol", "direction"]
)
errors_total = Counter("borg_errors_total", "Total errors", ["component"])
ingested_rows_total = Counter("borg_ingested_rows_total", "Rows ingested from files", ["status"])

# Histograms (latency)
brain_cycle_duration_seconds = Histogram(
    "borg_brain_cycle_duration_seconds",
    "Duration of one brain cycle",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)
llm_inference_latency_seconds = Histogram(
    "borg_llm_inference_latency_seconds",
    "LLM inference latency",
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)
image_generation_latency_seconds = Histogram(
    "borg_image_generation_latency_seconds",
    "Pollinations image generation latency",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)
db_query_latency_seconds = Histogram(
    "borg_db_query_latency_seconds",
    "Database query latency",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)

# Gauges (current state)
memory_usage_mb = Gauge("borg_memory_usage_mb", "Memory usage in MB")
cpu_usage_percent = Gauge("borg_cpu_usage_percent", "CPU usage percentage")
active_threads = Gauge("borg_active_threads", "Number of active threads")
pending_tasks = Gauge("borg_pending_tasks", "Pending tasks in queue")


@contextmanager
def timed(histogram: Histogram) -> Generator[None, None, None]:
    """Context manager that observes elapsed seconds into a histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - start)


def record_resource_usage() -> None:
    """Update CPU/memory/thread gauges from current process state."""
    try:
        import psutil

        process = psutil.Process()
        memory_usage_mb.set(process.memory_info().rss / (1024 * 1024))
        cpu_usage_percent.set(process.cpu_percent(interval=None))
        active_threads.set(process.num_threads())
    except Exception:
        pass
