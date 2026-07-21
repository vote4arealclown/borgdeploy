"""Stress tests for parallelism and throughput."""
from __future__ import annotations

import asyncio
import time

import pytest

from borg.llm import llm


@pytest.mark.asyncio
async def test_parallel_forecasts_speedup(monkeypatch) -> None:
    """Run several forecasts in parallel and verify asyncio parallelism is effective.

    Uses a mock async forecast that sleeps for a short I/O-like delay. This
    isolates event-loop concurrency from the CPU-bound fallback rule engine.
    """
    delay = 0.05
    call_count = 0

    async def mock_analyze(symbol: str, summary: str) -> dict:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(delay)
        return {"direction": "up", "confidence": 70.0, "model_used": "mock"}

    monkeypatch.setattr(llm, "analyze_market", mock_analyze)

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"]

    # Sequential baseline.
    start = time.perf_counter()
    sequential = [await llm.analyze_market(s, "summary") for s in symbols]
    sequential_time = time.perf_counter() - start

    # Parallel via asyncio.gather.
    start = time.perf_counter()
    parallel = await asyncio.gather(*[llm.analyze_market(s, "summary") for s in symbols])
    parallel_time = time.perf_counter() - start

    speedup = sequential_time / parallel_time if parallel_time else 1.0
    print(f"Sequential: {sequential_time:.3f}s")
    print(f"Parallel:   {parallel_time:.3f}s")
    print(f"Speedup:    {speedup:.2f}x")

    assert len(sequential) == len(symbols)
    assert len(parallel) == len(symbols)
    assert call_count == len(symbols) * 2
    # Expect near-linear speedup for I/O-bound mocked forecasts.
    assert speedup > 1.5, f"Async parallelism not effective: speedup only {speedup:.2f}x"
