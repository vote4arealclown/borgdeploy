#!/usr/bin/env python3
"""Profile one brain cycle and report per-phase latency."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from borg.brain import Brain
from borg.config import settings
from borg.db import Database
from borg.llm import llm
from borg.memory import Memory
from borg.monitor import monitor


async def profile_cycle(brain: Brain, symbol: str) -> dict[str, float]:
    """Run one brain cycle for a single symbol and time each phase."""
    timings: dict[str, float] = {}

    # Observe
    start = time.perf_counter()
    market_data = await brain.observe(symbol)
    timings["observe"] = time.perf_counter() - start

    # Plan (includes LLM forecast)
    start = time.perf_counter()
    trades = await brain.plan(market_data)
    timings["plan"] = time.perf_counter() - start
    forecast = market_data.get("last_forecast")

    # Act
    start = time.perf_counter()
    await brain.act(symbol, trades, market_data)
    timings["act"] = time.perf_counter() - start

    # Reflect
    start = time.perf_counter()
    await brain.reflect(symbol, forecast, market_data)
    timings["reflect"] = time.perf_counter() - start

    return timings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a Borg brain cycle")
    parser.add_argument("--symbol", default=settings.symbol_list[0], help="Symbol to profile")
    parser.add_argument("--out", type=Path, default=Path("profile_latest.json"), help="Output JSON path")
    parser.add_argument("--ollama", action="store_true", help="Force Ollama instead of fallback")
    args = parser.parse_args()

    # Use a temporary database so profiling does not pollute production data.
    with TemporaryDirectory() as tmp:
        db_url = f"sqlite:///{tmp}/profile.db"
        db = Database(db_url)
        mem = Memory(database=db)
        brain = Brain(database=db, mem=mem)

        # Seed synthetic history.
        brain.data_feed.seed(db)

        # Disable throttling for profiling.
        monitor.should_throttle = lambda: False  # type: ignore[method-assign]

        if not args.ollama:
            # Use deterministic fallback for reproducible profiling.
            llm._ollama_available = False
        else:
            # Recheck Ollama so we test the real path.
            llm._ollama_available = None
            await llm._check_ollama()

        print("=" * 64)
        print("BORG BRAIN CYCLE PROFILE")
        print("=" * 64)
        print(f"Symbol: {args.symbol}")
        print(f"Database: {db_url}")
        print(f"Ollama available: {await llm._check_ollama()}")

        timings = await profile_cycle(brain, args.symbol)
        total = sum(timings.values())

        print("-" * 64)
        for phase, duration in timings.items():
            pct = duration / total * 100 if total else 0
            print(f"{phase:20s} {duration:8.3f}s  ({pct:5.1f}%)")
        print("-" * 64)
        print(f"{'TOTAL':20s} {total:8.3f}s")
        print("=" * 64)

        # Add context for interpretation.
        result = {
            "symbol": args.symbol,
            "ollama_available": await llm._check_ollama(),
            "timings": timings,
            "total_seconds": total,
        }
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.out}")

        # Interpretation hints.
        if timings.get("plan", 0) > 1.5:
            print("\nHint: plan/forecast phase is slow; consider process pool (Phase 2).")
        if timings.get("observe", 0) > 0.5:
            print("\nHint: observe phase is slow; consider caching or DB tuning.")


if __name__ == "__main__":
    asyncio.run(main())
