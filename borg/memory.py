"""Semantic memory / learnings with pgvector similarity."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from borg.cpu_worker import EmbeddingWorker
from borg.db import Database, db
from borg.llm import llm
from borg.schemas import Episode, StrategyEfficacy


def _json_safe(obj: Any) -> Any:
    """Recursively make a value JSON-serializable."""
    from datetime import date, datetime

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _calculate_sharpe(returns: list[float]) -> float:
    """Simple annualized Sharpe approximation for a list of returns."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) or 1e-9
    return mean / std


def _calculate_max_drawdown(returns: list[float]) -> float:
    """Maximum peak-to-trough drawdown from a return series."""
    if not returns:
        return 0.0
    peak = 0.0
    drawdown = 0.0
    cumulative = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > drawdown:
            drawdown = dd
    return drawdown


class Memory:
    """Store and retrieve learnings and episodic memory with embedding search."""

    def __init__(self, database: Database = db, worker: Optional[EmbeddingWorker] = None) -> None:
        self.db = database
        self._worker = worker

    async def remember(
        self,
        summary: str,
        detail: Optional[str] = None,
        task_id: Optional[int] = None,
        tags: Optional[list[str]] = None,
    ) -> int:
        """Store a learning. Embedding is generated asynchronously."""
        text = f"{summary} {detail or ''}".strip()
        embedding = await llm.embed(text)
        return self.db.insert_learning(
            {
                "task_id": task_id,
                "summary": summary,
                "detail": detail,
                "tags": tags or [],
                "embedding": embedding,
            }
        )

    async def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the most relevant learnings for a query."""
        query_vec = await llm.embed(query)
        return self.db.search_learnings(query_vec, top_k=top_k)

    async def observe(
        self,
        content: str,
        kind: str = "observation",
        source: Optional[str] = None,
    ) -> int:
        """Store an episodic memory entry."""
        embedding = await llm.embed(content)
        return self.db.insert_memory(
            {"kind": kind, "content": content, "embedding": embedding, "source": source}
        )

    async def store_episode(self, episode: Episode) -> int:
        """Store an episode and embed it for similarity search."""
        text = f"{episode.regime} {episode.trigger} {episode.outcome}"
        embedding = await llm.embed(text)
        episode.embedding = embedding
        data = episode.model_dump()
        # Ensure JSON serializability for SQLite storage.
        data = _json_safe(data)
        return self.db.insert_episode(data)

    async def find_similar_episodes(
        self,
        current_regime: str,
        trigger: str,
        limit: int = 10,
    ) -> list[Episode]:
        """Find episodes similar to the current state via vector similarity."""
        query = f"{current_regime} {trigger}"
        query_vec = await llm.embed(query)
        rows = self.db.search_episodes(query_vec, top_k=limit)
        return [Episode(**r) for r in rows]

    async def get_strategy_efficacy(
        self,
        strategy_name: str,
        regime: str,
        window_days: int = 30,
    ) -> StrategyEfficacy:
        """Win rate, Sharpe, and drawdown for a strategy in a regime."""
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        episodes = self.db.query_episodes(
            actor=strategy_name,
            regime=regime,
            since=since,
            limit=10_000,
        )

        if not episodes:
            return StrategyEfficacy(
                strategy=strategy_name,
                regime=regime,
                win_rate=0.0,
                sample_size=0,
                window_days=window_days,
            )

        wins = sum(1 for e in episodes if e.get("outcome", {}).get("win"))
        win_rate = wins / len(episodes)
        returns: list[float] = []
        for e in episodes:
            outcome = e.get("outcome") or {}
            hr = outcome.get("horizon_return")
            if hr is not None:
                returns.append(float(hr))

        avg_pnl = sum(returns) / len(returns) if returns else 0.0
        sharpe = _calculate_sharpe(returns)
        max_dd = _calculate_max_drawdown(returns)

        return StrategyEfficacy(
            strategy=strategy_name,
            regime=regime,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            sharpe=sharpe,
            max_drawdown=max_dd,
            sample_size=len(episodes),
            window_days=window_days,
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, optionally using the process pool."""
        if self._worker is not None:
            return self._worker.embed_texts(texts)
        return [_fake_embedding_sync(text) for text in texts]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.recent_learnings(limit=limit)


def _fake_embedding_sync(text: str, dims: int = 768) -> list[float]:
    """Synchronous fallback embedding for batch use."""
    import math
    import random

    rng = random.Random(text)
    vec = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


memory = Memory()
