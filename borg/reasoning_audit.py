"""Audit trail for reasoning decisions and calibration tracking."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from borg.db import Database, db


class ReasoningAudit:
    """Log reasoning decisions, compare to outcomes, and measure calibration."""

    def __init__(self, database: Database = db) -> None:
        self.db = database

    async def record(
        self,
        reasoning_output: dict[str, Any],
        forecast_outcome: dict[str, Any],
        forecast_id: Optional[int] = None,
    ) -> int:
        """After a forecast resolves, link its reasoning to the outcome."""
        confidence = float(reasoning_output.get("confidence", 0.0))
        win = bool(forecast_outcome.get("win"))
        calibration_error = confidence / 100.0 - (1.0 if win else 0.0)
        return self.db.insert_reasoning_audit(
            {
                "forecast_id": forecast_id,
                "reasoning_decision": reasoning_output.get("decision"),
                "reasoning_confidence": confidence,
                "reasoning_why": reasoning_output.get("reasoning"),
                "outcome_win": win,
                "outcome_pnl": forecast_outcome.get("horizon_return"),
                "calibration_error": calibration_error,
                "metadata": {"risks": reasoning_output.get("risks", [])},
            }
        )

    async def get_calibration_report(self, window_days: int = 30) -> dict[str, Any]:
        """Measure whether confidence predicted outcome."""
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        audits = self.db.query_reasoning_audits(since=since, limit=10_000)

        buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for audit in audits:
            conf = float(audit.get("reasoning_confidence", 0.0))
            bucket = round(conf / 25.0) * 25.0  # 0, 25, 50, 75, 100
            buckets[bucket].append(audit)

        report: dict[str, Any] = {}
        total_error = 0.0
        total_count = 0
        for bucket in sorted(buckets.keys()):
            trades = buckets[bucket]
            wins = sum(1 for t in trades if t.get("outcome_win"))
            actual_rate = wins / len(trades) if trades else 0.0
            expected_rate = bucket / 100.0
            report[str(bucket)] = {
                "expected": expected_rate,
                "actual": actual_rate,
                "count": len(trades),
                "calibration_error": abs(expected_rate - actual_rate),
            }
            total_error += abs(expected_rate - actual_rate) * len(trades)
            total_count += len(trades)

        return {
            "buckets": report,
            "mean_calibration_error": total_error / total_count if total_count else 0.0,
            "total_audits": total_count,
            "window_days": window_days,
        }

    async def accuracy(self, window_days: int = 30) -> float:
        """Overall accuracy of reasoning decisions in the window."""
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        audits = self.db.query_reasoning_audits(since=since, limit=10_000)
        if not audits:
            return 0.0
        wins = sum(1 for a in audits if a.get("outcome_win"))
        return wins / len(audits)
