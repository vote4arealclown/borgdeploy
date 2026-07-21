"""Consciousness score: composite maturity metric for the agent."""
from __future__ import annotations

from typing import Any


class ConsciousnessScore:
    """Measure "consciousness" on a scale 0-100."""

    @staticmethod
    def calculate(
        reasoning_accuracy: float,
        learning_updates_count: int,
        report_count: int,
        performance_consistency: float,
    ) -> float:
        """Weighted score combining transparency, learning, reflection, and adaptability."""
        score = 0.0

        # Transparency: well-calibrated reasoning (30%)
        score += reasoning_accuracy * 0.3

        # Learning quality: autonomous updates deployed (25%)
        score += min(learning_updates_count / 10, 1.0) * 0.25

        # Reflection: regular reports generated (25%)
        score += min(report_count / 30, 1.0) * 0.25

        # Adaptability: performance consistency under changing regimes (20%)
        score += performance_consistency * 0.2

        return min(100.0, score * 100)

    @staticmethod
    def from_system_state(state: dict[str, Any]) -> float:
        """Convenience helper to compute from a system-state dict."""
        return ConsciousnessScore.calculate(
            reasoning_accuracy=state.get("reasoning_accuracy", 0.0),
            learning_updates_count=state.get("learning_updates_count", 0),
            report_count=state.get("report_count", 0),
            performance_consistency=state.get("performance_consistency", 0.0),
        )
