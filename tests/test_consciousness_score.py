"""Tests for consciousness score calculation."""
from __future__ import annotations

from borg.consciousness_score import ConsciousnessScore


def test_consciousness_score_perfect() -> None:
    score = ConsciousnessScore.calculate(
        reasoning_accuracy=1.0,
        learning_updates_count=20,
        report_count=100,
        performance_consistency=1.0,
    )
    assert score == 100.0


def test_consciousness_score_zero() -> None:
    score = ConsciousnessScore.calculate(
        reasoning_accuracy=0.0,
        learning_updates_count=0,
        report_count=0,
        performance_consistency=0.0,
    )
    assert score == 0.0


def test_consciousness_score_partial() -> None:
    score = ConsciousnessScore.calculate(
        reasoning_accuracy=0.5,
        learning_updates_count=5,
        report_count=15,
        performance_consistency=0.5,
    )
    assert 0.0 < score < 100.0
