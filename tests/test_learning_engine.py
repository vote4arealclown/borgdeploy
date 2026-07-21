"""Tests for the learning engine."""
from __future__ import annotations

import pytest

from borg.coordinator import StrategyCoordinator
from borg.db import Database
from borg.learning import LearningEngine


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'learning.db'}"
    database = Database(db_url)
    yield database


def test_compute_regime_gradients(fresh_db) -> None:
    coordinator = StrategyCoordinator(database=fresh_db)
    learner = LearningEngine(coordinator=coordinator)
    backtest = {
        "trades": [
            {
                "regime": "bull_low_vol",
                "win": True,
                "horizon_return": 0.001,
                "reasoning": {
                    "signals_input": [{"strategy": "mr", "action": "BUY", "confidence": 70.0}]
                },
            }
            for _ in range(5)
        ]
        + [
            {
                "regime": "bull_low_vol",
                "win": False,
                "horizon_return": -0.001,
                "reasoning": {
                    "signals_input": [{"strategy": "mr", "action": "BUY", "confidence": 70.0}]
                },
            }
            for _ in range(5)
        ],
    }
    gradients = learner.compute_regime_gradients(backtest)
    assert "mr::bull_low_vol" in gradients
    assert gradients["mr::bull_low_vol"]["win_rate"] == 0.5


def test_propose_weight_updates(fresh_db) -> None:
    coordinator = StrategyCoordinator(database=fresh_db)
    learner = LearningEngine(coordinator=coordinator)
    backtest = {
        "trades": [
            {
                "regime": "bull_low_vol",
                "win": True,
                "horizon_return": 0.002,
                "reasoning": {
                    "signals_input": [{"strategy": "mr", "action": "BUY", "confidence": 70.0}]
                },
            }
            for _ in range(10)
        ],
    }
    proposed = learner.propose_weight_updates(backtest, {})
    assert ("mr", "bull_low_vol") in proposed
    assert 0.5 <= proposed[("mr", "bull_low_vol")] <= 2.0


def test_validate_updates_rejects_overfitting(fresh_db) -> None:
    coordinator = StrategyCoordinator(database=fresh_db)
    learner = LearningEngine(coordinator=coordinator)
    train = {"sharpe": 3.0}
    val = {"sharpe": 1.0}
    assert learner.validate_updates({}, train, val) is False


def test_validate_updates_accepts_reasonable(fresh_db) -> None:
    coordinator = StrategyCoordinator(database=fresh_db)
    learner = LearningEngine(coordinator=coordinator)
    train = {"sharpe": 1.2}
    val = {"sharpe": 1.0}
    assert learner.validate_updates({}, train, val) is True
