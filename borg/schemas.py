"""Pydantic schemas for data validation and API contracts."""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class CandleInput(BaseModel):
    """Validated market candle (OHLCV)."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symbol": "EURUSD",
            "ts": "2025-01-15T12:30:00Z",
            "open": 1.0840,
            "high": 1.0860,
            "low": 1.0820,
            "close": 1.0850,
            "volume": 1_500_000,
        }
    })

    symbol: str = Field(min_length=2, max_length=12)
    ts: datetime
    open: float = Field(gt=0, lt=1e6)
    high: float = Field(gt=0, lt=1e6)
    low: float = Field(gt=0, lt=1e6)
    close: float = Field(gt=0, lt=1e6)
    volume: float = Field(ge=0, le=1e12)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.match(r"^[A-Z0-9]{2,12}$", v):
            raise ValueError(f"Invalid symbol format: {v}")
        return v

    @model_validator(mode="after")
    def validate_ohlc(self) -> "CandleInput":
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must be within [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must be within [low, high]")
        return self


class ForecastInput(BaseModel):
    """Validated forecast submission."""

    symbol: str = Field(min_length=2, max_length=12)
    direction: Direction
    confidence: float = Field(ge=0.0, le=100.0)
    rationale: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("confidence")
    @classmethod
    def confidence_sane(cls, v: float) -> float:
        if v > 95:
            raise ValueError("confidence > 95 is considered overconfident; cap at 95")
        return v


class ForecastResult(BaseModel):
    """Forecast produced by a strategy or the brain."""

    symbol: str
    direction: Direction
    confidence: float
    rationale: Optional[str] = None
    model_used: str = "unknown"
    raw_analysis: Optional[str] = None


class LearningInput(BaseModel):
    """Validated learning / memory entry."""

    task_id: Optional[int] = None
    summary: str = Field(min_length=3, max_length=2000)
    detail: Optional[str] = Field(default=None, max_length=10000)
    tags: Optional[str] = Field(default=None, max_length=500)


class SystemStatus(BaseModel):
    """Runtime health snapshot."""

    cpu_percent: float
    memory_used_mb: float
    memory_total_mb: float
    db_path: str
    ollama_reachable: bool
    active_symbols: list[str]
    last_cycle_at: Optional[datetime] = None


class EpisodeOutcome(BaseModel):
    """Outcome of a forecast episode."""

    win: bool
    correct: bool
    resolved_at: Optional[datetime] = None
    horizon_return: Optional[float] = None
    outcome: Optional[str] = None


class EpisodeSignal(BaseModel):
    """Signal that produced an episode."""

    direction: str
    confidence: float
    rationale: Optional[str] = None
    size_multiplier: float = 1.0


class Episode(BaseModel):
    """Structured episodic memory entry tied to a forecast outcome."""

    id: Optional[int] = None
    timestamp: datetime
    actor: str
    trigger: str
    market_state: dict[str, Any]
    regime: str
    trade_signal: EpisodeSignal
    executed: bool = True
    outcome: Optional[EpisodeOutcome] = None
    reasoning_output: Optional[dict[str, Any]] = None
    embedding: Optional[list[float]] = None


class StrategyEfficacy(BaseModel):
    """Performance summary for a strategy in a regime."""

    strategy: str
    regime: str
    win_rate: float
    avg_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    sample_size: int
    window_days: int = 30


class ImageGenerationInput(BaseModel):
    """Validated image generation request."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prompt": "a serene mountain lake at sunset, digital art",
            "model": "flux",
            "width": 1024,
            "height": 1024,
            "seed": 42,
        }
    })

    prompt: str = Field(min_length=1, max_length=32000)
    model: Optional[str] = Field(default=None, max_length=64)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    seed: Optional[int] = Field(default=None, ge=-1)
    response_format: str = Field(default="url", pattern="^(url|b64_json)$")


class ImageGenerationResult(BaseModel):
    """Result of an image generation call."""

    status: str
    prompt: str
    model: str
    url: Optional[str] = None
    b64_json: Optional[str] = None
    local_path: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
