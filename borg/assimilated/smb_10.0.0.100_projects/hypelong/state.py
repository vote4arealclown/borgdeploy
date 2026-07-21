"""
Shared state and trade logging for the bot and dashboard.
"""
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

TRADE_LOG = Path("trades.jsonl")
TARGET_LOG = Path("targets.jsonl")
LIQ_LOG = Path("liquidations.jsonl")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    side: str
    price: float
    size_usd: float
    leverage: int
    session: str
    session_bias: str
    rsi: float
    reason: str
    tx_hash: Optional[str] = None
    trade_type: str = "entry"  # "entry" or "exit"


@dataclass
class TargetRecord:
    timestamp: str
    symbol: str
    current_price: float
    target: str
    ready: bool
    session: str
    bias: str
    session_score: float
    checks: Dict[str, Any]
    in_position: bool = False
    position_size_usd: float = 0.0
    entry_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    hard_sl_price: Optional[float] = None
    hard_tp_price: Optional[float] = None
    highest_price: Optional[float] = None


@dataclass
class LiquidationRecord:
    timestamp: str
    symbol: str
    long_usd: float
    short_usd: float
    total_usd: float
    long_count: int
    short_count: int
    ratio: float
    dominant_side: str
    funding_rate_pct: float
    open_interest: float
    liq_risk_score: float
    liq_trend: str
    period_hours: int = 1


def log_trade(record: TradeRecord) -> None:
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(asdict(record), cls=NumpyEncoder) + "\n")


def log_target(record: TargetRecord) -> None:
    with open(TARGET_LOG, "a") as f:
        f.write(json.dumps(asdict(record), cls=NumpyEncoder) + "\n")


def log_liquidation(record: LiquidationRecord) -> None:
    with open(LIQ_LOG, "a") as f:
        f.write(json.dumps(asdict(record), cls=NumpyEncoder) + "\n")


def load_trades(limit: int = 50, symbol: Optional[str] = None) -> List[TradeRecord]:
    if not TRADE_LOG.exists():
        return []
    
    records = []
    with open(TRADE_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = TradeRecord(**json.loads(line))
                    if symbol is None or rec.symbol == symbol:
                        records.append(rec)
                except Exception:
                    continue
    
    if symbol:
        return records[-limit:]
    return records[-limit:]


def load_targets(limit: int = 100, symbol: Optional[str] = None) -> List[TargetRecord]:
    if not TARGET_LOG.exists():
        return []
    
    records = []
    with open(TARGET_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    rec = TargetRecord(**data)
                    if symbol is None or rec.symbol == symbol:
                        records.append(rec)
                except Exception:
                    continue
    
    if symbol:
        return records[-limit:]
    return records[-limit:]


def load_liquidations(limit: int = 100, symbol: Optional[str] = None) -> List[LiquidationRecord]:
    if not LIQ_LOG.exists():
        return []
    
    records = []
    with open(LIQ_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    rec = LiquidationRecord(**data)
                    if symbol is None or rec.symbol == symbol:
                        records.append(rec)
                except Exception:
                    continue
    
    if symbol:
        return records[-limit:]
    return records[-limit:]


def load_all_trades(symbol: Optional[str] = None) -> List[TradeRecord]:
    """Load all trades, optionally filtered by symbol."""
    if not TRADE_LOG.exists():
        return []
    
    records = []
    with open(TRADE_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = TradeRecord(**json.loads(line))
                    if symbol is None or rec.symbol == symbol:
                        records.append(rec)
                except Exception:
                    continue
    return records
