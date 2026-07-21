"""
RSI + Extender Strategy
Based on the BiGGER EXTENDER concept:
- EMA trend filter (price above EMA = uptrend)
- RSI momentum (oversold = long entry)
- ADX trend strength (ADX above threshold and crossing EMA = strong trend)
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sessions import analyze_session_trend, is_entry_session, is_exit_session

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    entry: bool = False
    exit: bool = False
    side: str = "long"          # Only long for this bot
    confidence: float = 0.0
    reason: str = ""


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3) -> pd.Series:
    """
    Calculate Stochastic RSI (%K line).
    Returns values between 0.0 and 1.0.
    """
    rsi = calculate_rsi(series, rsi_period)
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_k = stoch.rolling(window=k_period).mean()
    return stoch_k


def calculate_bb_pct(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    Calculate Bollinger Bands %B.
    Returns values where 0 = lower band, 1 = upper band.
    """
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct = (series - lower) / (upper - lower)
    return pct


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate ADX, +DI, -DI.
    Returns DataFrame with adx, plus_di, minus_di columns.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    result = pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)
    return result


class RSIExtenderStrategy:
    def __init__(
        self,
        ema_enabled: bool = True,
        ema_period: int = 200,
        ema_retest_pct: float = 0.03,
        rsi_enabled: bool = True,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        stoch_rsi_enabled: bool = False,
        stoch_rsi_period: int = 14,
        stoch_rsi_oversold: float = 0.20,
        stoch_rsi_overbought: float = 0.80,
        bb_enabled: bool = False,
        bb_period: int = 20,
        bb_entry_threshold: float = 0.10,
        adx_enabled: bool = True,
        adx_period: int = 14,
        adx_ema_period: int = 14,
        adx_threshold: float = 20,
        entry_mode: str = "all",
        sessions_enabled: bool = True,
        entry_session: str = "asian",
        exit_session: str = "nyc",
        require_bullish_bias: bool = True,
    ):
        self.ema_enabled = ema_enabled
        self.ema_period = ema_period
        self.ema_retest_pct = ema_retest_pct
        self.rsi_enabled = rsi_enabled
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_rsi_enabled = stoch_rsi_enabled
        self.stoch_rsi_period = stoch_rsi_period
        self.stoch_rsi_oversold = stoch_rsi_oversold
        self.stoch_rsi_overbought = stoch_rsi_overbought
        self.bb_enabled = bb_enabled
        self.bb_period = bb_period
        self.bb_entry_threshold = bb_entry_threshold
        self.adx_enabled = adx_enabled
        self.adx_period = adx_period
        self.adx_ema_period = adx_ema_period
        self.adx_threshold = adx_threshold
        self.entry_mode = entry_mode  # "all" = all enabled must agree
        self.sessions_enabled = sessions_enabled
        self.entry_session = entry_session
        self.exit_session = exit_session
        self.require_bullish_bias = require_bullish_bias

    def analyze(self, df: pd.DataFrame, current_hour: Optional[int] = None) -> Signal:
        min_period = max(
            self.ema_period,
            self.rsi_period,
            self.stoch_rsi_period * 2 if self.stoch_rsi_enabled else 0,
            self.bb_period if self.bb_enabled else 0,
            self.adx_period,
        )
        if len(df) < min_period + 10:
            return Signal(reason="Not enough data")

        df = df.copy()
        
        # Indicators
        if self.ema_enabled:
            df["ema"] = calculate_ema(df["close"], self.ema_period)
        if self.rsi_enabled:
            df["rsi"] = calculate_rsi(df["close"], self.rsi_period)
        if self.stoch_rsi_enabled:
            df["stoch_rsi"] = calculate_stoch_rsi(df["close"], self.rsi_period, self.stoch_rsi_period)
        if self.bb_enabled:
            df["bb_pct"] = calculate_bb_pct(df["close"], self.bb_period)
        if self.adx_enabled:
            adx_df = calculate_adx(df, self.adx_period)
            df["adx"] = adx_df["adx"]
            df["adx_ema"] = calculate_ema(df["adx"], self.adx_ema_period)
            df["plus_di"] = adx_df["plus_di"]
            df["minus_di"] = adx_df["minus_di"]
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        checks = []
        reasons = []
        
        # EMA Trend Filter: Price above EMA = uptrend -> OK for long
        # Also allow retest entries within ema_retest_pct below EMA
        if self.ema_enabled:
            price_above_ema = latest["close"] > latest["ema"]
            price_near_ema = latest["close"] > latest["ema"] * (1 - self.ema_retest_pct)
            ema_ok = price_above_ema or price_near_ema
            checks.append(ema_ok)
            if price_above_ema:
                reasons.append(f"Price > EMA{self.ema_period} (${latest['close']:.4f} vs ${latest['ema']:.4f})")
            elif price_near_ema:
                reasons.append(f"Price near EMA{self.ema_period} retest (${latest['close']:.4f} vs ${latest['ema']:.4f}, tol {self.ema_retest_pct:.1%})")
            else:
                reasons.append(f"Price below EMA{self.ema_period} (${latest['close']:.4f} vs ${latest['ema']:.4f})")
        
        # Momentum: RSI oversold OR StochRSI oversold (catches fast reversals)
        momentum_ok = False
        if self.rsi_enabled:
            rsi_val = latest["rsi"]
            rsi_ok = rsi_val <= self.rsi_oversold
            if rsi_ok:
                momentum_ok = True
                reasons.append(f"RSI={rsi_val:.1f} (oversold <= {self.rsi_oversold})")
            else:
                reasons.append(f"RSI={rsi_val:.1f} (not oversold > {self.rsi_oversold})")
        
        if self.stoch_rsi_enabled:
            stoch_val = latest["stoch_rsi"]
            stoch_ok = stoch_val <= self.stoch_rsi_oversold
            if stoch_ok:
                momentum_ok = True
                reasons.append(f"StochRSI={stoch_val:.2f} (oversold <= {self.stoch_rsi_oversold})")
            else:
                reasons.append(f"StochRSI={stoch_val:.2f} (not oversold > {self.stoch_rsi_oversold})")
        
        if self.rsi_enabled and self.stoch_rsi_enabled:
            checks.append(momentum_ok)
        elif self.rsi_enabled:
            checks.append(rsi_ok)
        elif self.stoch_rsi_enabled:
            checks.append(stoch_ok)
        
        # Bollinger Bands %B: confirmation when price is near lower band
        if self.bb_enabled:
            bb_val = latest["bb_pct"]
            bb_ok = bb_val <= self.bb_entry_threshold
            checks.append(bb_ok)
            reasons.append(f"BB%={bb_val:.2f} ({'near lower' if bb_ok else 'mid/upper'} <= {self.bb_entry_threshold})")
        
        # ADX: Trend strength + ADX crossing above its EMA
        if self.adx_enabled:
            adx_val = latest["adx"]
            adx_ema_val = latest["adx_ema"]
            adx_strong = adx_val >= self.adx_threshold
            adx_cross = (prev["adx"] <= prev["adx_ema"]) and (latest["adx"] > latest["adx_ema"])
            adx_ok = adx_strong or adx_cross
            checks.append(adx_ok)
            reasons.append(f"ADX={adx_val:.1f} (threshold={self.adx_threshold}, cross={'yes' if adx_cross else 'no'})")
        
        # Session analysis
        exit_signal = False
        if self.sessions_enabled:
            current_session = is_exit_session(current_hour) if current_hour is not None else is_exit_session()
            entry_session = is_entry_session(current_hour) if current_hour is not None else is_entry_session()
            
            trend = analyze_session_trend(df)
            bias = trend.get("bias", "unknown")
            score = trend.get("score", 0)
            session_reason = trend.get("reason", "")
            
            reasons.append(f"Session: {bias} (score {score:+.0f})")
            
            # Entry only in Asian session
            if entry_session:
                if self.require_bullish_bias:
                    # Score-based entry: allow unless strongly bearish
                    # If RSI is very oversold, even bearish session score can be a buy (NYC dip)
                    if score >= -15:
                        bias_ok = True
                        reasons.append("Asian session entry, bias OK")
                    elif score >= -50 and (rsi_val if self.rsi_enabled else 50) <= 35:
                        bias_ok = True
                        reasons.append("Asian session entry, buying NYC dip (oversold)")
                    else:
                        bias_ok = False
                        reasons.append("Asian session entry, too bearish — wait")
                    checks.append(bias_ok)
                else:
                    reasons.append("Asian session entry (no bias filter)")
            else:
                checks.append(False)
                reasons.append("Not Asian session — no entries")
            
            # Exit only on strongly bearish score during NYC
            if current_session and score <= -50:
                exit_signal = True
                reasons.append("NYC session, strongly bearish — consider unload")
        
        # Combine signals
        if not checks:
            return Signal(reason="No indicators enabled")
        
        if self.entry_mode == "all":
            entry = all(checks)
        else:
            entry = any(checks)
        
        confidence = sum(checks) / len(checks) if checks else 0.0
        
        reason = " | ".join(reasons)
        if exit_signal:
            reason = f"EXIT SIGNAL: {reason}"
        elif entry:
            reason = f"LONG ENTRY: {reason}"
        else:
            reason = f"NO ENTRY: {reason}"
        
        return Signal(entry=entry, exit=exit_signal, confidence=confidence, reason=reason)
