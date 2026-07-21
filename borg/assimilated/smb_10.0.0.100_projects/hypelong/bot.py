"""
Multi-Asset Long Leverage Bot
RSI + Extender Strategy on Hyperliquid
Forked for BTC, XRP, ETH, SUI, SOL, ZEC, DOGE
"""
import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from exchange import HyperliquidExchange
from strategy import RSIExtenderStrategy, Signal, calculate_ema, calculate_rsi, calculate_stoch_rsi, calculate_bb_pct, calculate_adx
from sessions import get_current_session, analyze_session_trend
from state import log_trade, TradeRecord
from targets import log_symbol_target
from liquidations import get_full_liquidation_picture
from liquidation_clusters import estimate_clusters, get_top_clusters
from alerts import AlertEngine, log_alert
from state import log_liquidation, LiquidationRecord

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("hype_bot")

STATE_FILE = Path("bot_state.json")


class MultiAssetLongBot:
    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.symbols = self.config.get("symbols", [])
        if not self.symbols:
            raise ValueError("No symbols configured in config.yaml")
        
        self.dry_run = self.config["bot"].get("dry_run", True)
        self.check_interval = self.config["bot"].get("check_interval_seconds", 60)
        self.rate_limit_delay = self.config["bot"].get("rate_limit_delay_ms", 200) / 1000.0
        
        # Risk settings
        risk = self.config.get("risk", {})
        self.stop_loss_pct = risk.get("stop_loss_pct", 0)
        self.take_profit_pct = risk.get("take_profit_pct", 0)
        self.trailing_stop_pct = risk.get("trailing_stop_pct", 0.0)
        self.cooldown_minutes = risk.get("cooldown_minutes", 60)
        self.max_open_positions = risk.get("max_open_positions", 5)
        
        # Portfolio limits
        portfolio = self.config.get("portfolio", {})
        self.max_total_position_usd = portfolio.get("max_total_position_usd", float('inf'))
        self.max_active_assets = portfolio.get("max_active_assets", len(self.symbols))
        
        # Default position settings
        defaults = self.config.get("defaults", {})
        self.default_leverage = defaults.get("leverage", 3)
        self.default_margin_mode = defaults.get("margin_mode", "isolated")
        self.default_size_usd = defaults.get("size_usd", 20)
        self.default_max_position_usd = defaults.get("max_position_size_usd", 100)
        
        # Per-asset overrides
        self.assets_config = self.config.get("assets", {})
        
        # Initialize exchange
        wallet = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
        key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        testnet = os.getenv("HYPERLIQUID_TESTNET", "false").lower() == "true"
        
        if not wallet or not key:
            raise ValueError("Missing HYPERLIQUID_WALLET_ADDRESS or HYPERLIQUID_PRIVATE_KEY in environment")
        
        self.exchange = HyperliquidExchange(wallet, key, testnet)
        self.exchange.load_markets()
        
        # Initialize strategy
        strat = self.config.get("strategy", {})
        sessions = strat.get("sessions", {})
        self.strategy = RSIExtenderStrategy(
            ema_enabled=strat.get("ema_trend", {}).get("enabled", True),
            ema_period=strat.get("ema_trend", {}).get("period", 200),
            ema_retest_pct=strat.get("ema_trend", {}).get("retest_tolerance_pct", 0.03),
            rsi_enabled=strat.get("rsi", {}).get("enabled", True),
            rsi_period=strat.get("rsi", {}).get("period", 14),
            rsi_oversold=strat.get("rsi", {}).get("oversold", 30),
            rsi_overbought=strat.get("rsi", {}).get("overbought", 70),
            stoch_rsi_enabled=strat.get("stoch_rsi", {}).get("enabled", False),
            stoch_rsi_period=strat.get("stoch_rsi", {}).get("period", 14),
            stoch_rsi_oversold=strat.get("stoch_rsi", {}).get("oversold", 0.20),
            stoch_rsi_overbought=strat.get("stoch_rsi", {}).get("overbought", 0.80),
            bb_enabled=strat.get("bollinger", {}).get("enabled", False),
            bb_period=strat.get("bollinger", {}).get("period", 20),
            bb_entry_threshold=strat.get("bollinger", {}).get("entry_threshold", 0.10),
            adx_enabled=strat.get("adx", {}).get("enabled", True),
            adx_period=strat.get("adx", {}).get("period", 14),
            adx_ema_period=strat.get("adx", {}).get("ema_period", 14),
            adx_threshold=strat.get("adx", {}).get("threshold", 20),
            entry_mode=strat.get("entry_mode", "all"),
            sessions_enabled=sessions.get("enabled", True),
            entry_session=sessions.get("entry_session", "asian"),
            exit_session=sessions.get("exit_session", "nyc"),
            require_bullish_bias=sessions.get("require_bullish_bias", True),
        )
        
        # Per-symbol state (loaded from disk for persistence)
        self.positions_state: Dict[str, Dict[str, Any]] = {}
        self.load_state()
        
        # Track last time liquidation data was logged to JSONL per symbol
        # (we still fetch every iteration for alerts, but only write once an hour)
        self._last_liq_log_time: Dict[str, datetime] = {}
        
        # Alert engine
        self.alert_engine = AlertEngine()
        
        self._log_startup()
    
    def _should_log_liquidation(self, symbol: str) -> bool:
        """Return True if we haven't written a liquidation record for this symbol in the past hour."""
        now = datetime.now(timezone.utc)
        last = self._last_liq_log_time.get(symbol)
        if last is None:
            return True
        return (now - last).total_seconds() >= 3600
    
    def _get_asset_config(self, symbol: str) -> Dict[str, Any]:
        """Get effective config for a symbol (defaults + overrides)."""
        overrides = self.assets_config.get(symbol, {})
        return {
            "leverage": overrides.get("leverage", self.default_leverage),
            "margin_mode": overrides.get("margin_mode", self.default_margin_mode),
            "size_usd": overrides.get("size_usd", self.default_size_usd),
            "max_position_size_usd": overrides.get("max_position_size_usd", self.default_max_position_usd),
        }
    
    def _log_startup(self):
        logger.info("=" * 60)
        logger.info("Multi-Asset Long Leverage Bot Starting")
        logger.info(f"Assets: {', '.join(self.symbols)}")
        logger.info(f"Mode: {'DRY RUN (no real trades)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"Portfolio Max: ${self.max_total_position_usd} | Max Active: {self.max_active_assets}")
        logger.info(f"Check Interval: {self.check_interval}s | Rate Limit: {self.rate_limit_delay*1000:.0f}ms")
        logger.info("=" * 60)
        for sym in self.symbols:
            cfg = self._get_asset_config(sym)
            logger.info(f"  {sym}: {cfg['leverage']}x {cfg['margin_mode']} | Size ${cfg['size_usd']} | Max ${cfg['max_position_size_usd']}")
        logger.info("=" * 60)
    
    def load_state(self):
        """Load per-symbol state from disk."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                for sym, state in data.items():
                    self.positions_state[sym] = {
                        "entry_price": state.get("entry_price"),
                        "highest_price": state.get("highest_price"),
                        "last_entry_time": datetime.fromisoformat(state["last_entry_time"]) if state.get("last_entry_time") else None,
                    }
                logger.info(f"Loaded state for {len(self.positions_state)} assets")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def save_state(self):
        """Save per-symbol state to disk."""
        data = {}
        for sym, state in self.positions_state.items():
            data[sym] = {
                "entry_price": state.get("entry_price"),
                "highest_price": state.get("highest_price"),
                "last_entry_time": state.get("last_entry_time").isoformat() if state.get("last_entry_time") else None,
            }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save state: {e}")
    
    def setup(self):
        """Set leverage and margin mode for all symbols."""
        if self.dry_run:
            return
        for sym in self.symbols:
            cfg = self._get_asset_config(sym)
            try:
                self.exchange.set_leverage(sym, cfg["leverage"], cfg["margin_mode"])
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Could not set leverage for {sym}: {e}")
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current open position for symbol."""
        positions = self.exchange.fetch_positions(symbol)
        if positions:
            return positions[0]
        return None
    
    def get_position_size_usd(self, position: Optional[Dict[str, Any]]) -> float:
        if not position:
            return 0.0
        return abs(float(position.get("notional", 0)))
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all positions for all symbols."""
        all_positions = {}
        try:
            positions = self.exchange.fetch_positions()
            for pos in positions:
                sym = pos.get("symbol")
                if sym in self.symbols and float(pos.get("contracts", 0)) != 0:
                    all_positions[sym] = pos
        except Exception as e:
            logger.error(f"Failed to fetch all positions: {e}")
        return all_positions
    
    def get_total_position_usd(self) -> float:
        """Get total notional across all open positions."""
        total = 0.0
        for sym in self.symbols:
            pos = self.get_position(sym)
            total += self.get_position_size_usd(pos)
        return total
    
    def get_active_asset_count(self) -> int:
        """Count assets with open positions."""
        count = 0
        for sym in self.symbols:
            pos = self.get_position(sym)
            if self.get_position_size_usd(pos) > 0:
                count += 1
        return count
    
    def check_cooldown(self, symbol: str) -> bool:
        state = self.positions_state.get(symbol, {})
        last = state.get("last_entry_time")
        if last is None:
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60
        return elapsed >= self.cooldown_minutes
    
    def check_exit_conditions(self, symbol: str, position: Dict[str, Any], current_price: float) -> Optional[str]:
        """Check if position should be exited. Returns reason or None."""
        entry_price = float(position.get("entryPrice", 0))
        side = position.get("side", "long")
        
        if entry_price == 0:
            return None
        
        state = self.positions_state.setdefault(symbol, {})
        highest = state.get("highest_price")
        if highest is None or current_price > highest:
            state["highest_price"] = current_price
            highest = current_price
        
        if side == "long":
            # Take Profit (only if enabled)
            if self.take_profit_pct > 0:
                tp_price = entry_price * (1 + self.take_profit_pct / 100)
                if current_price >= tp_price:
                    return f"TAKE PROFIT hit @ ${current_price:.4f} (entry ${entry_price:.4f})"
            
            # Trailing Stop (only if enabled)
            if self.trailing_stop_pct > 0 and highest:
                trail_price = highest * (1 - self.trailing_stop_pct / 100)
                if current_price <= trail_price:
                    return f"TRAILING STOP hit @ ${current_price:.4f} (peak ${highest:.4f})"
            
            # Stop Loss (only if enabled)
            if self.stop_loss_pct > 0:
                sl_price = entry_price * (1 - self.stop_loss_pct / 100)
                if current_price <= sl_price:
                    return f"STOP LOSS hit @ ${current_price:.4f} (entry ${entry_price:.4f})"
        
        return None
    
    def enter_position(self, symbol: str, signal: Signal, current_price: float):
        """Open a new long position or add to existing."""
        cfg = self._get_asset_config(symbol)
        size_usd = cfg["size_usd"]
        max_pos_usd = cfg["max_position_size_usd"]
        
        if not self.check_cooldown(symbol):
            logger.info(f"[{symbol}] In cooldown period, skipping entry")
            return
        
        # Portfolio-level checks
        total_pos = self.get_total_position_usd()
        if total_pos + size_usd > self.max_total_position_usd:
            logger.info(f"[{symbol}] Portfolio max position reached (${total_pos:.2f}), skipping")
            return
        
        active_assets = self.get_active_asset_count()
        position = self.get_position(symbol)
        pos_size = self.get_position_size_usd(position)
        is_new_asset = pos_size == 0
        if is_new_asset and active_assets >= self.max_active_assets:
            logger.info(f"[{symbol}] Max active assets reached ({active_assets}/{self.max_active_assets}), skipping new asset")
            return
        
        # Layered entry checks
        layers = round(pos_size / size_usd) if pos_size > 0 else 0
        if layers >= self.max_open_positions:
            logger.info(f"[{symbol}] Max entry layers reached ({layers}/{self.max_open_positions}), skipping")
            return
        
        if pos_size + size_usd > max_pos_usd:
            logger.info(f"[{symbol}] Max position size reached (${pos_size:.2f}), skipping entry")
            return
        
        if self.dry_run:
            logger.info(f"[{symbol}] [DRY RUN] Would open LONG ${size_usd} @ ${current_price:.4f} | {signal.reason}")
            state = self.positions_state.setdefault(symbol, {})
            state["last_entry_time"] = datetime.now(timezone.utc)
            state["entry_price"] = current_price
            state["highest_price"] = current_price
            self.save_state()
            return
        
        try:
            order = self.exchange.create_market_order(symbol, "buy", size_usd)
            logger.info(f"[{symbol}] LONG order placed: {order.get('id')}")
            state = self.positions_state.setdefault(symbol, {})
            state["last_entry_time"] = datetime.now(timezone.utc)
            state["entry_price"] = current_price
            state["highest_price"] = current_price
            self.save_state()
            
            # Log trade
            try:
                session_trend = analyze_session_trend(self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=100))
            except Exception:
                session_trend = {"bias": "unknown"}
            log_trade(TradeRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                side="long",
                price=current_price,
                size_usd=size_usd,
                leverage=cfg["leverage"],
                session=get_current_session(),
                session_bias=session_trend.get("bias", "unknown"),
                rsi=0.0,
                reason=signal.reason[:100],
                tx_hash=order.get("id"),
            ))
            
            # Attach stop loss and take profit only if enabled
            amount = float(order.get("amount", 0))
            if amount > 0:
                if self.stop_loss_pct > 0:
                    try:
                        sl_price = current_price * (1 - self.stop_loss_pct / 100)
                        self.exchange.create_stop_loss_order(symbol, "sell", amount, sl_price)
                    except Exception as e:
                        logger.warning(f"[{symbol}] Could not attach SL: {e}")
                if self.take_profit_pct > 0:
                    try:
                        tp_price = current_price * (1 + self.take_profit_pct / 100)
                        self.exchange.create_take_profit_order(symbol, "sell", amount, tp_price)
                    except Exception as e:
                        logger.warning(f"[{symbol}] Could not attach TP: {e}")
        except Exception as e:
            logger.error(f"[{symbol}] Failed to enter position: {e}")
    
    def exit_position(self, symbol: str, reason: str):
        """Close the current position for a symbol."""
        if self.dry_run:
            logger.info(f"[{symbol}] [DRY RUN] Would close position: {reason}")
            state = self.positions_state.setdefault(symbol, {})
            state["entry_price"] = None
            state["highest_price"] = None
            self.save_state()
            return
        
        try:
            position = self.get_position(symbol)
            pos_size = self.get_position_size_usd(position)
            
            result = self.exchange.close_position(symbol)
            if result:
                logger.info(f"[{symbol}] Position closed: {reason} | Order: {result.get('id')}")
                
                try:
                    fill_price = float(result.get("price", 0) or result.get("average", 0) or 0)
                except Exception:
                    fill_price = 0.0
                
                try:
                    session_trend = analyze_session_trend(self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=100))
                except Exception:
                    session_trend = {"bias": "unknown"}
                
                log_trade(TradeRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    symbol=symbol,
                    side="long",
                    price=fill_price,
                    size_usd=pos_size,
                    leverage=self._get_asset_config(symbol)["leverage"],
                    session=get_current_session(),
                    session_bias=session_trend.get("bias", "unknown"),
                    rsi=0.0,
                    reason=reason[:100],
                    tx_hash=result.get("id"),
                    trade_type="exit",
                ))
                
            state = self.positions_state.setdefault(symbol, {})
            state["entry_price"] = None
            state["highest_price"] = None
            self.save_state()
        except Exception as e:
            logger.error(f"[{symbol}] Failed to exit position: {e}")
    
    def run_symbol(self, symbol: str):
        """Execute one iteration for a single symbol."""
        try:
            timeframe = self.config["strategy"].get("timeframe", "1h")
            lookback = self.config["strategy"].get("lookback", 250)
            df = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
            
            current_price = df["close"].iloc[-1]
            logger.info(f"[{symbol}] Price: ${current_price:.4f}")
            
            current_hour = pd.Timestamp.now(tz="UTC").hour
            signal = self.strategy.analyze(df, current_hour=current_hour)
            logger.info(f"[{symbol}] Signal: {signal.reason}")
            
            session_trend = analyze_session_trend(df)
            logger.info(f"[{symbol}] Bias: {session_trend.get('bias', 'unknown')} — {session_trend.get('reason', '')}")
            
            # Compute indicators for target logging
            try:
                ema_val = float(calculate_ema(df["close"], self.strategy.ema_period).iloc[-1])
                rsi_val = float(calculate_rsi(df["close"], self.strategy.rsi_period).iloc[-1])
                stoch_val = float(calculate_stoch_rsi(df["close"], self.strategy.rsi_period, self.strategy.stoch_rsi_period).iloc[-1]) if self.strategy.stoch_rsi_enabled else 0.5
                bb_val = float(calculate_bb_pct(df["close"], self.strategy.bb_period).iloc[-1]) if self.strategy.bb_enabled else 0.5
                adx_df = calculate_adx(df, self.strategy.adx_period)
                adx_val = float(adx_df["adx"].iloc[-1])
            except Exception:
                ema_val = rsi_val = stoch_val = bb_val = adx_val = 0.0
            
            position = self.get_position(symbol)
            pos_size = self.get_position_size_usd(position)
            
            # Log target
            state = self.positions_state.get(symbol, {})
            try:
                log_symbol_target(
                    symbol=symbol,
                    price=current_price,
                    ema=ema_val,
                    rsi=rsi_val,
                    stoch_rsi=stoch_val,
                    bb_pct=bb_val,
                    adx=adx_val,
                    session=get_current_session(current_hour),
                    bias=session_trend.get("bias", "unknown"),
                    session_score=session_trend.get("score", 0),
                    position=position,
                    trailing_stop_pct=self.trailing_stop_pct,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                    highest_price=state.get("highest_price"),
                )
            except Exception as e:
                logger.warning(f"[{symbol}] Could not log target: {e}")
            
            # Fetch and log liquidation data + clusters + alerts
            liq_data_for_alerts = {"total_usd": 0, "long_usd": 0, "short_usd": 0, "dominant_side": "neutral"}
            funding_rate_pct = 0.0
            clusters = []
            try:
                base = symbol.split("/")[0]
                funding = self.exchange.exchange.fetchFundingRate(symbol)
                fr = funding.get("fundingRate", 0.0) if funding else 0.0
                oi = float(funding.get("info", {}).get("openInterest", 0)) if funding else 0.0
                funding_rate_pct = fr * 100
                liq_pic = get_full_liquidation_picture(base, fr, oi)
                okx = liq_pic.get("okx_liquidations", {})
                risk = liq_pic.get("risk_proxy", {})
                liq_data_for_alerts = okx
                if self._should_log_liquidation(symbol):
                    log_liquidation(LiquidationRecord(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        symbol=symbol,
                        long_usd=okx.get("long_usd", 0),
                        short_usd=okx.get("short_usd", 0),
                        total_usd=okx.get("total_usd", 0),
                        long_count=okx.get("long_count", 0),
                        short_count=okx.get("short_count", 0),
                        ratio=okx.get("ratio", 1.0),
                        dominant_side=okx.get("dominant_side", "neutral"),
                        funding_rate_pct=risk.get("funding_rate_pct", 0),
                        open_interest=risk.get("open_interest", 0),
                        liq_risk_score=risk.get("score", 0),
                        liq_trend=liq_pic.get("trend", "neutral"),
                    ))
                    self._last_liq_log_time[symbol] = datetime.now(timezone.utc)
                    logger.info(
                        f"[{symbol}] Liquidations logged (hourly): L=${okx.get('long_usd', 0):,.0f} "
                        f"S=${okx.get('short_usd', 0):,.0f} | Trend: {liq_pic.get('trend', 'neutral')}"
                    )
                else:
                    logger.info(
                        f"[{symbol}] Liquidations 1h: L=${okx.get('long_usd', 0):,.0f} "
                        f"S=${okx.get('short_usd', 0):,.0f} | Trend: {liq_pic.get('trend', 'neutral')}"
                    )
                
                # Estimate liquidation clusters
                clusters = estimate_clusters(symbol, current_price, oi, fr * 100)
                top_clusters = get_top_clusters(clusters, limit=3)
                if top_clusters:
                    cluster_msg = " | ".join(
                        f"{c['side'][:1].upper()}{c['leverage']}x@{c['price']:,.0f}({c['distance_pct']:.1f}%)"
                        for c in top_clusters
                    )
                    logger.info(f"[{symbol}] Top liq clusters: {cluster_msg}")
                
            except Exception as e:
                logger.debug(f"[{symbol}] Could not log liquidations: {e}")
            
            # Run alert checks
            try:
                alerts = self.alert_engine.run_all_checks(
                    symbol=symbol,
                    current_price=current_price,
                    liq_data=liq_data_for_alerts,
                    funding_rate_pct=funding_rate_pct,
                    clusters=clusters,
                )
                for alert in alerts:
                    log_alert(alert)
                    logger.warning(f"[{symbol}] ALERT [{alert.severity.upper()}]: {alert.message}")
            except Exception as e:
                logger.debug(f"[{symbol}] Alert check failed: {e}")
            
            # Handle session-based exits
            if signal.exit and position and pos_size > 0:
                logger.info(f"[{symbol}] Session exit signal during {get_current_session(current_hour).upper()} session")
                self.exit_position(symbol, f"Session unload: {session_trend.get('reason', '')}")
                return
            
            if position and pos_size > 0:
                logger.info(f"[{symbol}] Open: {position.get('side')} ${pos_size:.2f} @ {position.get('entryPrice')}")
                
                exit_reason = self.check_exit_conditions(symbol, position, current_price)
                if exit_reason:
                    self.exit_position(symbol, exit_reason)
                else:
                    logger.info(f"[{symbol}] Holding position...")
            else:
                if signal.entry:
                    self.enter_position(symbol, signal, current_price)
                else:
                    logger.info(f"[{symbol}] No entry signal")
        
        except Exception as e:
            logger.exception(f"[{symbol}] Error: {e}")
    
    def run_once(self):
        """Execute one iteration for all symbols."""
        logger.info("-" * 60)
        logger.info(f"Scanning {len(self.symbols)} assets...")
        
        for i, symbol in enumerate(self.symbols):
            self.run_symbol(symbol)
            if i < len(self.symbols) - 1 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
        
        # Portfolio summary
        total = self.get_total_position_usd()
        active = self.get_active_asset_count()
        logger.info(f"Portfolio: ${total:.2f} total | {active}/{self.max_active_assets} active assets")
        logger.info("-" * 60)
    
    def run(self):
        """Main loop."""
        self.setup()
        logger.info(f"Bot running. Checking every {self.check_interval}s. Press Ctrl+C to stop.")
        
        try:
            while True:
                self.run_once()
                logger.info(f"Sleeping {self.check_interval}s...")
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Long Leverage Bot")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit")
    args = parser.parse_args()
    
    bot = MultiAssetLongBot(config_path=args.config)
    
    if args.once:
        bot.run_once()
    else:
        bot.run()


if __name__ == "__main__":
    main()
