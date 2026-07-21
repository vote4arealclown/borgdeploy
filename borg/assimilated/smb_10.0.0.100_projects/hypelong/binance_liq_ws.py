"""
Binance Futures Liquidation WebSocket Collector.
Public stream — no API key needed.
Runs in background thread and accumulates liquidation events.
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from collections import defaultdict

from websocket import WebSocketApp

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"


class BinanceLiquidationCollector:
    """
    Background websocket collector for Binance futures liquidations.
    Aggregates events into 1-hour buckets per symbol.
    """
    
    def __init__(self):
        self.events: List[Dict] = []
        self.ws: WebSocketApp = None
        self.thread: threading.Thread = None
        self.running = False
        self.lock = threading.Lock()
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # Binance forceOrder format:
            # {
            #   "e":"forceOrder",
            #   "E":1630987879188,
            #   "o":{
            #     "s":"BTCUSDT",
            #     "S":"SELL",
            #     "f":"IOC",
            #     "q":"0.014",
            #     "p":"42328.43",
            #     "ap":"42328.43",
            #     "X":"FILLED",
            #     "l":"0.014",
            #     "z":"0.014",
            #     "T":1630987879187
            #   }
            # }
            if data.get("e") == "forceOrder":
                o = data.get("o", {})
                event = {
                    "symbol": o.get("s", ""),
                    "side": o.get("S", ""),      # SELL = long liquidation, BUY = short liquidation
                    "price": float(o.get("p", 0)),
                    "qty": float(o.get("q", 0)),
                    "time_ms": int(o.get("T", 0)),
                }
                with self.lock:
                    self.events.append(event)
                    # Prune old events (> 2 hours)
                    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp() * 1000)
                    self.events = [e for e in self.events if e["time_ms"] >= cutoff]
        except Exception as e:
            logger.debug(f"WS parse error: {e}")
    
    def _on_error(self, ws, error):
        logger.debug(f"Binance WS error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("Binance liquidation WS closed")
        if self.running:
            time.sleep(5)
            self.start()
    
    def _on_open(self, ws):
        logger.info("Binance liquidation WS connected")
    
    def start(self):
        if self.running and self.thread and self.thread.is_alive():
            return
        self.running = True
        self.ws = WebSocketApp(
            BINANCE_WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()
    
    def get_aggregated(self, symbol: str, hours: int = 1) -> Dict[str, Any]:
        """
        Aggregate liquidations for a symbol over N hours.
        symbol format: BTCUSDT (Binance native)
        """
        cutoff_ms = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
        
        with self.lock:
            matching = [e for e in self.events if e["symbol"] == symbol and e["time_ms"] >= cutoff_ms]
        
        if not matching:
            return {"total_usd": 0, "long_usd": 0, "short_usd": 0, "count": 0, "dominant_side": "neutral"}
        
        long_usd = 0.0
        short_usd = 0.0
        # SELL side liquidation = longs getting liquidated
        # BUY side liquidation = shorts getting liquidated
        for e in matching:
            usd = e["price"] * e["qty"]
            if e["side"] == "SELL":
                long_usd += usd
            else:
                short_usd += usd
        
        total = long_usd + short_usd
        return {
            "total_usd": round(total, 2),
            "long_usd": round(long_usd, 2),
            "short_usd": round(short_usd, 2),
            "count": len(matching),
            "dominant_side": "long" if long_usd > short_usd else "short",
            "source": "binance_ws",
        }
    
    def get_all_symbols(self) -> List[str]:
        """Get list of symbols that have recent liquidation data."""
        with self.lock:
            symbols = set(e["symbol"] for e in self.events)
        return sorted(list(symbols))


# Global singleton instance
_collector: BinanceLiquidationCollector = None


def get_collector() -> BinanceLiquidationCollector:
    global _collector
    if _collector is None:
        _collector = BinanceLiquidationCollector()
        _collector.start()
    return _collector
