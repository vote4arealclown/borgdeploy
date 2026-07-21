"""
Hyperliquid exchange wrapper using CCXT.
Handles authentication, market data, orders, and positions.
"""
import os
import logging
from typing import Optional, Dict, Any, List

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class HyperliquidExchange:
    def __init__(self, wallet_address: str, private_key: str, testnet: bool = False):
        self.wallet_address = wallet_address
        self.testnet = testnet
        config = {
            "walletAddress": wallet_address,
            "privateKey": private_key,
            "options": {
                "defaultType": "swap",
            },
        }
        if testnet:
            config["options"]["sandbox"] = True

        self.exchange = ccxt.hyperliquid(config)
        self.markets_loaded = False

    def load_markets(self) -> Dict[str, Any]:
        if not self.markets_loaded:
            self.exchange.load_markets()
            self.markets_loaded = True
        return self.exchange.markets

    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balance."""
        return self.exchange.fetch_balance()

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 250) -> pd.DataFrame:
        """
        Fetch OHLCV candles and return as a pandas DataFrame.
        Columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching {limit} candles for {symbol} @ {timeframe}")
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch current open positions."""
        positions = self.exchange.fetch_positions()
        if symbol:
            positions = [p for p in positions if p.get("symbol") == symbol and float(p.get("contracts", 0)) != 0]
        return positions

    def set_leverage(self, symbol: str, leverage: int, margin_mode: str = "isolated") -> None:
        """Set leverage and margin mode for a symbol."""
        logger.info(f"Setting {symbol} to {leverage}x {margin_mode} margin")
        self.exchange.set_margin_mode(margin_mode.lower(), symbol, params={"leverage": leverage})

    def create_market_order(self, symbol: str, side: str, amount_usd: float, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a market order.
        side: "buy" or "sell"
        amount_usd: USD notional to trade
        """
        if params is None:
            params = {}
        
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker.get("last")
        if not price:
            raise ValueError("Could not fetch current price for order sizing")
        
        amount = amount_usd / price
        logger.info(f"Market {side} {amount:.4f} {symbol} (~${amount_usd} @ ${price})")
        return self.exchange.create_order(symbol, "market", side, amount, params=params)

    def create_stop_loss_order(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict[str, Any]:
        """Create a stop-loss order."""
        logger.info(f"Stop-loss {side} {amount:.4f} {symbol} @ ${stop_price}")
        return self.exchange.create_order(
            symbol,
            "market",
            side,
            amount,
            params={"stopLossPrice": stop_price, "reduceOnly": True}
        )

    def create_take_profit_order(self, symbol: str, side: str, amount: float, tp_price: float) -> Dict[str, Any]:
        """Create a take-profit order."""
        logger.info(f"Take-profit {side} {amount:.4f} {symbol} @ ${tp_price}")
        return self.exchange.create_order(
            symbol,
            "market",
            side,
            amount,
            params={"takeProfitPrice": tp_price, "reduceOnly": True}
        )

    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Close the entire position for a symbol."""
        positions = self.fetch_positions(symbol)
        if not positions:
            logger.info(f"No open position for {symbol} to close")
            return None
        
        pos = positions[0]
        contracts = float(pos.get("contracts", 0))
        side = pos.get("side", "long")
        
        if contracts == 0:
            return None
        
        close_side = "sell" if side == "long" else "buy"
        logger.info(f"Closing {side} position: {contracts} {symbol}")
        return self.exchange.create_order(symbol, "market", close_side, abs(contracts), params={"reduceOnly": True})
