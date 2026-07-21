"""
HYPE Bot Web Dashboard (Single Asset)
Accessible from any browser. No wallet needed.
Shows indicators, entry signals, funding rate, and news.
Runs on port 8082 to avoid conflict with multi-asset dashboard.
"""
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from dotenv import load_dotenv

load_dotenv()

import ccxt
from strategy import RSIExtenderStrategy, calculate_rsi, calculate_ema, calculate_adx, calculate_stoch_rsi, calculate_bb_pct
from sessions import analyze_session_trend, get_current_session
from hype_state import load_trades
from state import load_targets, load_liquidations
from news import get_hype_news

app = Flask(__name__)

OPTIMAL = {"ema": 100, "rsi": 45, "adx": 20}
EXCHANGE = ccxt.hyperliquid()


def fetch_ohlcv(hours: int = 12):
    """Fetch OHLCV data for specified hours."""
    limit = max(hours, 250)
    ohlcv = EXCHANGE.fetch_ohlcv("HYPE/USDC:USDC", timeframe="1h", limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_funding():
    """Fetch current funding rate."""
    try:
        fr = EXCHANGE.fetchFundingRate("HYPE/USDC:USDC")
        return {
            "rate": fr.get("fundingRate", 0),
            "rate_pct": fr.get("fundingRate", 0) * 100,
            "next_time": fr.get("fundingDatetime", ""),
            "mark_price": fr.get("markPrice", 0),
            "index_price": fr.get("indexPrice", 0),
            "premium": fr.get("info", {}).get("premium", "0"),
            "open_interest": fr.get("info", {}).get("openInterest", "0"),
        }
    except Exception:
        return {
            "rate": 0, "rate_pct": 0, "next_time": "",
            "mark_price": 0, "index_price": 0,
            "premium": "0", "open_interest": "0",
        }


def get_status():
    df = fetch_ohlcv(100)
    price = float(df["close"].iloc[-1])
    hour = df.index[-1].hour
    
    ema = float(calculate_ema(df["close"], OPTIMAL["ema"]).iloc[-1])
    rsi = float(calculate_rsi(df["close"], 14).iloc[-1])
    stoch_rsi = float(calculate_stoch_rsi(df["close"], 14, 14, 3).iloc[-1])
    bb_pct = float(calculate_bb_pct(df["close"], 20, 2.0).iloc[-1])
    adx_df = calculate_adx(df, 14)
    adx = float(adx_df["adx"].iloc[-1])
    
    strat = RSIExtenderStrategy(
        ema_enabled=True, ema_period=OPTIMAL["ema"],
        rsi_enabled=True, rsi_period=14, rsi_oversold=OPTIMAL["rsi"], rsi_overbought=70,
        stoch_rsi_enabled=True, stoch_rsi_period=14, stoch_rsi_oversold=0.25, stoch_rsi_overbought=0.80,
        bb_enabled=True, bb_period=20, bb_entry_threshold=0.10,
        adx_enabled=True, adx_period=14, adx_ema_period=14, adx_threshold=OPTIMAL["adx"],
        entry_mode="all",
        sessions_enabled=True, entry_session="asian", exit_session="nyc", require_bullish_bias=True,
    )
    signal = strat.analyze(df, current_hour=hour)
    trend = analyze_session_trend(df)
    funding = fetch_funding()
    
    price_ok = price > ema
    rsi_ok = rsi <= OPTIMAL["rsi"]
    stoch_ok = stoch_rsi <= 0.25
    bb_ok = bb_pct <= 0.10
    adx_ok = adx >= OPTIMAL["adx"]
    session_ok = 0 <= hour < 8
    ready = price_ok and (rsi_ok or stoch_ok) and adx_ok and session_ok
    
    now = datetime.now(timezone.utc)
    hour_now = now.hour
    
    if 0 <= hour_now < 8:
        session_status = "IN ASIAN SESSION"
        session_ends = now.replace(hour=8, minute=0, second=0, microsecond=0)
        time_remaining = session_ends - now
        session_timer = f"Asian ends in {int(time_remaining.total_seconds() // 60)}m"
    else:
        session_status = "OUTSIDE ASIAN SESSION"
        if hour_now >= 8:
            next_asian = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            next_asian = now.replace(hour=0, minute=0, second=0, microsecond=0)
        time_to_asian = next_asian - now
        hours_to = int(time_to_asian.total_seconds() // 3600)
        mins_to = int((time_to_asian.total_seconds() % 3600) // 60)
        session_timer = f"Asian starts in {hours_to}h {mins_to}m"
    
    trades = load_trades(limit=1)
    last_trade_time = trades[-1].timestamp if trades else None
    last_trade_ago = ""
    if last_trade_time:
        lt = datetime.fromisoformat(last_trade_time)
        ago = now - lt
        days = ago.days
        hours = ago.seconds // 3600
        mins = (ago.seconds % 3600) // 60
        if days > 0:
            last_trade_ago = f"{days}d {hours}h ago"
        elif hours > 0:
            last_trade_ago = f"{hours}h {mins}m ago"
        else:
            last_trade_ago = f"{mins}m ago"
    
    target = ""
    if not price_ok:
        target = f"Price needs to hold above EMA ${ema:.2f}"
    elif not (rsi_ok or stoch_ok):
        target = f"RSI needs to drop from {rsi:.1f} to ≤ {OPTIMAL['rsi']} (or StochRSI ≤ 0.25)"
    elif not bb_ok:
        target = f"BB% needs to drop from {bb_pct:.2f} to ≤ 0.10 (near lower band)"
    elif not adx_ok:
        target = f"ADX needs to rise to ≥ {OPTIMAL['adx']}"
    elif not session_ok:
        target = f"Wait for Asian session ({session_timer})"
    else:
        target = f"ALL CONDITIONS MET — ENTER NOW ({session_timer})"
    
    return {
        "price": price,
        "ema": ema,
        "rsi": rsi,
        "stoch_rsi": stoch_rsi,
        "bb_pct": bb_pct,
        "adx": adx,
        "session": get_current_session(hour).upper(),
        "bias": str(trend.get("bias", "unknown")),
        "session_score": float(trend.get("score", 0)),
        "bias_reason": str(trend.get("reason", "")),
        "ready": ready,
        "checks": {
            "price_above_ema": price_ok,
            "rsi_oversold": rsi_ok,
            "stoch_rsi_oversold": stoch_ok,
            "bb_near_lower": bb_ok,
            "adx_strong": adx_ok,
            "asian_session": session_ok,
        },
        "target": target,
        "signal": str(signal.reason),
        "utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "session_status": session_status,
        "session_timer": session_timer,
        "last_trade_ago": last_trade_ago,
        "funding": funding,
    }


# Cache for news (refresh every 5 minutes)
_news_cache = {"data": None, "timestamp": None}

def get_cached_news():
    now = datetime.now(timezone.utc)
    if _news_cache["timestamp"] is None or (now - _news_cache["timestamp"]).total_seconds() > 300:
        _news_cache["data"] = get_hype_news()
        _news_cache["timestamp"] = now
    return _news_cache["data"]


def get_chart_data(hours: int = 48):
    """Get chart data for the last N hours with indicators and trade markers."""
    df = fetch_ohlcv(hours)
    
    df["ema100"] = calculate_ema(df["close"], 100)
    df["ema200"] = calculate_ema(df["close"], 200)
    df["rsi14"] = calculate_rsi(df["close"], 14)
    df["stoch_rsi"] = calculate_stoch_rsi(df["close"], 14, 14, 3)
    adx_df = calculate_adx(df, 14)
    df["adx14"] = adx_df["adx"]
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_upper"] = bb_mid + 2 * bb_std
    
    cutoff = df.index[-1] - timedelta(hours=hours)
    df_slice = df[df.index >= cutoff].copy()
    
    labels = [t.strftime("%m/%d %H:%M") for t in df_slice.index]
    
    # Trade markers
    trades = load_trades(limit=50)
    markers = []
    for t in trades:
        try:
            ts = datetime.fromisoformat(t.timestamp)
            if ts >= df_slice.index[0] and ts <= df_slice.index[-1]:
                markers.append({
                    "x": ts.strftime("%m/%d %H:%M"),
                    "y": t.price,
                    "type": t.trade_type,
                    "side": t.side,
                })
        except Exception:
            pass
    
    return {
        "labels": labels,
        "price": df_slice["close"].round(4).tolist(),
        "ema100": df_slice["ema100"].round(4).tolist(),
        "ema200": df_slice["ema200"].round(4).tolist(),
        "rsi14": df_slice["rsi14"].round(2).tolist(),
        "stoch_rsi": df_slice["stoch_rsi"].round(3).tolist(),
        "adx14": df_slice["adx14"].round(2).tolist(),
        "bb_upper": df_slice["bb_upper"].round(4).tolist(),
        "bb_lower": df_slice["bb_lower"].round(4).tolist(),
        "markers": markers,
    }


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HYPE Bot Dashboard</title>
    <meta http-equiv="refresh" content="15">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a0f; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; padding: 20px; }
        h1 { color: #00d4ff; margin-bottom: 4px; }
        .subtitle { color: #666; font-size: 14px; margin-bottom: 16px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; max-width: 1400px; margin-bottom: 16px; }
        .card { background: #13131f; border: 1px solid #222; border-radius: 10px; padding: 16px; }
        .card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 12px; }
        .status-ready { border-color: #00ff88; background: rgba(0,255,136,0.04); }
        .status-wait { border-color: #ff4444; background: rgba(255,68,68,0.04); }
        .big { font-size: 28px; font-weight: 700; }
        .green { color: #00ff88; }
        .red { color: #ff4444; }
        .yellow { color: #ffd166; }
        .cyan { color: #00d4ff; }
        .dim { color: #666; font-size: 13px; }
        .check { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a2e; font-size: 13px; }
        .check:last-child { border: none; }
        .target { margin-top: 12px; padding: 10px; background: #1a1a2e; border-radius: 6px; font-size: 13px; color: #ffd166; }
        .funding-box { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; }
        .funding-box div { padding: 4px 0; }
        .updated { position: fixed; bottom: 10px; right: 10px; color: #444; font-size: 11px; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th { text-align: left; padding: 6px; color: #888; font-weight: 500; border-bottom: 1px solid #222; }
        td { padding: 6px; border-bottom: 1px solid #1a1a2e; }
        .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; text-transform: uppercase; }
        .badge-asian { background: #ffd16620; color: #ffd166; }
        .badge-nyc { background: #00d4ff20; color: #00d4ff; }
        .badge-london { background: #ff6b6b20; color: #ff6b6b; }
        .badge-long { background: #00ff8820; color: #00ff88; }
        .badge-short { background: #ff444420; color: #ff4444; }
        .badge-exit { background: #ffd16620; color: #ffd166; }
        .badge-entry { background: #00d4ff20; color: #00d4ff; }
        .chart-container { position: relative; height: 320px; width: 100%; }
        .chart-row { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; }
    </style>
</head>
<body>
    <h1>🚀 HYPE Long Bot</h1>
    <p class="subtitle">
        EMA {{ ema }} | RSI ≤ {{ rsi_thresh }} | ADX ≥ {{ adx_thresh }} | 
        <span style="color:#00d4ff;">{{ status.utc_time }}</span>
    </p>
    
    <div class="grid">
        <div class="card {{ 'status-ready' if status.ready else 'status-wait' }}">
            <h2>Entry Status</h2>
            <div class="big {{ 'green' if status.ready else 'red' }}">
                {{ 'READY' if status.ready else 'WAIT' }}
            </div>
            <div class="dim" style="margin:6px 0;">{{ status.session_status }} — {{ status.session_timer }}</div>
            {% if status.last_trade_ago %}<div class="dim">Last trade: {{ status.last_trade_ago }}</div>{% endif %}
            <div class="target">📍 {{ status.target }}</div>
        </div>
        
        <div class="card">
            <h2>Price & Time</h2>
            <div class="big">${{ "%.4f"|format(status.price) }}</div>
            <div class="dim">EMA{{ ema }}: ${{ "%.2f"|format(status.ema) }}</div>
            <div class="dim">Session: {{ status.session }}</div>
            <div class="dim">Bias: <span class="{{ 'green' if status.session_score > 15 else 'red' if status.session_score < -15 else 'yellow' }}">{{ status.bias.upper() }}</span> ({{ "%+.0f"|format(status.session_score) }})</div>
            <div style="margin-top:10px; padding-top:10px; border-top:1px solid #1a1a2e;">
                <div class="dim">UTC: <span style="color:#fff;">{{ status.utc_time }}</span></div>
                <div class="dim">Local: {{ status.local_time }}</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Indicators</h2>
            {% for name, ok, label in [
                ('Price > EMA' + ema|string, status.checks.price_above_ema, '$' + "%.2f"|format(status.price) + ' > $' + "%.2f"|format(status.ema)),
                ('RSI ≤ ' + rsi_thresh|string, status.checks.rsi_oversold, "%.1f"|format(status.rsi)),
                ('StochRSI ≤ 0.25', status.checks.stoch_rsi_oversold, "%.2f"|format(status.stoch_rsi)),
                ('BB %B ≤ 0.10', status.checks.bb_near_lower, "%.2f"|format(status.bb_pct)),
                ('ADX ≥ ' + adx_thresh|string, status.checks.adx_strong, "%.1f"|format(status.adx)),
                ('Asian Session', status.checks.asian_session, status.session),
            ] %}
            <div class="check">
                <span>{{ '✅' if ok else '❌' }} {{ name }}</span>
                <span class="{{ 'green' if ok else 'red' }}">{{ label }}</span>
            </div>
            {% endfor %}
        </div>
        
        <div class="card">
            <h2>Funding Rate</h2>
            <div class="funding-box">
                <div>Rate: <span class="{{ 'green' if status.funding.rate_pct < 0 else 'red' }}">{{ "%.4f"|format(status.funding.rate_pct * 100) }}%</span></div>
                <div>Hourly: {{ "%.6f"|format(status.funding.rate) }}</div>
                <div>Mark: ${{ "%.2f"|format(status.funding.mark_price) }}</div>
                <div>Index: ${{ "%.2f"|format(status.funding.index_price) }}</div>
                <div class="dim" style="grid-column:1/-1;">Next: {{ status.funding.next_time }}</div>
                <div class="dim" style="grid-column:1/-1;">Premium: {{ status.funding.premium }} | OI: {{ "%.0f"|format(status.funding.open_interest|float) }} HYPE</div>
            </div>
        </div>
        
        <div class="card" style="grid-column:1/-1;">
            <h2>Signal Logic</h2>
            <p style="font-size:12px; line-height:1.4; color:#aaa;">{{ status.signal }}</p>
            <p class="dim" style="margin-top:8px;">{{ status.bias_reason }}</p>
        </div>
    </div>
    
    <div class="grid" style="margin-top:16px;">
        <div class="card" style="grid-column:1/-1;">
            <h2>💥 Liquidation Data (1h)</h2>
            {% if liqs %}
            {% set latest = liqs[-1] %}
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:12px; margin-bottom:12px;">
                <div>
                    <div class="dim" style="font-size:11px;">LONG LIQS</div>
                    <div class="red" style="font-size:18px; font-weight:700;">${{ "%.0f"|format(latest.long_usd) }}</div>
                    <div class="dim" style="font-size:11px;">{{ latest.long_count }} orders</div>
                </div>
                <div>
                    <div class="dim" style="font-size:11px;">SHORT LIQS</div>
                    <div class="green" style="font-size:18px; font-weight:700;">${{ "%.0f"|format(latest.short_usd) }}</div>
                    <div class="dim" style="font-size:11px;">{{ latest.short_count }} orders</div>
                </div>
                <div>
                    <div class="dim" style="font-size:11px;">TOTAL</div>
                    <div style="font-size:18px; font-weight:700;">${{ "%.0f"|format(latest.total_usd) }}</div>
                    <div class="dim" style="font-size:11px;">{{ latest.total_count }} orders</div>
                </div>
                <div>
                    <div class="dim" style="font-size:11px;">DOMINANT</div>
                    <div class="{{ 'red' if latest.dominant_side == 'long' else 'green' }}" style="font-size:18px; font-weight:700;">{{ latest.dominant_side.upper() }}</div>
                    <div class="dim" style="font-size:11px;">Ratio {{ "%.2f"|format(latest.ratio) }}</div>
                </div>
                <div>
                    <div class="dim" style="font-size:11px;">FUNDING</div>
                    <div class="{{ 'green' if latest.funding_rate_pct < 0 else 'red' }}" style="font-size:18px; font-weight:700;">{{ "%.4f"|format(latest.funding_rate_pct) }}%</div>
                    <div class="dim" style="font-size:11px;">OI ${{ "%.0f"|format(latest.open_interest) }}</div>
                </div>
                <div>
                    <div class="dim" style="font-size:11px;">LIQ TREND</div>
                    <div class="{{ 'green' if 'bullish' in latest.liq_trend else 'red' if 'bearish' in latest.liq_trend else 'yellow' }}" style="font-size:14px; font-weight:700;">{{ latest.liq_trend.upper().replace('_', ' ') }}</div>
                    <div class="dim" style="font-size:11px;">Score {{ "%.0f"|format(latest.liq_risk_score) }}</div>
                </div>
            </div>
            <table style="margin-top:10px;">
                <tr><th>Time</th><th>Long</th><th>Short</th><th>Total</th><th>Dom</th><th>Funding</th><th>Trend</th></tr>
                {% for l in liqs|reverse %}
                <tr>
                    <td style="white-space:nowrap; font-size:11px;">{{ l.timestamp[:19] }}</td>
                    <td class="red" style="font-size:11px;">${{ "%.0f"|format(l.long_usd) }}</td>
                    <td class="green" style="font-size:11px;">${{ "%.0f"|format(l.short_usd) }}</td>
                    <td style="font-size:11px;">${{ "%.0f"|format(l.total_usd) }}</td>
                    <td style="font-size:11px;"><span class="badge {{ 'badge-short' if l.dominant_side == 'short' else 'badge-long' }}">{{ l.dominant_side.upper() }}</span></td>
                    <td class="{{ 'green' if l.funding_rate_pct < 0 else 'red' }}" style="font-size:11px;">{{ "%.4f"|format(l.funding_rate_pct) }}%</td>
                    <td class="{{ 'green' if 'bullish' in l.liq_trend else 'red' if 'bearish' in l.liq_trend else 'yellow' }}" style="font-size:11px;">{{ l.liq_trend.upper().replace('_', ' ') }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="dim">No liquidation data logged yet.</p>
            {% endif %}
        </div>
    </div>
    
    <div class="grid" style="margin-top:16px;">
        <div class="card" style="grid-column:1/-1;">
            <h2>Price Chart + EMA + Bollinger Bands</h2>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
            </div>
        </div>
    </div>
    
    <div class="grid chart-row" style="margin-top:16px;">
        <div class="card">
            <h2>Stochastic RSI</h2>
            <div class="chart-container" style="height:180px;">
                <canvas id="stochChart"></canvas>
            </div>
        </div>
        <div class="card">
            <h2>ADX</h2>
            <div class="chart-container" style="height:180px;">
                <canvas id="adxChart"></canvas>
            </div>
        </div>
    </div>
    
    <div class="grid" style="margin-top:16px;">
        <div class="card" style="grid-column:1/-1;">
            <h2>Trade Log</h2>
            {% if trades %}
            <table>
                <tr><th>Time</th><th>Type</th><th>Side</th><th>Price</th><th>Size</th><th>Session</th><th>Reason</th></tr>
                {% for t in trades %}
                <tr>
                    <td>{{ t.timestamp[:19] }}</td>
                    <td><span class="badge badge-{{ t.trade_type|default('entry') }}">{{ t.trade_type|default('entry')|upper }}</span></td>
                    <td><span class="badge badge-{{ t.side }}">{{ t.side.upper() }}</span></td>
                    <td>${{ "%.2f"|format(t.price) }}</td>
                    <td>${{ "%.0f"|format(t.size_usd) }}</td>
                    <td><span class="badge badge-{{ t.session }}">{{ t.session.upper() }}</span></td>
                    <td style="max-width:250px; overflow:hidden; text-overflow:ellipsis;">{{ t.reason }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="dim">No trades logged yet.</p>
            {% endif %}
        </div>
    </div>
    
    <div class="grid" style="margin-top:16px;">
        <div class="card" style="grid-column:1/-1;">
            <h2>🎯 Target History</h2>
            {% if targets %}
            <table>
                <tr>
                    <th>Time</th>
                    <th>Price</th>
                    <th>Target</th>
                    <th>Ready</th>
                    <th>RSI</th>
                    <th>Stoch</th>
                    <th>ADX</th>
                    <th>Session</th>
                    <th>Bias</th>
                    <th>In Pos</th>
                    <th>Trail Stop</th>
                </tr>
                {% for tgt in targets|reverse %}
                <tr>
                    <td style="white-space:nowrap;">{{ tgt.timestamp[:19] }}</td>
                    <td>${{ "%.4f"|format(tgt.current_price) }}</td>
                    <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; font-size:11px;" class="{{ 'green' if tgt.ready else 'yellow' }}">{{ tgt.target }}</td>
                    <td><span class="badge {{ 'badge-ready' if tgt.ready else 'badge-wait' }}">{{ 'YES' if tgt.ready else 'NO' }}</span></td>
                    <td>{{ "%.1f"|format(tgt.checks.rsi) }}</td>
                    <td>{{ "%.2f"|format(tgt.checks.stoch_rsi) }}</td>
                    <td>{{ "%.1f"|format(tgt.checks.adx) }}</td>
                    <td><span class="badge badge-{{ tgt.session|lower }}">{{ tgt.session.upper() }}</span></td>
                    <td class="{{ 'green' if tgt.session_score > 15 else 'red' if tgt.session_score < -15 else 'yellow' }}">{{ tgt.bias.upper()[:4] }}</td>
                    <td>{{ 'YES' if tgt.in_position else 'NO' }}</td>
                    <td class="dim">{{ "%.4f"|format(tgt.trailing_stop_price) if tgt.trailing_stop_price else '-' }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="dim">No target history logged yet.</p>
            {% endif %}
        </div>
    </div>
    
    <div class="grid" style="margin-top:16px;">
        <div class="card" style="grid-column:1/-1;">
            <h2>Latest HYPE News & Mentions</h2>
            {% if news.news %}
            <table>
                <tr><th>Source</th><th>Headline</th><th>Date</th></tr>
                {% for item in news.news %}
                <tr>
                    <td><span class="badge" style="background:#00d4ff20; color:#00d4ff;">{{ item.source }}</span></td>
                    <td><a href="{{ item.url }}" target="_blank" style="color:#e0e0e0; text-decoration:none;">{{ item.title }}</a></td>
                    <td class="dim" style="white-space:nowrap;">{{ item.date[:16] if item.date else '' }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="dim">No recent news mentions found.</p>
            {% endif %}
            <div style="margin-top:10px; display:flex; gap:12px; flex-wrap:wrap;">
                {% if news.coin_info.twitter %}
                <a href="https://twitter.com/{{ news.coin_info.twitter }}" target="_blank" style="color:#00d4ff; font-size:12px;">🐦 @{{ news.coin_info.twitter }}</a>
                {% endif %}
                {% if news.coin_info.homepage %}
                <a href="{{ news.coin_info.homepage }}" target="_blank" style="color:#00d4ff; font-size:12px;">🌐 Hyperliquid</a>
                {% endif %}
                {% if news.coin_info.telegram %}
                <a href="https://t.me/{{ news.coin_info.telegram }}" target="_blank" style="color:#00d4ff; font-size:12px;">✈️ Telegram</a>
                {% endif %}
                {% if news.coin_info.market_cap_rank %}
                <span class="dim" style="font-size:12px;">🏆 Rank #{{ news.coin_info.market_cap_rank }}</span>
                {% endif %}
            </div>
            <p class="dim" style="margin-top:8px; font-size:11px;">Fetched: {{ news.fetched_at[:19] if news.fetched_at else '' }}</p>
        </div>
    </div>
    
    <div style="text-align:center; padding:20px 0; color:#444; font-size:12px;">
        <a href="https://www.mandominutes.com/Latest" target="_blank" style="color:#00d4ff; text-decoration:none;">📰 Mando Minutes</a>
    </div>
    
    <div class="updated">Updated: {{ status.utc_time }}</div>
    
    <script>
    fetch('/api/chart')
        .then(r => r.json())
        .then(data => {
            const labels = data.labels;
            const common = {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { labels: { color: '#aaa', font: { size: 10 } } } },
                scales: {
                    x: { ticks: { color: '#666', font: { size: 9 }, maxTicksLimit: 8 }, grid: { color: '#1a1a2e' } },
                    y: { ticks: { color: '#666', font: { size: 9 } }, grid: { color: '#1a1a2e' } }
                }
            };
            
            const markerData = data.markers.map(m => ({ x: m.x, y: m.y, type: m.type, side: m.side }));
            new Chart(document.getElementById('priceChart'), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'Price', data: data.price, borderColor: '#e0e0e0', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
                        { label: 'EMA100', data: data.ema100, borderColor: '#00d4ff', borderWidth: 1, pointRadius: 0, tension: 0.1 },
                        { label: 'EMA200', data: data.ema200, borderColor: '#ff00ff', borderWidth: 1, pointRadius: 0, tension: 0.1, borderDash: [5,5] },
                        { label: 'BB Upper', data: data.bb_upper, borderColor: '#4444ff', borderWidth: 0.8, pointRadius: 0, fill: false, tension: 0.1 },
                        { label: 'BB Lower', data: data.bb_lower, borderColor: '#4444ff', borderWidth: 0.8, pointRadius: 0, fill: '-1', backgroundColor: 'rgba(68,68,255,0.05)', tension: 0.1 },
                        { label: 'Entry', data: markerData.filter(m => m.type === 'entry').map(m => ({ x: m.x, y: m.y })), backgroundColor: '#00ff88', pointStyle: 'triangle', pointRadius: 6, showLine: false },
                        { label: 'Exit', data: markerData.filter(m => m.type === 'exit').map(m => ({ x: m.x, y: m.y })), backgroundColor: '#ffd166', pointStyle: 'triangle', pointRadius: 6, showLine: false, rotation: 180 },
                    ]
                },
                options: common
            });
            
            new Chart(document.getElementById('stochChart'), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'StochRSI', data: data.stoch_rsi, borderColor: '#ffd166', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
                        { label: 'Oversold', data: labels.map(() => 0.20), borderColor: '#00ff88', borderWidth: 1, pointRadius: 0, borderDash: [4,4] },
                        { label: 'Overbought', data: labels.map(() => 0.80), borderColor: '#ff4444', borderWidth: 1, pointRadius: 0, borderDash: [4,4] },
                    ]
                },
                options: {
                    ...common,
                    scales: {
                        x: { ticks: { color: '#666', font: { size: 9 }, maxTicksLimit: 6 }, grid: { color: '#1a1a2e' } },
                        y: { min: 0, max: 1, ticks: { color: '#666', font: { size: 9 } }, grid: { color: '#1a1a2e' } }
                    }
                }
            });
            
            new Chart(document.getElementById('adxChart'), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'ADX', data: data.adx14, borderColor: '#ff00ff', borderWidth: 1.5, pointRadius: 0, tension: 0.1 },
                        { label: 'Threshold', data: labels.map(() => 20), borderColor: '#888', borderWidth: 1, pointRadius: 0, borderDash: [4,4] },
                    ]
                },
                options: {
                    ...common,
                    scales: {
                        x: { ticks: { color: '#666', font: { size: 9 }, maxTicksLimit: 6 }, grid: { color: '#1a1a2e' } },
                        y: { min: 0, max: 60, ticks: { color: '#666', font: { size: 9 } }, grid: { color: '#1a1a2e' } }
                    }
                }
            });
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    status = get_status()
    trades = load_trades(limit=20)
    targets = load_targets(limit=50, symbol="HYPE/USDC:USDC")
    liqs = load_liquidations(limit=20, symbol="HYPE/USDC:USDC")
    news = get_cached_news()
    return render_template_string(
        HTML_TEMPLATE,
        status=status,
        trades=trades,
        targets=targets,
        liqs=liqs,
        news=news,
        ema=OPTIMAL["ema"],
        rsi_thresh=OPTIMAL["rsi"],
        adx_thresh=OPTIMAL["adx"],
    )


@app.route("/api/status")
def api_status():
    return jsonify(get_status())


@app.route("/api/chart")
def api_chart():
    return jsonify(get_chart_data(hours=48))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082))
    app.run(host="0.0.0.0", port=port, debug=False)
