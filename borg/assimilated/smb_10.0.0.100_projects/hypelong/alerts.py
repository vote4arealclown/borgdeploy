"""
Alert system for the trading bot.
Detects spikes in liquidations, funding, OI, and price approaching liq clusters.
"""
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

ALERT_LOG = Path("alerts.jsonl")
ALERT_STATE = Path("alert_state.json")


@dataclass
class AlertRecord:
    timestamp: str
    alert_type: str
    severity: str  # info, warning, critical
    symbol: str
    message: str
    metric: str
    value: float
    threshold: float
    context: Dict[str, Any]
    acknowledged: bool = False


class AlertEngine:
    """
    Detects alert conditions by comparing current metrics against thresholds
    and tracking state across scans.
    """
    
    # Default thresholds
    THRESHOLDS = {
        "liq_total_usd": 5_000_000,        # $5M in 1h = warning
        "liq_total_usd_critical": 20_000_000,  # $20M = critical
        "liq_spike_multiplier": 3.0,        # 3x vs previous scan = spike
        "funding_extreme": 0.01,            # 0.01% = extreme
        "funding_critical": 0.03,           # 0.03% = critical
        "oi_change_pct": 10.0,              # 10% OI change in 1h
        "cluster_proximity_pct": 2.0,       # Price within 2% of liq cluster
        "cascade_min_scans": 3,             # 3 consecutive scans
        "cascade_growth_pct": 20.0,         # 20% growth each scan
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        self.state: Dict[str, Any] = {}
        self.load_state()
    
    def load_state(self):
        if ALERT_STATE.exists():
            try:
                with open(ALERT_STATE, "r") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}
    
    def save_state(self):
        try:
            with open(ALERT_STATE, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save alert state: {e}")
    
    def _get_prev_liq(self, symbol: str) -> Optional[Dict]:
        return self.state.get("last_liquidations", {}).get(symbol)
    
    def _set_prev_liq(self, symbol: str, data: Dict):
        if "last_liquidations" not in self.state:
            self.state["last_liquidations"] = {}
        self.state["last_liquidations"][symbol] = data
    
    def _track_cascade(self, symbol: str, liq_data: Dict) -> Optional[Dict]:
        """Track if liquidations are growing scan-over-scan."""
        key = f"cascade_{symbol}"
        history = self.state.get(key, [])
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_usd": liq_data.get("total_usd", 0),
            "long_usd": liq_data.get("long_usd", 0),
            "short_usd": liq_data.get("short_usd", 0),
            "dominant_side": liq_data.get("dominant_side", "neutral"),
        }
        
        history.append(entry)
        # Keep last 10 scans
        history = history[-10:]
        self.state[key] = history
        
        # Check cascade pattern: N consecutive scans of growing liqs on same side
        min_scans = self.THRESHOLDS["cascade_min_scans"]
        if len(history) < min_scans:
            return None
        
        recent = history[-min_scans:]
        sides = [e["dominant_side"] for e in recent]
        if len(set(sides)) != 1:
            return None
        
        # Check growth
        growing = True
        for i in range(1, len(recent)):
            prev = recent[i - 1]["total_usd"]
            curr = recent[i]["total_usd"]
            if prev == 0:
                continue
            growth = (curr - prev) / prev * 100
            if growth < self.THRESHOLDS["cascade_growth_pct"]:
                growing = False
                break
        
        if growing:
            return {
                "side": sides[0],
                "scans": min_scans,
                "start_total": recent[0]["total_usd"],
                "end_total": recent[-1]["total_usd"],
                "growth_pct": round((recent[-1]["total_usd"] - recent[0]["total_usd"]) / max(recent[0]["total_usd"], 1) * 100, 1),
            }
        return None
    
    def check_liquidation_alerts(self, symbol: str, liq_data: Dict) -> List[AlertRecord]:
        """Check for liquidation-related alerts."""
        alerts = []
        total = liq_data.get("total_usd", 0)
        long_usd = liq_data.get("long_usd", 0)
        short_usd = liq_data.get("short_usd", 0)
        dominant = liq_data.get("dominant_side", "neutral")
        
        # Critical: massive liquidations
        if total >= self.THRESHOLDS["liq_total_usd_critical"]:
            alerts.append(AlertRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                alert_type="LIQUIDATION_CRITICAL",
                severity="critical",
                symbol=symbol,
                message=f"CRITICAL: ${total:,.0f} liquidated in 1h (${long_usd:,.0f} long / ${short_usd:,.0f} short)",
                metric="total_usd",
                value=total,
                threshold=self.THRESHOLDS["liq_total_usd_critical"],
                context=liq_data,
            ))
        # Warning: significant liquidations
        elif total >= self.THRESHOLDS["liq_total_usd"]:
            alerts.append(AlertRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                alert_type="LIQUIDATION_SPIKE",
                severity="warning",
                symbol=symbol,
                message=f"Warning: ${total:,.0f} liquidated in 1h — {dominant.upper()}s getting wiped",
                metric="total_usd",
                value=total,
                threshold=self.THRESHOLDS["liq_total_usd"],
                context=liq_data,
            ))
        
        # Spike detection: vs previous scan
        prev = self._get_prev_liq(symbol)
        if prev and prev.get("total_usd", 0) > 0:
            prev_total = prev["total_usd"]
            multiplier = total / prev_total
            if multiplier >= self.THRESHOLDS["liq_spike_multiplier"]:
                alerts.append(AlertRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    alert_type="LIQUIDATION_SPIKE_VS_PRIOR",
                    severity="warning",
                    symbol=symbol,
                    message=f"Liquidations {multiplier:.1f}x vs prior scan (${prev_total:,.0f} → ${total:,.0f})",
                    metric="spike_multiplier",
                    value=multiplier,
                    threshold=self.THRESHOLDS["liq_spike_multiplier"],
                    context={"prev": prev, "current": liq_data},
                ))
        
        # Cascade detection
        cascade = self._track_cascade(symbol, liq_data)
        if cascade:
            alerts.append(AlertRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                alert_type="LIQUIDATION_CASCADE",
                severity="critical",
                symbol=symbol,
                message=f"CASCADE: {cascade['side'].upper()} liquidations growing {cascade['scans']} scans in a row (+{cascade['growth_pct']}%)",
                metric="cascade_growth_pct",
                value=cascade["growth_pct"],
                threshold=self.THRESHOLDS["cascade_growth_pct"],
                context=cascade,
            ))
        
        self._set_prev_liq(symbol, liq_data)
        return alerts
    
    def check_funding_alerts(self, symbol: str, funding_rate_pct: float) -> List[AlertRecord]:
        """Check for funding rate extremes."""
        alerts = []
        abs_fr = abs(funding_rate_pct)
        
        if abs_fr >= self.THRESHOLDS["funding_critical"]:
            alerts.append(AlertRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                alert_type="FUNDING_CRITICAL",
                severity="critical",
                symbol=symbol,
                message=f"CRITICAL funding: {funding_rate_pct:+.4f}% — extreme {'long' if funding_rate_pct > 0 else 'short'} crowdedness",
                metric="funding_rate_pct",
                value=funding_rate_pct,
                threshold=self.THRESHOLDS["funding_critical"],
                context={"side": "long" if funding_rate_pct > 0 else "short"},
            ))
        elif abs_fr >= self.THRESHOLDS["funding_extreme"]:
            alerts.append(AlertRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                alert_type="FUNDING_EXTREME",
                severity="warning",
                symbol=symbol,
                message=f"Extreme funding: {funding_rate_pct:+.4f}% — {'longs' if funding_rate_pct > 0 else 'shorts'} paying heavy",
                metric="funding_rate_pct",
                value=funding_rate_pct,
                threshold=self.THRESHOLDS["funding_extreme"],
                context={"side": "long" if funding_rate_pct > 0 else "short"},
            ))
        
        return alerts
    
    def check_cluster_alerts(self, symbol: str, current_price: float, clusters: List[Dict]) -> List[AlertRecord]:
        """Check if price is approaching a liquidation cluster."""
        alerts = []
        proximity_pct = self.THRESHOLDS["cluster_proximity_pct"]
        
        for cluster in clusters:
            cluster_price = cluster.get("price", 0)
            if cluster_price <= 0:
                continue
            
            distance_pct = abs(current_price - cluster_price) / current_price * 100
            if distance_pct <= proximity_pct:
                side = cluster.get("side", "unknown")
                estimated_usd = cluster.get("estimated_usd", 0)
                
                alerts.append(AlertRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    alert_type="LIQ_CLUSTER_APPROACH",
                    severity="warning",
                    symbol=symbol,
                    message=f"Price ${current_price:.2f} is {distance_pct:.1f}% from {side} liq cluster at ${cluster_price:.2f} (est. ${estimated_usd:,.0f})",
                    metric="cluster_distance_pct",
                    value=distance_pct,
                    threshold=proximity_pct,
                    context=cluster,
                ))
        
        return alerts
    
    def run_all_checks(
        self,
        symbol: str,
        current_price: float,
        liq_data: Dict,
        funding_rate_pct: float,
        clusters: Optional[List[Dict]] = None,
    ) -> List[AlertRecord]:
        """Run all alert checks and return triggered alerts."""
        alerts = []
        alerts.extend(self.check_liquidation_alerts(symbol, liq_data))
        alerts.extend(self.check_funding_alerts(symbol, funding_rate_pct))
        if clusters:
            alerts.extend(self.check_cluster_alerts(symbol, current_price, clusters))
        
        if alerts:
            self.save_state()
        return alerts


def log_alert(record: AlertRecord) -> None:
    with open(ALERT_LOG, "a") as f:
        f.write(json.dumps(asdict(record), cls=NumpyEncoder) + "\n")


def load_alerts(limit: int = 50, symbol: Optional[str] = None, severity: Optional[str] = None) -> List[AlertRecord]:
    if not ALERT_LOG.exists():
        return []
    
    records = []
    with open(ALERT_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    rec = AlertRecord(**data)
                    if symbol and rec.symbol != symbol:
                        continue
                    if severity and rec.severity != severity:
                        continue
                    records.append(rec)
                except Exception:
                    continue
    return records[-limit:]


# Reuse the numpy encoder from state
from state import NumpyEncoder
