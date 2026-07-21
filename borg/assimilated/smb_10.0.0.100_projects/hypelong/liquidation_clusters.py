"""
Liquidation cluster estimator.
Calculates estimated liquidation price zones around current price
based on leverage tiers and open interest.
"""
import logging
from typing import List, Dict, Any, Optional

import ccxt

logger = logging.getLogger(__name__)

# Max leverage by asset (approximate, exchange-dependent)
# These are typical max leverage values across major perp exchanges
LEVERAGE_TIERS = {
    "BTC": [2, 3, 5, 10, 20, 50, 100, 125],
    "ETH": [2, 3, 5, 10, 20, 50, 100],
    "SOL": [2, 3, 5, 10, 20, 50],
    "XRP": [2, 3, 5, 10, 20, 50],
    "DOGE": [2, 3, 5, 10, 20, 50],
    "SUI": [2, 3, 5, 10, 20, 50],
    "ZEC": [2, 3, 5, 10, 20],
    "HYPE": [2, 3, 5, 10, 20],
}

# Default tiers for unknown assets
DEFAULT_TIERS = [2, 3, 5, 10, 20, 50]


def estimate_clusters(
    symbol: str,
    current_price: float,
    open_interest: float,
    funding_rate_pct: float,
    use_orderbook: bool = False,
    exchange: Any = None,
) -> List[Dict[str, Any]]:
    """
    Estimate liquidation clusters for a symbol.
    
    Returns list of dicts with:
        - price: estimated liquidation price
        - side: 'long' or 'short'
        - leverage: leverage tier
        - distance_pct: distance from current price
        - estimated_usd: estimated liquidation size at this cluster
        - density_score: 0-100, higher = more estimated liqs here
    """
    base = symbol.split("/")[0] if "/" in symbol else symbol
    tiers = LEVERAGE_TIERS.get(base.upper(), DEFAULT_TIERS)
    
    clusters = []
    
    # Estimate position distribution based on funding + OI
    # Positive funding = more longs, negative = more shorts
    funding_bias = max(-1.0, min(1.0, funding_rate_pct / 0.01))  # Normalize to -1..1
    long_weight = 0.5 + funding_bias * 0.3  # 0.2 to 0.8
    short_weight = 1.0 - long_weight
    
    # Higher leverage = smaller positions (retail), lower leverage = bigger (whales)
    # We assume OI is distributed inversely with leverage tier
    for lev in tiers:
        leverage_factor = 1.0 / lev  # Higher lev = smaller factor
        
        # Long liquidation price
        long_liq_price = current_price * (1 - leverage_factor)
        long_est_usd = open_interest * long_weight * leverage_factor
        long_density = calculate_density(
            current_price, long_liq_price, lev, open_interest, long_weight, funding_rate_pct, "long"
        )
        clusters.append({
            "price": round(long_liq_price, 2),
            "side": "long",
            "leverage": lev,
            "distance_pct": round(leverage_factor * 100, 2),
            "estimated_usd": round(long_est_usd, 2),
            "density_score": round(long_density, 1),
        })
        
        # Short liquidation price
        short_liq_price = current_price * (1 + leverage_factor)
        short_est_usd = open_interest * short_weight * leverage_factor
        short_density = calculate_density(
            current_price, short_liq_price, lev, open_interest, short_weight, funding_rate_pct, "short"
        )
        clusters.append({
            "price": round(short_liq_price, 2),
            "side": "short",
            "leverage": lev,
            "distance_pct": round(leverage_factor * 100, 2),
            "estimated_usd": round(short_est_usd, 2),
            "density_score": round(short_density, 1),
        })
    
    # Sort by density score descending
    clusters.sort(key=lambda x: x["density_score"], reverse=True)
    
    # If orderbook available, refine estimates
    if use_orderbook and exchange:
        try:
            clusters = refine_with_orderbook(symbol, current_price, clusters, exchange)
        except Exception as e:
            logger.debug(f"Could not refine clusters with orderbook: {e}")
    
    return clusters


def calculate_density(
    current_price: float,
    liq_price: float,
    leverage: int,
    open_interest: float,
    side_weight: float,
    funding_rate_pct: float,
    side: str,
) -> float:
    """
    Calculate a density score (0-100) for a liquidation cluster.
    Higher = more likely to have significant liquidations at this level.
    """
    score = 0.0
    
    # Base: higher OI = higher density
    oi_factor = min(50.0, open_interest / 1_000_000)  # Cap at $1M OI
    score += oi_factor
    
    # Side alignment with funding
    if side == "long" and funding_rate_pct > 0:
        score += 20 * min(1.0, funding_rate_pct / 0.03)
    elif side == "short" and funding_rate_pct < 0:
        score += 20 * min(1.0, abs(funding_rate_pct) / 0.03)
    
    # Leverage tier: mid-range leverage (10-50x) tends to have most retail participation
    if 10 <= leverage <= 50:
        score += 15
    elif leverage < 10:
        score += 5  # Whales, fewer positions
    else:
        score += 10  # Degens, very small positions
    
    # Distance: closer clusters are more immediate threats
    distance_pct = abs(current_price - liq_price) / current_price * 100
    if distance_pct < 1.0:
        score += 10
    elif distance_pct < 3.0:
        score += 5
    
    return min(100.0, score)


def refine_with_orderbook(
    symbol: str,
    current_price: float,
    clusters: List[Dict],
    exchange: Any,
) -> List[Dict]:
    """
    Refine cluster estimates using order book depth.
    Thin orderbook = liquidation cascade risk increases.
    """
    try:
        ob = exchange.fetch_order_book(symbol, limit=50)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        
        for cluster in clusters:
            liq_price = cluster["price"]
            side = cluster["side"]
            
            # Calculate depth between current price and liq price
            if side == "long" and liq_price < current_price:
                # Long liqs happen when price drops → check bid depth
                depth = sum(
                    vol for p, vol in bids
                    if liq_price <= p <= current_price
                )
            elif side == "short" and liq_price > current_price:
                # Short liqs happen when price rises → check ask depth
                depth = sum(
                    vol for p, vol in asks
                    if current_price <= p <= liq_price
                )
            else:
                depth = 0
            
            # Thin depth = higher cascade risk
            if depth < 1:
                cluster["density_score"] = min(100, cluster["density_score"] + 15)
                cluster["thin_depth"] = True
            else:
                cluster["thin_depth"] = False
            
            cluster["depth_to_cluster"] = round(depth, 4)
    except Exception as e:
        logger.debug(f"Orderbook refinement failed: {e}")
    
    return clusters


def get_top_clusters(
    clusters: List[Dict],
    side: Optional[str] = None,
    min_density: float = 30.0,
    limit: int = 5,
) -> List[Dict]:
    """Get top liquidation clusters filtered by side and minimum density."""
    filtered = [c for c in clusters if c["density_score"] >= min_density]
    if side:
        filtered = [c for c in filtered if c["side"] == side]
    return filtered[:limit]


def format_cluster_summary(clusters: List[Dict]) -> str:
    """Format clusters into a readable summary."""
    if not clusters:
        return "No significant liquidation clusters detected."
    
    lines = []
    for c in clusters[:5]:
        emoji = "🔴" if c["side"] == "long" else "🟢"
        lines.append(
            f"{emoji} {c['side'].upper()} {c['leverage']}x @ ${c['price']:,.2f} "
            f"({c['distance_pct']:.1f}% away) | Density: {c['density_score']:.0f}/100 "
            f"| Est. ${c['estimated_usd']:,.0f}"
        )
    return "\n".join(lines)
