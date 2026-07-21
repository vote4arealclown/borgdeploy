"""
Fetch latest news/mentions for HYPE/Hyperliquid.
Uses CoinGecko Pro API + RSS feeds.
"""
import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict

API_KEY = os.getenv("COINGECKO_API_KEY", "")
HEADERS = {"x-cg-demo-api-key": API_KEY} if API_KEY else {}
BASE_URL = "https://api.coingecko.com"

RSS_SOURCES = [
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"),
]


def fetch_rss_news() -> List[Dict]:
    """Fetch and filter RSS feeds for HYPE mentions."""
    mentions = []
    seen = set()

    for source_name, url in RSS_SOURCES:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue

            root = ET.fromstring(r.content)
            items = root.findall(".//item")

            for item in items:
                title_el = item.find("title")
                desc_el = item.find("description")
                link_el = item.find("link")
                date_el = item.find("pubDate")

                title = title_el.text if title_el is not None else ""
                desc = desc_el.text if desc_el is not None else ""
                link = link_el.text if link_el is not None else ""
                pub_date = date_el.text if date_el is not None else ""

                text = (title + " " + desc).lower()
                if "hype" in text or "hyperliquid" in text:
                    key = title[:60]
                    if key not in seen:
                        seen.add(key)
                        mentions.append({
                            "source": source_name,
                            "title": title,
                            "url": link,
                            "date": pub_date,
                        })
        except Exception:
            continue

    return mentions[:10]


def fetch_coin_info() -> Dict:
    """Fetch coin info from CoinGecko Pro API."""
    try:
        url = f"{BASE_URL}/api/v3/coins/hyperliquid"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "false",
        }
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code != 200:
            return {}

        data = r.json()
        links = data.get("links", {})
        community = data.get("community_data", {})
        market = data.get("market_data", {})

        return {
            "twitter": links.get("twitter_screen_name", ""),
            "homepage": links.get("homepage", [""])[0] if links.get("homepage") else "",
            "telegram": links.get("telegram_channel_identifier", ""),
            "reddit": links.get("subreddit_url", ""),
            "twitter_followers": community.get("twitter_followers", 0),
            "reddit_subscribers": community.get("reddit_subscribers", 0),
            "market_cap_rank": market.get("market_cap_rank", 0),
            "sentiment_votes_up": market.get("sentiment_votes_up_percentage", 0),
            "sentiment_votes_down": market.get("sentiment_votes_down_percentage", 0),
        }
    except Exception:
        return {}


def fetch_coingecko_news() -> List[Dict]:
    """Fetch news from CoinGecko status updates endpoint (Pro)."""
    if not API_KEY:
        return []
    try:
        url = f"{BASE_URL}/api/v3/coins/hyperliquid/status_updates"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        updates = data.get("status_updates", [])[:5]
        return [
            {
                "source": "CoinGecko",
                "title": u.get("description", "")[:120],
                "url": u.get("user", ""),
                "date": u.get("created_at", ""),
            }
            for u in updates
        ]
    except Exception:
        return []


def get_hype_news() -> Dict:
    """Get all HYPE news and info."""
    rss = fetch_rss_news()
    cg_news = fetch_coingecko_news()
    all_news = sorted(rss + cg_news, key=lambda x: x.get("date", ""), reverse=True)[:10]

    return {
        "news": all_news,
        "coin_info": fetch_coin_info(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
