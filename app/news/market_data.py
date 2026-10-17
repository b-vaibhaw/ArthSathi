"""
market_data.py
================
Financial news + market data tools, callable by the agent. Free sources only:
- RSS feeds (Economic Times, Moneycontrol, Reuters Business) via `feedparser`.
- Stock/index quotes via `yfinance` (no API key required).

Both dependencies are optional (commented out in requirements.txt by default —
uncomment to enable). If they're missing, or the network is unreachable (as in
this sandbox), these functions return a clear "unavailable" result instead of
raising, so the agent can tell the user honestly rather than crashing or
inventing numbers.

IMPORTANT: this is a tool-calling agent. The LLM must never state a price,
rate, or headline it wasn't given by one of these functions — numbers here
are the only numbers that are allowed to be "current."
"""

from typing import Dict, List

DEFAULT_FEEDS = {
    "india_business": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "india_markets": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "world_business": "https://feeds.reuters.com/reuters/businessNews",
}


def get_financial_news(topic: str = "", feed: str = "india_business", max_items: int = 5) -> Dict:
    """Fetch recent financial news headlines. `topic` filters headlines by
    a keyword match (case-insensitive substring); leave empty for the
    latest unfiltered headlines from the feed."""
    try:
        import feedparser
    except ImportError:
        return {
            "available": False,
            "reason": "feedparser not installed. Run: pip install feedparser, then uncomment it in requirements.txt.",
        }

    url = DEFAULT_FEEDS.get(feed, DEFAULT_FEEDS["india_business"])
    try:
        parsed = feedparser.parse(url)
    except Exception as e:
        return {"available": False, "reason": f"Could not fetch feed: {e}"}

    if getattr(parsed, "bozo", 0) and not parsed.entries:
        return {"available": False, "reason": "Feed fetch failed or returned no entries (check network access)."}

    items = []
    for entry in parsed.entries:
        title = getattr(entry, "title", "")
        if topic and topic.lower() not in title.lower():
            continue
        items.append({
            "title": title,
            "link": getattr(entry, "link", ""),
            "published": getattr(entry, "published", ""),
        })
        if len(items) >= max_items:
            break

    return {"available": True, "feed": feed, "topic": topic or None, "items": items}


def get_stock_quote(ticker: str) -> Dict:
    """Fetch a current/recent stock or index quote. `ticker` uses Yahoo
    Finance symbols, e.g. 'RELIANCE.NS', 'TCS.NS', 'AAPL', '^NSEI' (Nifty 50)."""
    try:
        import yfinance as yf
    except ImportError:
        return {
            "available": False,
            "reason": "yfinance not installed. Run: pip install yfinance, then uncomment it in requirements.txt.",
        }

    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)
        if price is None:
            return {"available": False, "reason": f"No price data found for ticker '{ticker}'."}
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else None
        return {
            "available": True, "ticker": ticker, "last_price": round(price, 2),
            "previous_close": round(prev_close, 2) if prev_close else None,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "currency": getattr(info, "currency", None),
        }
    except Exception as e:
        return {"available": False, "reason": f"Could not fetch quote for '{ticker}': {e}"}
