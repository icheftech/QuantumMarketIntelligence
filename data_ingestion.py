# data_ingestion.py
"""
================================================================================
DATA INGESTION
================================================================================
Fetches / simulates financial news for the platform
"""

import datetime as dt
from typing import List

import requests
import pandas as pd

import config


def fetch_news_from_api(query: str = "markets", page_size: int = 20) -> pd.DataFrame:
    """
    Uses NewsAPI if API key is configured.
    Falls back to empty DataFrame if not available.
    """
    if not config.NEWS_API_KEY:
        return pd.DataFrame()

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": config.NEWS_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    articles = data.get("articles", [])
    rows = []
    for a in articles:
        rows.append(
            {
                "source": a.get("source", {}).get("name", ""),
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "content": a.get("content", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            }
        )
    return pd.DataFrame(rows)


def sample_offline_articles() -> pd.DataFrame:
    """Offline sample articles for demo / assignment."""
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    data = [
        {
            "source": "SampleWire",
            "title": "Tech stocks rally on strong AI earnings",
            "description": "Mega-cap tech leads market higher as AI demand boosts profits.",
            "content": "Shares of major technology firms rallied today after several companies "
            "reported stronger-than-expected earnings tied to artificial intelligence demand.",
            "url": "",
            "published_at": today,
        },
        {
            "source": "MacroWatch",
            "title": "Recession fears rise as manufacturing contracts",
            "description": "Weak manufacturing and rising unemployment weigh on risk assets.",
            "content": "Global markets fell as fresh manufacturing data showed contraction, "
            "fueling concerns that a broader recession may be forming.",
            "url": "",
            "published_at": today,
        },
        {
            "source": "FXDesk",
            "title": "Dollar gains as Fed signals higher for longer",
            "description": "Currency markets react to updated rate guidance from the Fed.",
            "content": "The U.S. dollar strengthened against major peers after the Federal Reserve "
            "signaled that interest rates may remain elevated for an extended period.",
            "url": "",
            "published_at": today,
        },
    ]
    return pd.DataFrame(data)


def get_financial_news(query: str = "markets", page_size: int = 20) -> pd.DataFrame:
    """
    Try NewsAPI first; if no key or failure, use offline samples.
    """
    try:
        df = fetch_news_from_api(query, page_size)
        if not df.empty:
            return df
    except Exception:
        pass
    return sample_offline_articles()
