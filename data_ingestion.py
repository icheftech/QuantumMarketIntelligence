# data_ingestion.py
"""
========================================================================
DATA INGESTION
========================================================================
Fetches real financial news from RSS feeds
"""

import datetime as dt
from typing import List
import xml.etree.ElementTree as ET

import requests
import pandas as pd

import config


def fetch_news_from_rss(query: str = "markets", page_size: int = 20) -> pd.DataFrame:
    """
    Fetches real news from financial RSS feeds.
    No API key required - uses public RSS feeds.
    """
    
    # Financial RSS feeds (free, no API key needed)
    rss_feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Markets
        "https://www.cnbc.com/id/15839135/device/rss/rss.html",  # Stocks
        "https://www.marketwatch.com/rss/topstories",
    ]
    
    rows = []
    articles_collected = 0
    
    for feed_url in rss_feeds:
        if articles_collected >= page_size:
            break
            
        try:
            resp = requests.get(feed_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
            })
            resp.raise_for_status()
            
            # Parse RSS XML
            root = ET.fromstring(resp.content)
            
            # Handle different RSS formats
            items = root.findall('.//item')
            if not items:
                items = root.findall('.//entry')  # Atom format
            
            for item in items:
                if articles_collected >= page_size:
                    break
                
                # Extract title
                title_elem = item.find('title')
                title = title_elem.text if title_elem is not None else ""
                
                # Extract description/content
                desc_elem = item.find('description')
                if desc_elem is None:
                    desc_elem = item.find('summary')  # Atom format
                if desc_elem is None:
                    desc_elem = item.find('content')  # Alternative
                description = desc_elem.text if desc_elem is not None else ""
                
                # Extract link
                link_elem = item.find('link')
                if link_elem is not None:
                    link = link_elem.text if link_elem.text else link_elem.get('href', '')
                else:
                    link = ""
                
                # Extract publish date
                pubdate_elem = item.find('pubDate')
                if pubdate_elem is None:
                    pubdate_elem = item.find('published')  # Atom format
                pubdate = pubdate_elem.text if pubdate_elem is not None else str(dt.datetime.now())
                
                # Get source name from URL
                source = "Unknown"
                if "yahoo" in feed_url:
                    source = "Yahoo Finance"
                elif "cnbc" in feed_url:
                    source = "CNBC"
                elif "marketwatch" in feed_url:
                    source = "MarketWatch"
                
                # Filter by query if needed
                if query.lower() in title.lower() or query.lower() in description.lower():
                    rows.append({
                        "source": source,
                        "title": title,
                        "description": description,
                        "content": description,  # Use description as content
                        "url": link,
                        "published_at": pubdate,
                    })
                    articles_collected += 1
                elif query == "markets":  # Default: include all for markets
                    rows.append({
                        "source": source,
                        "title": title,
                        "description": description,
                        "content": description,
                        "url": link,
                        "published_at": pubdate,
                    })
                    articles_collected += 1
                    
        except Exception as e:
            print(f"Error fetching from {feed_url}: {e}")
            continue
    
    # If we didn't get enough articles, fall back to sample data
    if len(rows) < 3:
        print("Warning: Could not fetch enough real news. Using sample data.")
        return sample_offline_articles()
    
    return pd.DataFrame(rows)


def sample_offline_articles() -> pd.DataFrame:
    """Offline sample articles for demo / assignment."""
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    data = [
        {
            "source": "SampleWire",
            "title": "Tech stocks rally on strong AI earnings",
            "description": "Mega-cap tech leads market higher as AI demand boosts profits.",
            "content": "Shares of major technology firms rallied today after several companies reported stronger-than-expected earnings tied to artificial intelligence demand.",
            "url": "",
            "published_at": today,
        },
        {
            "source": "MacroWatch",
            "title": "Recession fears rise as manufacturing contracts",
            "description": "Weak manufacturing and rising unemployment weigh on risk assets.",
            "content": "Global markets fell as fresh manufacturing data showed contraction, fueling concerns that a broader recession may be forming.",
            "url": "",
            "published_at": today,
        },
        {
            "source": "FXDesk",
            "title": "Dollar gains as Fed signals higher for longer",
            "description": "Federal Reserve officials indicate rates may stay elevated.",
            "content": "The dollar rose against major currencies as Federal Reserve officials signaled that interest rates may remain higher for longer than markets expected.",
            "url": "",
            "published_at": today,
        },
    ]
    return pd.DataFrame(data)
