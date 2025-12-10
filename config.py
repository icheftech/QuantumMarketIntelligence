# config.py
"""
================================================================================
QUANTUM MARKET INTELLIGENCE PLATFORM - Configuration
================================================================================
Enterprise-grade configuration management for market intelligence system
"""

import os

# ============================================================================
# APPLICATION METADATA
# ============================================================================
APP_NAME = "Quantum Market Intelligence Platform"
APP_VERSION = "1.0.0"
COMPANY = "Southern Shade LLC"
COURSE = "ITAI-2372 - AI in Finance"

# ============================================================================
# API CONFIGURATION
# ============================================================================

# NewsAPI - For real-time financial news (optional)
# Get your free API key at: https://newsapi.org/register
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # leave blank for offline demo

# Alpha Vantage - For stock/forex data (optional, not used in MVP)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# OpenAI API (Optional - if you want to use cloud LLM instead of local)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Local LLM Configuration (LM Studio / Ollama)
LOCAL_LLM_ENABLED = True
LOCAL_LLM_URL = "http://localhost:11434/api/generate"  # Ollama default
LOCAL_LLM_MODEL = "llama2"  # or "mistral", "phi3", etc.

# ============================================================================
# DATA SOURCES
# ============================================================================

NEWS_SOURCES = [
    "bloomberg",
    "financial-times",
    "the-wall-street-journal",
    "reuters",
    "cnbc",
    "fortune",
    "business-insider",
]

FINANCE_KEYWORDS = [
    "stock market", "forex", "currency", "trading", "investment",
    "fed", "federal reserve", "interest rate", "inflation", "gdp",
    "earnings", "revenue", "profit", "loss", "merger", "acquisition",
    "eur/usd", "gbp/usd", "usd/jpy", "gold", "oil", "crypto", "bitcoin"
]

CURRENCY_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD"
]

# ============================================================================
# SENTIMENT & SIGNAL CONFIGURATION
# ============================================================================

SENTIMENT_THRESHOLDS = {
    "very_negative": -0.6,
    "negative": -0.2,
    "neutral": 0.2,
    "positive": 0.6,
    "very_positive": 1.0,
}

SIGNAL_THRESHOLDS = {
    "strong_sell": -0.7,
    "sell": -0.3,
    "hold": 0.3,
    "buy": 0.7,
    "strong_buy": 1.0,
}

# ============================================================================
# DATA STORAGE
# ============================================================================

DATA_DIR = "data"
MODELS_DIR = "models"
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")

for d in [DATA_DIR, MODELS_DIR, EXPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

MAX_NEWS_DISPLAY = 50
REFRESH_INTERVAL = 300  # seconds (5 minutes)

COLOR_POSITIVE = "#00C853"
COLOR_NEGATIVE = "#FF1744"
COLOR_NEUTRAL = "#FFB300"
COLOR_BUY = "#00C853"
COLOR_SELL = "#FF1744"

LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(DATA_DIR, "platform.log")

DISCLAIMER = """
⚠️ **IMPORTANT REGULATORY DISCLAIMER**

This system is for informational and educational purposes only.

**NOT FINANCIAL ADVICE:** This platform does not provide investment, financial,
legal, or tax advice. Outputs are informational only and must not be treated as
recommendations to buy, sell, or hold any security.

**RISK DISCLOSURE:** Trading and investing involve substantial risk of loss.
Past performance does not guarantee future results.

**HUMAN OVERSIGHT REQUIRED:** This system provides decision support only.
All final investment decisions must be made by qualified human professionals.
"""
