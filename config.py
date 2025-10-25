import os
from dotenv import load_dotenv

load_dotenv()

# SmartAPI Rate Limiting (from official documentation)
SMARTAPI_RATE_LIMITS = {
    "requests_per_second": 3,
    "requests_per_minute": 180,
    "requests_per_hour": 5000,
    "min_delay_between_requests": 0.35  # ~333ms to stay under 3 req/sec
}

# API Endpoints
ENDPOINTS = {
    "market_breadth": "https://brkpoint.in/api/market-stats",
    "sector_performance": "https://brkpoint.in/api/sector-indices-performance",
    "support_resistance": "https://mo.streak.tech/api/sr_analysis_multi/",
    "candlestick_data": "https://technicalwidget.streak.tech/api/candles/",  # Base URL only
    "shareholdings": "https://zerodha.com/markets/stocks/NSE/{}/shareholdings/",  # Replace {} with symbol
    "financials": "https://zerodha.com/markets/stocks/NSE/{}/financials/",  # Replace {} with symbol
    "technical_analysis": "https://technicalwidget.streak.tech/api/streak_tech_analysis/",
    "telegram": f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage"
}

# Cache Settings
CACHE_DURATION = {
    "market_breadth": 900,  # 15 minutes
    "sector_performance": 900,  # 15 minutes
    "support_resistance": 0,  # No cache
    "candlestick_data": 0,  # No cache
    "shareholdings": 86400,  # 24 hours
    "financials": 86400,  # 24 hours
    "technical_analysis": 300  # 5 minutes
}

# Trading Parameters
TRADING_CONFIG = {
    "swing": {
        "timeframe": "day",
        "min_score": 50,  # Lowered from 70 to 50 for testing
        "risk_reward_ratio": 1.0,  # Lowered from 2.0 to 1.0 for testing
        "max_positions": 5
    },
    "intraday": {
        "timeframe": "5min",
        "min_score": 50,  # Lowered from 65 to 50 for testing
        "risk_reward_ratio": 1.0,  # Lowered from 1.5 to 1.0 for testing
        "max_positions": 3
    }
}

# Technical Indicators Settings
TECHNICAL_CONFIG = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_period": 14,
    "adx_strong_trend": 25,
    "volume_threshold": 1.5  # 1.5x average volume
}

# Environment Variables
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# User Agent Pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
]

# Flask Configuration
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-in-production")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# Redis Cache Configuration (optional)
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "stockgenie:")
