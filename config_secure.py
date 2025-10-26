"""Configuration - Use Streamlit Secrets for deployment"""

import streamlit as st

# Try to load from Streamlit secrets (cloud deployment)
# Fall back to local values for development
try:
    CLIENT_ID = st.secrets["angel_one"]["CLIENT_ID"]
    PASSWORD = st.secrets["angel_one"]["PASSWORD"]
    TOTP_SECRET = st.secrets["angel_one"]["TOTP_SECRET"]
    HISTORICAL_API_KEY = st.secrets["angel_one"]["HISTORICAL_API_KEY"]
    TRADING_API_KEY = st.secrets["angel_one"]["TRADING_API_KEY"]
    MARKET_API_KEY = st.secrets["angel_one"]["MARKET_API_KEY"]
except:
    # Development/Local credentials
    CLIENT_ID = "AAAG399109"
    PASSWORD = "1503"
    TOTP_SECRET = "OLRQ3CYBLPN2XWQPHLKMB7WEKI"
    HISTORICAL_API_KEY = "c3C0tMGn"
    TRADING_API_KEY = "ruseeaBq"
    MARKET_API_KEY = "PflRFXyd"

# Telegram Configuration
try:
    TELEGRAM_ENABLED = st.secrets["telegram"]["TELEGRAM_ENABLED"]
    TELEGRAM_CHAT_ID = st.secrets["telegram"]["TELEGRAM_CHAT_ID"]
    TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["TELEGRAM_BOT_TOKEN"]
except:
    # Development/Local Telegram settings
    TELEGRAM_ENABLED = "true"
    TELEGRAM_CHAT_ID = "-1002411670969"
    TELEGRAM_BOT_TOKEN = "7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"

# Supabase Configuration
try:
    SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]
except:
    # Development/Local Supabase settings (add your values here)
    SUPABASE_URL = ""
    SUPABASE_KEY = ""

# Trading Parameters (matching your Pine Script)
TRADING_CONFIG = {
    # EMA Settings
    'ema_fast': 9,
    'ema_medium': 21,
    'ema_slow': 50,
    
    # RSI Settings
    'rsi_length': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    
    # MACD Settings
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    
    # Volume Settings
    'volume_ma_length': 20,
    'volume_multiplier': 1.5,
    
    # ATR Settings
    'atr_length': 14,
    'atr_multiplier_sl': 2.5,  # Widened from 2.0
    'atr_multiplier_tp': 3.0,
    
    # ADX Settings
    'adx_length': 14,
    'min_adx': 20,
    
    # Supertrend Settings
    'supertrend_period': 10,
    'supertrend_multiplier': 3.0,
    
    # Signal Filters
    'min_signal_strength': 1,
    'min_trade_quality': 2,
    'cooldown_bars': 3,
    'avoid_choppy': False,
    
    # Session Timing (IST)
    'session_start_hour': 9,
    'session_start_min': 20,
    'session_end_hour': 15,
    'session_end_min': 15,
    'enable_session_filter': True,
    
    # Risk Management
    'trailing_stop': True,
    'initial_capital': 100000,
    'risk_per_trade': 2,  # % of capital
}

# Stock List (matching your backtest)
STOCK_SYMBOLS = {
    'TATASTEEL': {'token': '3499', 'exchange': 'NSE', 'lot_size': 1},
    'RELIANCE': {'token': '2885', 'exchange': 'NSE', 'lot_size': 1},
    'TCS': {'token': '11536', 'exchange': 'NSE', 'lot_size': 1},
    'SUNPHARMA': {'token': '3351', 'exchange': 'NSE', 'lot_size': 1},
    'ITC': {'token': '1660', 'exchange': 'NSE', 'lot_size': 1},
    'ICICIBANK': {'token': '1330', 'exchange': 'NSE', 'lot_size': 1},
    'TATAMOTORS': {'token': '3456', 'exchange': 'NSE', 'lot_size': 1},
}

# Timeframe mapping
TIMEFRAME_MAP = {
    '1min': 'ONE_MINUTE',
    '5min': 'FIVE_MINUTE',
    '15min': 'FIFTEEN_MINUTE',
    '30min': 'THIRTY_MINUTE',
    '1hour': 'ONE_HOUR',
    '1day': 'ONE_DAY',
}
