"""
Configuration Template for Angel One Trading Dashboard

INSTRUCTIONS:
1. Copy this file and rename to: config.py
2. Replace all placeholder values with your actual Angel One credentials
3. Never commit config.py to Git (it's in .gitignore)

For Streamlit Cloud deployment, use config_secure.py which reads from st.secrets
"""

# ===========================
# Angel One Credentials
# ===========================
# Get these from: https://smartapi.angelbroking.com/
CLIENT_ID = "YOUR_CLIENT_ID"  # Example: "A12345678"
PASSWORD = "YOUR_PASSWORD"     # Your Angel One login password
TOTP_SECRET = "YOUR_TOTP_SECRET"  # From 2FA setup

# API Keys - Generate from Angel One Developer Portal
API_KEY = "YOUR_API_KEY"
CORRELATION_ID = "YOUR_CORRELATION_ID"  
SOURCE_ID = "YOUR_SOURCE_ID"

# ===========================
# Trading Parameters
# ===========================
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
    
    # ATR Settings (Risk Management)
    'atr_length': 14,
    'atr_multiplier_sl': 2.5,  # Stop Loss = 2.5x ATR
    'atr_multiplier_tp': 3.0,  # Take Profit = 3x ATR
    
    # Signal Filters
    'min_signal_strength': 1,  # 0-2 scale
    'min_trade_quality': 2,    # 0-4 scale
    'min_adx': 20,            # Trend strength threshold
}

# ===========================
# Stock Symbols with NSE Tokens
# ===========================
# Get tokens from: https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
STOCK_SYMBOLS = {
    'TATASTEEL': {'token': '3499', 'exchange': 'NSE', 'lot_size': 1},
    'RELIANCE': {'token': '2885', 'exchange': 'NSE', 'lot_size': 1},
    'TCS': {'token': '11536', 'exchange': 'NSE', 'lot_size': 1},
    'SUNPHARMA': {'token': '3351', 'exchange': 'NSE', 'lot_size': 1},
    'ITC': {'token': '1660', 'exchange': 'NSE', 'lot_size': 1},
    'ICICIBANK': {'token': '1330', 'exchange': 'NSE', 'lot_size': 1},
    'TATAMOTORS': {'token': '3456', 'exchange': 'NSE', 'lot_size': 1},
}

# ===========================
# Timeframe Mapping
# ===========================
TIMEFRAME_MAP = {
    '1min': 'ONE_MINUTE',
    '5min': 'FIVE_MINUTE',
    '15min': 'FIFTEEN_MINUTE',
    '30min': 'THIRTY_MINUTE',
    '1hour': 'ONE_HOUR',
    '1day': 'ONE_DAY',
}
