import pandas as pd
import ta
import logging
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from tqdm import tqdm
import plotly.express as px
import time
import requests
import io
import random
import spacy
from pytrends.request import TrendReq
import numpy as np
import itertools
from arch import arch_model
import warnings
import sqlite3
from diskcache import Cache
from SmartApi import SmartConnect
import pyotp
import os
from dotenv import load_dotenv
from scipy.stats.mstats import winsorize
from streamlit import cache_data
from itertools import cycle
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import hashlib
from dataclasses import dataclass

load_dotenv()

# ========================= CONFIGURATION =========================

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    DEFAULT_RISK_PER_TRADE: float = 0.02
    MAX_POSITION_SIZE_PCT: float = 0.25
    DEFAULT_STOP_LOSS_ATR_MULT: float = 2.0
    CACHE_TTL: int = 86400
    MAX_WORKERS: int = 3
    BATCH_SIZE: int = 10
    INITIAL_CAPITAL: float = 30000
    COMMISSION: float = 0.001
    
config = TradingConfig()

# ========================= LOGGING SETUP =========================

def setup_logging():
    """Setup comprehensive logging with rotation"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

# ========================= CUSTOM EXCEPTIONS =========================

class StockAnalysisError(Exception):
    """Custom exception for stock analysis errors"""
    pass

class DataFetchError(StockAnalysisError):
    """Exception for data fetching errors"""
    pass

class ValidationError(StockAnalysisError):
    """Exception for data validation errors"""
    pass

# ========================= RATE LIMITER =========================

class RateLimiter:
    """Thread-safe rate limiter"""
    def __init__(self, calls: int = 10, period: int = 60):
        self.calls = calls
        self.period = period
        self.call_times = deque()
        self._lock = None
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import threading
            if self._lock is None:
                self._lock = threading.Lock()
            
            with self._lock:
                now = time.time()
                
                # Remove old calls
                while self.call_times and self.call_times[0] < now - self.period:
                    self.call_times.popleft()
                
                if len(self.call_times) >= self.calls:
                    sleep_time = self.period - (now - self.call_times[0]) + 0.1
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                self.call_times.append(now)
            
            return func(*args, **kwargs)
        return wrapper

# ========================= CONSTANTS =========================

CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
API_KEYS = {
    "Historical": "c3C0tMGn",
    "Trading": os.getenv("TRADING_API_KEY"),
    "Market": os.getenv("MARKET_API_KEY")
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
]

# Initialize cache with better key generation
cache = Cache("stock_data_cache", size_limit=int(1e9))  # 1GB cache limit

TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "Ichimoku": "Ichimoku Cloud - Comprehensive trend indicator",
    "CMF": "Chaikin Money Flow - Buying/selling pressure",
    "Donchian": "Donchian Channels - Breakout detection",
    "Score": "Composite signal strength (-10 to +10). Higher = stronger buy signal."
}

SECTORS = {
    "Bank": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS"
    ],
    "Software & IT Services": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
        "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "CYIENT.NS"
    ],
    "Finance": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS",
        "SBICARD.NS", "M&MFIN.NS", "MUTHOOTFIN.NS", "LICHSGFIN.NS"
    ],
    "Automobile & Ancillaries": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "EICHERMOT.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "MRF.NS", "BALKRISIND.NS"
    ],
    "Healthcare": [
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "LUPIN.NS",
        "DIVISLAB.NS", "AUROPHARMA.NS", "ALKEM.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS"
    ],
    "Metals & Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS",
        "NMDC.NS", "HINDZINC.NS", "NALCO.NS", "JINDALSTEL.NS", "COALINDIA.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
        "GODREJCP.NS", "DABUR.NS", "COLPAL.NS", "MARICO.NS", "PGHH.NS"
    ],
    "Power": [
        "NTPC.NS", "POWERGRID.NS", "ADANIPOWER.NS", "TATAPOWER.NS", "JSWENERGY.NS",
        "NHPC.NS", "SJVN.NS", "TORNTPOWER.NS", "CESC.NS", "ADANIENSOL.NS"
    ],
    "Oil & Gas": [
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HPCL.NS", "GAIL.NS",
        "PETRONET.NS", "OIL.NS", "IGL.NS", "MGL.NS"
    ],
    "Real Estate": [
        "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS",
        "BRIGADE.NS", "SOBHA.NS", "SUNTECK.NS", "MAHLIFE.NS"
    ]
}

# ========================= DATA FETCHING =========================

@st.cache_data(ttl=86400)
def load_symbol_token_map() -> Dict[str, str]:
    """Load symbol to token mapping with enhanced error handling"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
    except requests.RequestException as e:
        logger.error(f"Failed to load symbol token map: {e}")
        st.warning(f"⚠️ Failed to load instrument list: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading symbol token map: {e}")
        return {}

def init_smartapi_client() -> Optional[SmartConnect]:
    """Initialize SmartAPI client with comprehensive error handling"""
    try:
        smart_api = SmartConnect(api_key=API_KEYS["Historical"])
        totp = pyotp.TOTP(TOTP_SECRET)
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
        
        if data['status']:
            logger.info("SmartAPI client initialized successfully")
            return smart_api
        else:
            error_msg = f"SmartAPI authentication failed: {data.get('message', 'Unknown error')}"
            logger.error(error_msg)
            st.error(f"⚠️ {error_msg}")
            return None
    except Exception as e:
        error_msg = f"Error initializing SmartAPI: {str(e)}"
        logger.error(error_msg)
        st.error(f"⚠️ {error_msg}")
        return None

def generate_cache_key(symbol: str, period: str, interval: str) -> str:
    """Generate a unique cache key using hashing for better performance"""
    key_string = f"{symbol}_{period}_{interval}_{datetime.now().date()}"
    return hashlib.md5(key_string.encode()).hexdigest()

@RateLimiter(calls=5, period=60)
def fetch_nse_stock_list() -> List[str]:
    """Fetch NSE stock list with rate limiting"""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        response = session.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}-EQ" for symbol in nse_data['SYMBOL']]
        logger.info(f"Fetched {len(stock_list)} stocks from NSE")
        return stock_list
    except Exception as e:
        logger.warning(f"Failed to fetch NSE stock list: {e}, using fallback")
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

@RateLimiter(calls=5, period=60)
def fetch_stock_data_with_auth(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch stock data with enhanced caching and error handling"""
    cache_key = generate_cache_key(symbol, period, interval)
    
    # Try to get from cache
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        logger.debug(f"Cache hit for {symbol}")
        return pd.read_pickle(io.BytesIO(cached_data))

    try:
        # Ensure symbol format
        if "-EQ" not in symbol:
            symbol = f"{symbol.split('.')[0]}-EQ"

        smart_api = init_smartapi_client()
        if not smart_api:
            raise DataFetchError("SmartAPI client initialization failed")

        # Calculate date range
        end_date = datetime.now()
        period_map = {
            "2y": timedelta(days=2 * 365),
            "1y": timedelta(days=365),
            "1mo": timedelta(days=30),
            "1w": timedelta(days=7)
        }
        start_date = end_date - period_map.get(period, timedelta(days=365))

        # Map intervals
        interval_map = {
            "1d": "ONE_DAY",
            "1h": "ONE_HOUR",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE"
        }
        api_interval = interval_map.get(interval, "ONE_DAY")

        # Get symbol token
        symbol_token_map = load_symbol_token_map()
        symboltoken = symbol_token_map.get(symbol)
        if not symboltoken:
            logger.warning(f"Token not found for symbol: {symbol}")
            return pd.DataFrame()

        # Fetch historical data
        historical_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": symboltoken,
            "interval": api_interval,
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        })

        if historical_data['status'] and historical_data['data']:
            data = pd.DataFrame(
                historical_data['data'], 
                columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Validate data quality
            is_valid, issues = validate_data_quality(data)
            if not is_valid:
                logger.warning(f"Data quality issues for {symbol}: {issues}")
            
            # Cache the data
            buffer = io.BytesIO()
            data.to_pickle(buffer)
            cache.set(cache_key, buffer.getvalue(), expire=config.CACHE_TTL)
            
            logger.info(f"Successfully fetched data for {symbol}: {len(data)} rows")
            return data
        else:
            error_msg = historical_data.get('message', 'Unknown error')
            raise DataFetchError(f"No data found for {symbol}: {error_msg}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {symbol}")
            st.warning(f"⚠️ Rate limit exceeded for {symbol}. Please wait before retrying.")
        else:
            logger.error(f"HTTP error for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        st.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=1000)
def fetch_stock_data_cached(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """LRU cached wrapper for stock data fetching"""
    return fetch_stock_data_with_auth(symbol, period, interval)

# ========================= DATA VALIDATION =========================

def validate_data_quality(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Enhanced data validation with detailed issue reporting"""
    issues = []
    
    if data is None or data.empty:
        issues.append("Empty dataframe")
        return False, issues
    
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")
    
    # Check for null values
    null_counts = data[required_columns].isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check for data anomalies
    if 'High' in data.columns and 'Low' in data.columns:
        invalid_hl = (data['High'] < data['Low']).sum()
        if invalid_hl > 0:
            issues.append(f"High < Low in {invalid_hl} rows")
    
    # Check for negative prices
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        if col in data.columns:
            negative_count = (data[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"Negative values in {col}: {negative_count} rows")
    
    # Check for extreme outliers
    if 'Close' in data.columns:
        returns = data['Close'].pct_change().dropna()
        extreme_returns = returns[abs(returns) > 0.5]  # 50% daily move
        if len(extreme_returns) > 0:
            issues.append(f"Extreme price moves detected: {len(extreme_returns)} instances")
    
    # Check for data gaps
    if isinstance(data.index, pd.DatetimeIndex):
        time_diff = data.index.to_series().diff()
        max_gap = time_diff.max()
        if max_gap > pd.Timedelta(days=5):
            issues.append(f"Large time gap detected: {max_gap}")
    
    return len(issues) == 0, issues

# ========================= TECHNICAL INDICATORS =========================

_indicator_cache = {}

def compute_indicators(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Optimized indicator computation with caching"""
    if df.empty or 'Close' not in df.columns:
        return df
    
    # Generate cache key
    cache_key = f"{symbol}_{len(df)}_{df.index[-1] if not df.empty else 'empty'}"
    if cache_key in _indicator_cache:
        return _indicator_cache[cache_key]
    
    df = df.copy()
    
    try:
        # Trend indicators - Vectorized operations
        windows = [20, 50, 200]
        for w in windows:
            if len(df) >= w:
                df[f'SMA_{w}'] = df['Close'].rolling(w, min_periods=1).mean()
            else:
                df[f'SMA_{w}'] = np.nan
        
        # RSI with optimized window
        if len(df) >= 14:
            optimal_rsi_window = optimize_rsi_window(df)
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=optimal_rsi_window).rsi()
        
        # MACD
        if len(df) >= 26:
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
        
        # Ichimoku - only if enough data
        if len(df) >= 52:
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            df['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
            df['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
            df['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
            df['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
            df['Ichimoku_Chikou'] = df['Close'].shift(-26)
        
        # Volume indicators
        if 'Volume' in df.columns and len(df) >= 20:
            df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
                df['High'], df['Low'], df['Close'], df['Volume'], window=20
            ).chaikin_money_flow()
            df['Avg_Volume'] = df['Volume'].rolling(20).mean()
            df['Volume_Spike'] = df['Volume'] > (df['Avg_Volume'] * 1.5)
        
        # Volatility indicators
        if len(df) >= 14:
            df['ATR'] = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close'], window=14
            ).average_true_range()
            df['ADX'] = ta.trend.ADXIndicator(
                df['High'], df['Low'], df['Close'], window=14
            ).adx()
        
        # Donchian Channels
        if len(df) >= 20:
            donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20)
            df['Donchian_Upper'] = donchian.donchian_channel_hband()
            df['Donchian_Lower'] = donchian.donchian_channel_lband()
            df['Donchian_Middle'] = donchian.donchian_channel_mband()
        
        # Bollinger Bands
        if len(df) >= 20:
            bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['Upper_Band'] = bb.bollinger_hband()
            df['Middle_Band'] = bb.bollinger_mavg()
            df['Lower_Band'] = bb.bollinger_lband()
        
        # Stochastic Oscillator
        if len(df) >= 14:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['SlowK'] = stoch.stoch()
            df['SlowD'] = stoch.stoch_signal()
            df['Stoch_K'] = df['SlowK']  # Alias for compatibility
            df['Stoch_D'] = df['SlowD']
        
        # VWAP calculation
        if 'Volume' in df.columns:
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Williams %R for additional momentum
        if len(df) >= 14:
            df['WilliamsR'] = ta.momentum.WilliamsRIndicator(
                df['High'], df['Low'], df['Close'], lbp=14
            ).williams_r()
        
        # Cache the result
        _indicator_cache[cache_key] = df
        
    except Exception as e:
        logger.error(f"Error computing indicators: {e}")
    
    return df

def optimize_rsi_window(data: pd.DataFrame, windows: range = range(5, 15), 
                        risk_free_rate: float = 0.025) -> int:
    """Optimize RSI window using Sharpe ratio with better error handling"""
    best_window, best_sharpe = 9, -float('inf')
    
    if data is None or data.empty or 'Close' not in data.columns:
        return best_window
    
    if len(data) < max(windows) + 20:
        return best_window
    
    try:
        returns = data['Close'].pct_change().dropna()
        
        for window in windows:
            if len(data) < window + 10:
                continue
                
            rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
            
            # Generate signals
            signals = pd.Series(0, index=rsi.index)
            signals[rsi < 30] = 1  # Buy signal
            signals[rsi > 70] = -1  # Sell signal
            
            # Calculate positions
            positions = signals.replace(0, np.nan).ffill().fillna(0).shift(1)
            
            # Calculate strategy returns
            aligned_index = returns.index.intersection(positions.index)
            if len(aligned_index) < 20:
                continue
                
            strategy_returns = returns.loc[aligned_index] * positions.loc[aligned_index]
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) > 0 and strategy_returns.std() != 0:
                sharpe = ((strategy_returns.mean() - risk_free_rate / 252) / 
                         strategy_returns.std()) * np.sqrt(252)
                
                if sharpe > best_sharpe:
                    best_sharpe, best_window = sharpe, window
    
    except Exception as e:
        logger.warning(f"Error optimizing RSI window: {e}")
    
    return best_window

# ========================= MARKET ANALYSIS =========================

def classify_market_regime(df: pd.DataFrame) -> str:
    """Enhanced market regime classification"""
    try:
        if len(df) < 50:
            return "Insufficient Data"
        
        sma20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else df['Close'].rolling(20).mean().iloc[-1]
        sma50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['SMA_200'].iloc[-1] if 'SMA_200' in df.columns and len(df) >= 200 else None
        close = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else None
        
        # Calculate trend slope
        slope_period = min(5, len(df) - 1)
        slope = (sma20 - df['SMA_20'].iloc[-slope_period] if 'SMA_20' in df.columns else 0) / slope_period
        
        # Volatility assessment
        atr_pct = (atr / close) if atr and close else 0
        
        # ADX for trend strength
        adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else None
        
        # Regime classification with multiple factors
        if atr_pct > 0.04:
            return "High Volatility"
        elif adx and adx > 40:
            if sma20 > sma50:
                return "Strong Uptrend"
            else:
                return "Strong Downtrend"
        elif sma200 and sma20 > sma50 > sma200 and slope > 0.001:
            return "Bullish"
        elif sma200 and sma20 < sma50 < sma200 and slope < -0.001:
            return "Bearish"
        elif abs(slope) < 0.0005:
            return "Ranging"
        elif sma20 > sma50:
            return "Mild Bullish"
        elif sma20 < sma50:
            return "Mild Bearish"
        else:
            return "Neutral"
            
    except Exception as e:
        logger.error(f"Error classifying market regime: {e}")
        return "Unknown"

def detect_divergence(data: pd.DataFrame, window: int = 14, 
                     rsi_threshold: float = 5) -> str:
    """Enhanced divergence detection with better validation"""
    try:
        if 'RSI' not in data.columns or 'Close' not in data.columns:
            return "Insufficient Data"
        
        price = data['Close']
        rsi = data['RSI']
        
        if len(price) < window or len(rsi) < window:
            return "Insufficient Data"
        
        # Find recent peaks and troughs
        price_window = price.iloc[-window:]
        rsi_window = rsi.iloc[-window:]
        
        # Price extremes
        price_peaks = price_window[price_window == price_window.rolling(3, center=True).max()]
        price_troughs = price_window[price_window == price_window.rolling(3, center=True).min()]
        
        # RSI extremes
        rsi_peaks = rsi_window[rsi_window == rsi_window.rolling(3, center=True).max()]
        rsi_troughs = rsi_window[rsi_window == rsi_window.rolling(3, center=True).min()]
        
        # Check for bullish divergence (price lower low, RSI higher low)
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if (price_troughs.iloc[-1] < price_troughs.iloc[-2] and 
                rsi_troughs.iloc[-1] > rsi_troughs.iloc[-2] and
                abs(rsi_troughs.iloc[-1] - rsi_troughs.iloc[-2]) > rsi_threshold):
                return "Bullish Divergence"
        
        # Check for bearish divergence (price higher high, RSI lower high)
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if (price_peaks.iloc[-1] > price_peaks.iloc[-2] and 
                rsi_peaks.iloc[-1] < rsi_peaks.iloc[-2] and
                abs(rsi_peaks.iloc[-1] - rsi_peaks.iloc[-2]) > rsi_threshold):
                return "Bearish Divergence"
        
        return "No Divergence"
        
    except Exception as e:
        logger.error(f"Error detecting divergence: {e}")
        return "Error"

# ========================= SIGNAL SCORING =========================

def compute_signal_score(df: pd.DataFrame, regime: str) -> float:
    """Enhanced signal scoring with regime-specific weights"""
    
    # Dynamic weights based on regime
    regime_weights = {
        "Strong Uptrend": {"trend": 2.0, "momentum": 1.5, "volatility": 0.5},
        "Bullish": {"trend": 1.5, "momentum": 1.5, "volatility": 0.8},
        "Neutral": {"trend": 0.8, "momentum": 1.2, "volatility": 1.0},
        "Ranging": {"trend": 0.5, "momentum": 1.0, "volatility": 1.5},
        "Bearish": {"trend": 1.5, "momentum": 1.5, "volatility": 0.8},
        "Strong Downtrend": {"trend": 2.0, "momentum": 1.5, "volatility": 0.5},
        "High Volatility": {"trend": 0.5, "momentum": 0.8, "volatility": 2.0}
    }
    
    weights = regime_weights.get(regime, {"trend": 1.0, "momentum": 1.0, "volatility": 1.0})
    
    score = 0
    signals_count = 0
    
    try:
        close = df['Close'].iloc[-1]
        
        # RSI Signal (Momentum)
        if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]):
            rsi = df['RSI'].iloc[-1]
            signals_count += 1
            
            if rsi < 30:
                score += 2.0 * weights["momentum"]
            elif rsi < 40:
                score += 1.0 * weights["momentum"]
            elif rsi > 70:
                score -= 2.0 * weights["momentum"]
            elif rsi > 60:
                score -= 1.0 * weights["momentum"]
        
        # MACD Signal (Trend)
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            
            if pd.notna(macd) and pd.notna(macd_signal):
                signals_count += 1
                macd_diff = macd - macd_signal
                
                # Check for crossover
                prev_macd = df['MACD'].iloc[-2]
                prev_signal = df['MACD_signal'].iloc[-2]
                
                if macd > macd_signal and prev_macd <= prev_signal:
                    score += 2.5 * weights["trend"]  # Bullish crossover
                elif macd < macd_signal and prev_macd >= prev_signal:
                    score -= 2.5 * weights["trend"]  # Bearish crossover
                elif macd_diff > 0:
                    score += 1.0 * weights["trend"]
                else:
                    score -= 1.0 * weights["trend"]
        
        # Ichimoku Cloud (Trend)
        if 'Ichimoku_Span_A' in df.columns and 'Ichimoku_Span_B' in df.columns:
            span_a = df['Ichimoku_Span_A'].iloc[-1]
            span_b = df['Ichimoku_Span_B'].iloc[-1]
            
            if pd.notna(span_a) and pd.notna(span_b):
                signals_count += 1
                cloud_top = max(span_a, span_b)
                cloud_bottom = min(span_a, span_b)
                
                if close > cloud_top:
                    score += 2.0 * weights["trend"]
                elif close < cloud_bottom:
                    score -= 2.0 * weights["trend"]
                else:
                    score += 0.5 * weights["trend"]  # Inside cloud
        
        # Bollinger Bands (Volatility)
        if 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
            upper = df['Upper_Band'].iloc[-1]
            lower = df['Lower_Band'].iloc[-1]
            
            if pd.notna(upper) and pd.notna(lower):
                signals_count += 1
                band_width = (upper - lower) / close
                
                if close < lower:
                    score += 1.5 * weights["volatility"]  # Oversold
                elif close > upper:
                    score -= 1.5 * weights["volatility"]  # Overbought
                
                # Band squeeze detection
                if band_width < 0.05:
                    score += 0.5  # Potential breakout
        
        # Volume Analysis
        if 'Volume' in df.columns and 'Avg_Volume' in df.columns:
            volume = df['Volume'].iloc[-1]
            avg_volume = df['Avg_Volume'].iloc[-1]
            
            if pd.notna(volume) and pd.notna(avg_volume) and avg_volume > 0:
                signals_count += 1
                volume_ratio = volume / avg_volume
                
                if volume_ratio > 1.5 and close > df['Close'].iloc[-2]:
                    score += 1.0  # High volume with price increase
                elif volume_ratio > 1.5 and close < df['Close'].iloc[-2]:
                    score -= 1.0  # High volume with price decrease
        
        # ADX for trend strength
        if 'ADX' in df.columns and pd.notna(df['ADX'].iloc[-1]):
            adx = df['ADX'].iloc[-1]
            signals_count += 1
            
            if adx > 25:
                score += 0.5 * weights["trend"]  # Strong trend
            elif adx < 15:
                score -= 0.5 * weights["trend"]  # Weak trend
        
        # Stochastic Oscillator
        if 'SlowK' in df.columns and 'SlowD' in df.columns:
            k = df['SlowK'].iloc[-1]
            d = df['SlowD'].iloc[-1]
            
            if pd.notna(k) and pd.notna(d):
                signals_count += 1
                
                if k < 20 and k > d:
                    score += 1.0 * weights["momentum"]  # Oversold with bullish crossover
                elif k > 80 and k < d:
                    score -= 1.0 * weights["momentum"]  # Overbought with bearish crossover
        
        # Normalize score based on number of signals
        if signals_count > 0:
            score = score / signals_count * 3  # Scale to reasonable range
        
    except Exception as e:
        logger.error(f"Error computing signal score: {e}")
        return 0
    
    return round(np.clip(score, -10, 10), 2)

# ========================= POSITION SIZING =========================

def calculate_position_size(account_size: float, risk_per_trade: float, 
                           entry_price: float, stop_loss: float,
                           max_position_pct: float = 0.25,
                           volatility_adjustment: bool = True,
                           atr: Optional[float] = None) -> int:
    """
    Enhanced position sizing with Kelly Criterion and volatility adjustment
    """
    try:
        # Basic position sizing based on risk
        risk_amount = account_size * risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        shares_by_risk = risk_amount / price_risk
        
        # Max position constraint
        max_position_value = account_size * max_position_pct
        shares_by_max = max_position_value / entry_price
        
        # Volatility adjustment
        if volatility_adjustment and atr:
            atr_pct = atr / entry_price
            if atr_pct > 0.03:  # High volatility
                shares_by_risk *= 0.7  # Reduce position size
            elif atr_pct < 0.01:  # Low volatility
                shares_by_risk *= 1.2  # Increase position size
        
        # Return integer number of shares
        return int(min(shares_by_risk, shares_by_max))
        
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0

# ========================= RECOMMENDATION ENGINE =========================

def adaptive_recommendation(df: pd.DataFrame, symbol: Optional[str] = None,
                           account_size: float = 30000,
                           max_position_size: int = 100) -> Dict[str, Any]:
    """Enhanced adaptive recommendation system"""
    try:
        # Compute indicators
        df = compute_indicators(df, symbol)
        
        if len(df) < 50:
            return {"Recommendation": "Hold", "Reason": "Insufficient data"}
        
        close = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else None
        volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else None
        avg_vol = df['Avg_Volume'].iloc[-1] if 'Avg_Volume' in df.columns else None
        
        # Classify market regime
        regime = classify_market_regime(df)
        
        # Compute signal score
        score = compute_signal_score(df, regime)
        
        # Check for divergence
        divergence = detect_divergence(df)
        
        # Adjust score based on divergence
        if divergence == "Bullish Divergence":
            score += 1.5
        elif divergence == "Bearish Divergence":
            score -= 1.5
        
        # Dynamic thresholds based on regime
        thresholds = {
            'High Volatility': 2.0,
            'Strong Uptrend': 0.5,
            'Bullish': 1.0,
            'Neutral': 1.5,
            'Ranging': 2.0,
            'Bearish': 1.0,
            'Strong Downtrend': 0.5
        }
        threshold = thresholds.get(regime, 1.5)
        
        # Generate recommendation
        if score > threshold:
            rec = "Buy"
            reason = f"Strong buy signals (score: {score:.1f})"
        elif score < -threshold:
            rec = "Sell"
            reason = f"Strong sell signals (score: {score:.1f})"
        else:
            rec = "Hold"
            reason = f"Neutral signals (score: {score:.1f})"
        
        # Calculate entry, stop loss, and target
        if rec == "Buy":
            buy_at = close * 1.002  # Slight premium for entry
            
            # Dynamic stop loss based on ATR and regime
            if atr:
                atr_multiplier = 2.0 if regime == "High Volatility" else 1.5
                stop_loss = close - (atr * atr_multiplier)
            else:
                stop_loss = close * 0.95
            
            # Target based on risk-reward ratio
            risk = close - stop_loss
            target = close + (risk * 3)  # 3:1 risk-reward
            
        elif rec == "Sell":
            buy_at = None
            stop_loss = close * 1.05
            target = close * 0.95
        else:
            buy_at = None
            stop_loss = None
            target = None
        
        # Calculate position size
        if rec in ["Buy", "Sell"] and stop_loss:
            position_size = calculate_position_size(
                account_size, 
                config.DEFAULT_RISK_PER_TRADE,
                close, 
                stop_loss,
                config.MAX_POSITION_SIZE_PCT,
                True,
                atr
            )
        else:
            position_size = 0
        
        # Calculate trailing stop
        if rec == "Buy" and atr:
            trailing_stop = close - (atr * 2.5)
        elif rec == "Sell" and atr:
            trailing_stop = close + (atr * 2.5)
        else:
            trailing_stop = None
        
        return {
            "Current Price": round(close, 2),
            "Buy At": round(buy_at, 2) if buy_at else None,
            "Stop Loss": round(stop_loss, 2) if stop_loss else None,
            "Target": round(target, 2) if target else None,
            "Recommendation": rec,
            "Score": score,
            "Regime": regime,
            "Divergence": divergence,
            "Position Size": position_size,
            "Trailing Stop": round(trailing_stop, 2) if trailing_stop else None,
            "Reason": reason,
            "Confidence": min(100, abs(score) * 10)  # Confidence percentage
        }
        
    except Exception as e:
        logger.error(f"Error in adaptive recommendation: {e}")
        return {"Recommendation": "Hold", "Reason": f"Error: {str(e)}"}

def generate_recommendations(data: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Standard recommendation generation with multiple timeframes"""
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold",
        "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }
    
    if data.empty or len(data) < 27:
        return recommendations
    
    try:
        # Ensure indicators are computed
        data = compute_indicators(data, symbol)
        
        recommendations["Current Price"] = float(data['Close'].iloc[-1])
        
        buy_score = 0
        sell_score = 0
        
        # RSI Analysis
        if 'RSI' in data.columns and pd.notna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            if rsi < 25:
                buy_score += 3
            elif rsi < 30:
                buy_score += 2
            elif rsi < 40:
                buy_score += 1
            elif rsi > 75:
                sell_score += 3
            elif rsi > 70:
                sell_score += 2
            elif rsi > 60:
                sell_score += 1
        
        # MACD Analysis
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_sig = data['MACD_signal'].iloc[-1]
            
            if pd.notna(macd) and pd.notna(macd_sig):
                # Check for crossover
                prev_macd = data['MACD'].iloc[-2] if len(data) > 1 else macd
                prev_signal = data['MACD_signal'].iloc[-2] if len(data) > 1 else macd_sig
                
                if macd > macd_sig and prev_macd <= prev_signal:
                    buy_score += 2  # Bullish crossover
                elif macd < macd_sig and prev_macd >= prev_signal:
                    sell_score += 2  # Bearish crossover
                elif macd > macd_sig:
                    buy_score += 1
                else:
                    sell_score += 1
        
        # Bollinger Bands
        if all(col in data.columns for col in ['Close', 'Lower_Band', 'Upper_Band']):
            close = data['Close'].iloc[-1]
            lower = data['Lower_Band'].iloc[-1]
            upper = data['Upper_Band'].iloc[-1]
            
            if pd.notna(lower) and pd.notna(upper):
                if close < lower:
                    buy_score += 2
                    recommendations["Mean_Reversion"] = "Buy"
                elif close > upper:
                    sell_score += 2
                    recommendations["Mean_Reversion"] = "Sell"
        
        # Volume Analysis
        if 'Volume' in data.columns and 'Avg_Volume' in data.columns:
            volume = data['Volume'].iloc[-1]
            avg_vol = data['Avg_Volume'].iloc[-1]
            
            if pd.notna(volume) and pd.notna(avg_vol) and avg_vol > 0:
                volume_ratio = volume / avg_vol
                close = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2]
                
                if volume_ratio > 1.5:
                    if close > prev_close:
                        buy_score += 2
                    else:
                        sell_score += 2
        
        # Ichimoku Analysis
        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns:
            span_a = data['Ichimoku_Span_A'].iloc[-1]
            span_b = data['Ichimoku_Span_B'].iloc[-1]
            close = data['Close'].iloc[-1]
            
            if pd.notna(span_a) and pd.notna(span_b):
                if close > max(span_a, span_b):
                    buy_score += 2
                    recommendations["Ichimoku_Trend"] = "Buy"
                elif close < min(span_a, span_b):
                    sell_score += 2
                    recommendations["Ichimoku_Trend"] = "Sell"
        
        # Donchian Breakout
        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns:
            upper = data['Donchian_Upper'].iloc[-1]
            lower = data['Donchian_Lower'].iloc[-1]
            close = data['Close'].iloc[-1]
            
            if pd.notna(upper) and pd.notna(lower):
                if close > upper:
                    buy_score += 2
                    recommendations["Breakout"] = "Buy"
                elif close < lower:
                    sell_score += 2
                    recommendations["Breakout"] = "Sell"
        
        # ADX for trend strength
        if 'ADX' in data.columns and pd.notna(data['ADX'].iloc[-1]):
            adx = data['ADX'].iloc[-1]
            if adx > 25:
                # Strong trend - enhance existing signals
                if buy_score > sell_score:
                    buy_score += 1
                elif sell_score > buy_score:
                    sell_score += 1
        
        # Generate timeframe recommendations
        net_score = buy_score - sell_score
        
        # Intraday (most sensitive)
        if net_score >= 3:
            recommendations["Intraday"] = "Strong Buy"
        elif net_score >= 2:
            recommendations["Intraday"] = "Buy"
        elif net_score <= -3:
            recommendations["Intraday"] = "Strong Sell"
        elif net_score <= -2:
            recommendations["Intraday"] = "Sell"
        
        # Swing (medium sensitivity)
        if net_score >= 4:
            recommendations["Swing"] = "Strong Buy"
        elif net_score >= 3:
            recommendations["Swing"] = "Buy"
        elif net_score <= -4:
            recommendations["Swing"] = "Strong Sell"
        elif net_score <= -3:
            recommendations["Swing"] = "Sell"
        
        # Short-term
        if net_score >= 5:
            recommendations["Short-Term"] = "Buy"
        elif net_score <= -5:
            recommendations["Short-Term"] = "Sell"
        
        # Long-term (least sensitive)
        if net_score >= 6:
            recommendations["Long-Term"] = "Buy"
        elif net_score <= -6:
            recommendations["Long-Term"] = "Sell"
        
        # Calculate entry, stop loss, and target
        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)
        recommendations["Score"] = min(max(net_score, -10), 10)
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
    
    return recommendations

# ========================= PRICE CALCULATIONS =========================

def calculate_buy_at(data: pd.DataFrame) -> Optional[float]:
    """Calculate optimal entry price"""
    try:
        if data.empty or 'Close' not in data.columns:
            return None
        
        last_close = data['Close'].iloc[-1]
        
        # Check RSI for entry timing
        if 'RSI' in data.columns and pd.notna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            if rsi < 30:
                # Oversold - can afford slight discount
                return round(last_close * 0.995, 2)
        
        # Check for support levels
        if 'Lower_Band' in data.columns and pd.notna(data['Lower_Band'].iloc[-1]):
            lower_band = data['Lower_Band'].iloc[-1]
            if last_close < lower_band * 1.02:
                # Near support - good entry
                return round(last_close, 2)
        
        # Default: slight premium for momentum
        return round(last_close * 1.002, 2)
        
    except Exception as e:
        logger.error(f"Error calculating buy price: {e}")
        return None

def calculate_stop_loss(data: pd.DataFrame) -> Optional[float]:
    """Calculate stop loss with multiple methods"""
    try:
        if data.empty or 'Close' not in data.columns:
            return None
        
        last_close = data['Close'].iloc[-1]
        stop_loss = last_close * 0.95  # Default 5% stop
        
        # ATR-based stop loss (preferred)
        if 'ATR' in data.columns and pd.notna(data['ATR'].iloc[-1]):
            atr = data['ATR'].iloc[-1]
            
            # Adjust multiplier based on ADX
            multiplier = config.DEFAULT_STOP_LOSS_ATR_MULT
            if 'ADX' in data.columns and pd.notna(data['ADX'].iloc[-1]):
                adx = data['ADX'].iloc[-1]
                if adx > 30:
                    multiplier = 2.5  # Wider stop for strong trends
                elif adx < 20:
                    multiplier = 1.5  # Tighter stop for weak trends
            
            stop_loss = last_close - (atr * multiplier)
        
        # Support-based stop loss
        if 'Lower_Band' in data.columns and pd.notna(data['Lower_Band'].iloc[-1]):
            lower_band = data['Lower_Band'].iloc[-1]
            stop_loss = max(stop_loss, lower_band * 0.98)
        
        # Recent swing low
        if len(data) >= 20:
            recent_low = data['Low'].iloc[-20:].min()
            stop_loss = max(stop_loss, recent_low * 0.99)
        
        # Ensure stop loss is not too tight
        min_stop = last_close * 0.9  # Maximum 10% stop
        stop_loss = max(stop_loss, min_stop)
        
        return round(stop_loss, 2)
        
    except Exception as e:
        logger.error(f"Error calculating stop loss: {e}")
        return None

def calculate_target(data: pd.DataFrame, risk_reward_ratio: float = 3.0) -> Optional[float]:
    """Calculate profit target with dynamic risk-reward"""
    try:
        stop_loss = calculate_stop_loss(data)
        if stop_loss is None:
            return None
        
        last_close = data['Close'].iloc[-1]
        risk = last_close - stop_loss
        
        # Adjust risk-reward based on market conditions
        if 'ADX' in data.columns and pd.notna(data['ADX'].iloc[-1]):
            adx = data['ADX'].iloc[-1]
            if adx > 30:
                risk_reward_ratio = 4.0  # Higher targets in strong trends
            elif adx < 20:
                risk_reward_ratio = 2.0  # Lower targets in weak trends
        
        # Resistance-based target
        target = last_close + (risk * risk_reward_ratio)
        
        if 'Upper_Band' in data.columns and pd.notna(data['Upper_Band'].iloc[-1]):
            upper_band = data['Upper_Band'].iloc[-1]
            # If calculated target is beyond upper band, adjust
            if target > upper_band * 1.05:
                target = upper_band * 1.02
        
        # Recent swing high as resistance
        if len(data) >= 20:
            recent_high = data['High'].iloc[-20:].max()
            if target > recent_high * 1.1:
                target = recent_high * 1.05
        
        # Cap at reasonable levels
        max_target = last_close * 1.2  # Maximum 20% target
        target = min(target, max_target)
        
        return round(target, 2)
        
    except Exception as e:
        logger.error(f"Error calculating target: {e}")
        return None

# ========================= BACKTESTING =========================

@st.cache_data(ttl=3600)
def backtest_stock(data: pd.DataFrame, symbol: str, strategy: str = "Swing", 
                   _data_hash: Optional[int] = None) -> Dict[str, Any]:
    """Enhanced backtesting with proper position management"""
    
    INITIAL_CAPITAL = config.INITIAL_CAPITAL
    commission = config.COMMISSION
    position_size_pct = st.session_state.get('position_size', 1.0)
    
    results = {
        "total_return": 0,
        "annual_return": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "trades": 0,
        "win_rate": 0,
        "profit_factor": 0,
        "average_win": 0,
        "average_loss": 0,
        "buy_signals": [],
        "sell_signals": [],
        "trade_details": [],
        "equity_curve": []
    }
    
    try:
        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
        
        portfolio_value = INITIAL_CAPITAL
        cash = INITIAL_CAPITAL
        position = None
        entry_price = 0
        entry_date = None
        trade_qty = 0
        trades = []
        returns = []
        equity_curve = []
        
        for i in range(50, len(data)):  # Start after enough data for indicators
            sliced_data = data.iloc[:i+1]
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            
            # Get recommendation
            if recommendation_mode == "Adaptive":
                rec = adaptive_recommendation(sliced_data, symbol)
                signal = rec.get("Recommendation", "Hold")
            else:
                rec = generate_recommendations(sliced_data, symbol)
                signal = rec.get(strategy, "Hold")
            
            # Simplified signal interpretation
            if "Buy" in str(signal) and position is None:
                # Execute buy
                entry_price = current_price
                entry_date = current_date
                allocation = cash * position_size_pct
                trade_qty = int(allocation // current_price)
                
                if trade_qty > 0:
                    cost = trade_qty * current_price * (1 + commission)
                    if cost <= cash:
                        cash -= cost
                        position = "Long"
                        results["buy_signals"].append((current_date, current_price))
            
            elif "Sell" in str(signal) and position == "Long":
                # Execute sell
                proceeds = trade_qty * current_price * (1 - commission)
                cash += proceeds
                
                # Calculate profit
                entry_cost = trade_qty * entry_price * (1 + commission)
                exit_proceeds = proceeds
                profit = exit_proceeds - entry_cost
                pct_return = profit / entry_cost if entry_cost > 0 else 0
                
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": current_date,
                    "exit_price": current_price,
                    "quantity": trade_qty,
                    "profit": profit,
                    "return": pct_return
                })
                
                returns.append(pct_return)
                results["sell_signals"].append((current_date, current_price))
                
                # Reset position
                position = None
                entry_price = 0
                entry_date = None
                trade_qty = 0
            
            # Update portfolio value
            if position == "Long":
                portfolio_value = cash + (trade_qty * current_price)
            else:
                portfolio_value = cash
            
            equity_curve.append((current_date, portfolio_value))
        
        # Close any open position
        if position == "Long" and trade_qty > 0:
            current_price = data['Close'].iloc[-1]
            current_date = data.index[-1]
            
            proceeds = trade_qty * current_price * (1 - commission)
            cash += proceeds
            
            entry_cost = trade_qty * entry_price * (1 + commission)
            profit = proceeds - entry_cost
            pct_return = profit / entry_cost if entry_cost > 0 else 0
            
            trades.append({
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": current_date,
                "exit_price": current_price,
                "quantity": trade_qty,
                "profit": profit,
                "return": pct_return
            })
            
            returns.append(pct_return)
            results["sell_signals"].append((current_date, current_price))
            portfolio_value = cash
            equity_curve.append((current_date, portfolio_value))
        
        # Calculate performance metrics
        if trades:
            results["trade_details"] = trades
            results["trades"] = len(trades)
            results["total_return"] = ((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            
            # Win rate
            winning_trades = [t for t in trades if t["profit"] > 0]
            losing_trades = [t for t in trades if t["profit"] <= 0]
            results["win_rate"] = (len(winning_trades) / len(trades)) * 100 if trades else 0
            
            # Profit factor
            total_wins = sum(t["profit"] for t in winning_trades) if winning_trades else 0
            total_losses = abs(sum(t["profit"] for t in losing_trades)) if losing_trades else 1
            results["profit_factor"] = total_wins / total_losses if total_losses > 0 else 0
            
            # Average win/loss
            results["average_win"] = np.mean([t["return"] for t in winning_trades]) * 100 if winning_trades else 0
            results["average_loss"] = np.mean([t["return"] for t in losing_trades]) * 100 if losing_trades else 0
            
            # Annual return
            if len(data) > 0:
                days = (data.index[-1] - data.index[0]).days
                years = days / 365.25
                if years > 0:
                    results["annual_return"] = ((portfolio_value / INITIAL_CAPITAL) ** (1/years) - 1) * 100
            
            # Sharpe ratio
            if returns:
                returns_array = np.array(returns)
                if len(returns_array) > 1 and returns_array.std() > 0:
                    results["sharpe_ratio"] = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
        
        # Max drawdown
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"])
            equity_df["RunningMax"] = equity_df["Equity"].cummax()
            equity_df["Drawdown"] = ((equity_df["Equity"] - equity_df["RunningMax"]) / 
                                     equity_df["RunningMax"]) * 100
            results["max_drawdown"] = equity_df["Drawdown"].min()
            results["equity_curve"] = equity_df
            
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        st.error(f"Backtesting error: {str(e)}")
    
    return results

# ========================= STOCK ANALYSIS =========================

def analyze_stock(data: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive stock analysis with all indicators"""
    
    if data is None or data.empty or len(data) < 14:
        logger.warning("Insufficient data for analysis")
        return data if data is not None else pd.DataFrame()
    
    try:
        # Compute all indicators
        data = compute_indicators(data)
        
        # Add divergence detection
        data['Divergence'] = detect_divergence(data)
        
        # Add market regime
        data['Regime'] = classify_market_regime(data)
        
        # Ensure all expected columns exist
        expected_columns = [
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'ATR', 'ADX', 'CMF',
            'Upper_Band', 'Middle_Band', 'Lower_Band',
            'SlowK', 'SlowD',
            'Donchian_Upper', 'Donchian_Lower', 'Donchian_Middle',
            'Ichimoku_Tenkan', 'Ichimoku_Kijun',
            'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou',
            'Volume_Spike', 'Avg_Volume'
        ]
        
        for col in expected_columns:
            if col not in data.columns:
                data[col] = np.nan
        
    except Exception as e:
        logger.error(f"Error in analyze_stock: {e}")
    
    return data

def analyze_stock_parallel(symbol: str) -> Optional[Dict[str, Any]]:
    """Parallel stock analysis for batch processing"""
    try:
        logger.info(f"Analyzing {symbol}")
        
        # Fetch data
        data = fetch_stock_data_cached(symbol)
        
        if data.empty or len(data) < 50:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} rows")
            return None
        
        # Analyze
        data = analyze_stock(data)
        
        # Get recommendation
        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
        
        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(data, symbol)
            return {
                "Symbol": symbol,
                "Current Price": rec.get("Current Price"),
                "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"),
                "Target": rec.get("Target"),
                "Recommendation": rec.get("Recommendation", "Hold"),
                "Score": rec.get("Score", 0),
                "Regime": rec.get("Regime"),
                "Position Size": rec.get("Position Size"),
                "Trailing Stop": rec.get("Trailing Stop"),
                "Reason": rec.get("Reason"),
                "Confidence": rec.get("Confidence", 0)
            }
        else:
            rec = generate_recommendations(data, symbol)
            return {
                "Symbol": symbol,
                "Current Price": rec.get("Current Price"),
                "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"),
                "Target": rec.get("Target"),
                "Intraday": rec.get("Intraday", "Hold"),
                "Swing": rec.get("Swing", "Hold"),
                "Short-Term": rec.get("Short-Term", "Hold"),
                "Long-Term": rec.get("Long-Term", "Hold"),
                "Mean_Reversion": rec.get("Mean_Reversion", "Hold"),
                "Breakout": rec.get("Breakout", "Hold"),
                "Ichimoku_Trend": rec.get("Ichimoku_Trend", "Hold"),
                "Score": rec.get("Score", 0)
            }
            
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

# ========================= BATCH PROCESSING =========================

def analyze_batch(stock_batch: List[str]) -> List[Dict[str, Any]]:
    """Analyze a batch of stocks with proper error handling"""
    results = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol): symbol 
                  for symbol in stock_batch}
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result(timeout=30)
                if result:
                    results.append(result)
            except Exception as e:
                error_msg = f"Error processing {symbol}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
    
    if errors:
        logger.warning(f"Batch processing completed with {len(errors)} errors")
    
    return results

def analyze_all_stocks(stock_list: List[str], batch_size: int = 10, 
                       progress_callback: Optional[callable] = None) -> pd.DataFrame:
    """Analyze all stocks in batches"""
    results = []
    
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        
        if progress_callback:
            progress = min(1.0, (i + len(batch)) / len(stock_list))
            progress_callback(progress)
        
        # Rate limiting between batches
        if i + batch_size < len(stock_list):
            time.sleep(2)
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Filter and sort
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    
    if recommendation_mode == "Adaptive":
        if "Recommendation" in results_df.columns:
            results_df = results_df[results_df["Recommendation"].isin(["Buy", "Sell"])]
    else:
        if "Intraday" in results_df.columns:
            buy_conditions = results_df["Intraday"].str.contains("Buy", case=False, na=False)
            results_df = results_df[buy_conditions]
    
    if "Score" in results_df.columns:
        results_df = results_df.sort_values(by="Score", ascending=False)
    
    return results_df.head(5)

# ========================= DATABASE OPERATIONS =========================

def init_database():
    """Initialize SQLite database for storing picks"""
    try:
        conn = sqlite3.connect('stock_picks.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_picks (
                date TEXT,
                symbol TEXT,
                score REAL,
                current_price REAL,
                buy_at REAL,
                stop_loss REAL,
                target REAL,
                intraday TEXT,
                swing TEXT,
                short_term TEXT,
                long_term TEXT,
                mean_reversion TEXT,
                breakout TEXT,
                ichimoku_trend TEXT,
                recommendation TEXT,
                regime TEXT,
                position_size REAL,
                trailing_stop REAL,
                reason TEXT,
                confidence REAL,
                pick_type TEXT,
                PRIMARY KEY (date, symbol, pick_type)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def insert_top_picks(results_df: pd.DataFrame, pick_type: str = "daily"):
    """Insert top picks into database"""
    try:
        conn = sqlite3.connect('stock_picks.db')
        
        for _, row in results_df.head(5).iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO daily_picks (
                    date, symbol, score, current_price, buy_at, stop_loss, target,
                    intraday, swing, short_term, long_term, mean_reversion, breakout,
                    ichimoku_trend, recommendation, regime, position_size, trailing_stop,
                    reason, confidence, pick_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                row.get('Symbol'),
                row.get('Score', 0),
                row.get('Current Price'),
                row.get('Buy At'),
                row.get('Stop Loss'),
                row.get('Target'),
                row.get('Intraday'),
                row.get('Swing'),
                row.get('Short-Term'),
                row.get('Long-Term'),
                row.get('Mean_Reversion'),
                row.get('Breakout'),
                row.get('Ichimoku_Trend'),
                row.get('Recommendation'),
                row.get('Regime'),
                row.get('Position Size'),
                row.get('Trailing Stop'),
                row.get('Reason'),
                row.get('Confidence', 0),
                pick_type
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(results_df.head(5))} picks of type {pick_type}")
        
    except Exception as e:
        logger.error(f"Error inserting picks: {e}")

# ========================= UI COMPONENTS =========================

def colored_recommendation(recommendation: Optional[str]) -> str:
    """Color code recommendations for better visualization"""
    if recommendation is None or not isinstance(recommendation, str):
        return "⚪ N/A"
    
    recommendation_lower = recommendation.lower()
    
    if "strong buy" in recommendation_lower:
        return f"🟢🟢 {recommendation}"
    elif "buy" in recommendation_lower:
        return f"🟢 {recommendation}"
    elif "strong sell" in recommendation_lower:
        return f"🔴🔴 {recommendation}"
    elif "sell" in recommendation_lower:
        return f"🔴 {recommendation}"
    else:
        return f"⚪ {recommendation}"

def tooltip(label: str, explanation: str) -> str:
    """Create tooltip for UI elements"""
    return f"{label} ℹ️ ({explanation})"

def update_progress(progress_bar, loading_text, progress_value, loading_messages):
    """Update progress bar with messages"""
    progress_bar.progress(progress_value)
    
    try:
        loading_message = next(loading_messages)
    except StopIteration:
        loading_message = "Processing..."
    
    dots = "." * (int(progress_value * 10) % 4)
    current_step = int(progress_value * 20)
    
    if not hasattr(update_progress, '_last_text_update'):
        update_progress._last_text_update = -1
    
    if current_step != update_progress._last_text_update:
        loading_text.text(f"{loading_message}{dots}")
        update_progress._last_text_update = current_step

# ========================= MAIN UI =========================

def display_dashboard(symbol: Optional[str] = None, 
                     data: Optional[pd.DataFrame] = None,
                     recommendations: Optional[Dict[str, Any]] = None):
    """Main dashboard display"""
    
    # Initialize session state
    if 'selected_sectors' not in st.session_state:
        st.session_state.selected_sectors = ["Bank"]
    if 'symbol' not in st.session_state:
        st.session_state.symbol = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'recommendation_mode' not in st.session_state:
        st.session_state.recommendation_mode = "Standard"
    
    # Update session state if new data provided
    if symbol and data is not None and recommendations is not None:
        st.session_state.symbol = symbol
        st.session_state.data = data
        st.session_state.recommendations = recommendations
    
    st.title("📊 StockGenie Pro - Advanced NSE Analysis")
    st.subheader(f"📅 Market Analysis for {datetime.now().strftime('%d %B %Y')}")
    
    # Sector selection
    st.sidebar.subheader("📂 Sector Selection")
    sector_options = ["All"] + list(SECTORS.keys())
    st.session_state.selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=sector_options,
        default=st.session_state.selected_sectors,
        help="Choose sectors to analyze. Select 'All' for comprehensive analysis."
    )
    
    if "All" in st.session_state.selected_sectors:
        selected_stocks = list(set([stock for sector in SECTORS.values() for stock in sector]))
    else:
        selected_stocks = list(set([stock for sector in st.session_state.selected_sectors 
                                   for stock in SECTORS.get(sector, [])]))
    
    if not selected_stocks:
        st.warning("⚠️ No stocks selected. Please choose at least one sector.")
        return
    
    # Analysis buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Generate Daily Top Picks", use_container_width=True):
            progress_bar = st.progress(0)
            loading_text = st.empty()
            loading_messages = itertools.cycle([
                "Analyzing market trends", "Evaluating indicators", 
                "Computing signals", "Ranking stocks", "Finalizing picks"
            ])
            
            results_df = analyze_all_stocks(
                selected_stocks,
                batch_size=config.BATCH_SIZE,
                progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
            )
            
            progress_bar.empty()
            loading_text.empty()
            
            if not results_df.empty:
                insert_top_picks(results_df, pick_type="daily")
                st.success("✅ Analysis Complete!")
                
                st.subheader("🏆 Today's Top 5 Stock Picks")
                
                for idx, row in results_df.iterrows():
                    score_color = "🟢" if row['Score'] > 3 else "🟡" if row['Score'] > 0 else "🔴"
                    
                    with st.expander(f"{score_color} {row['Symbol']} - Score: {row['Score']:.1f}/10"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"₹{row.get('Current Price', 'N/A')}")
                            st.metric("Buy At", f"₹{row.get('Buy At', 'N/A')}")
                        
                        with col2:
                            st.metric("Stop Loss", f"₹{row.get('Stop Loss', 'N/A')}")
                            st.metric("Target", f"₹{row.get('Target', 'N/A')}")
                        
                        with col3:
                            if st.session_state.recommendation_mode == "Adaptive":
                                st.metric("Signal", row.get('Recommendation', 'N/A'))
                                st.metric("Confidence", f"{row.get('Confidence', 0):.0f}%")
                            else:
                                st.metric("Intraday", row.get('Intraday', 'N/A'))
                                st.metric("Swing", row.get('Swing', 'N/A'))
                        
                        if st.session_state.recommendation_mode == "Adaptive":
                            st.info(f"**Market Regime:** {row.get('Regime', 'N/A')}")
                            st.info(f"**Position Size:** {row.get('Position Size', 'N/A')} shares")
                            st.info(f"**Reason:** {row.get('Reason', 'N/A')}")
            else:
                st.warning("⚠️ No suitable picks found. Try different sectors or parameters.")
    
    with col2:
        if st.button("⚡ Intraday Quick Picks", use_container_width=True):
            with st.spinner("Scanning for intraday opportunities..."):
                # Focus on high liquidity stocks for intraday
                intraday_stocks = selected_stocks[:20]  # Top 20 most liquid
                
                intraday_results = []
                for symbol in intraday_stocks:
                    data = fetch_stock_data_cached(symbol, period="1mo", interval="1d")
                    if not data.empty:
                        data = analyze_stock(data)
                        rec = adaptive_recommendation(data, symbol) if st.session_state.recommendation_mode == "Adaptive" else generate_recommendations(data, symbol)
                        
                        if st.session_state.recommendation_mode == "Adaptive":
                            if rec.get("Recommendation") == "Buy" and rec.get("Score", 0) > 2:
                                intraday_results.append({
                                    "Symbol": symbol,
                                    "Score": rec.get("Score"),
                                    "Current Price": rec.get("Current Price"),
                                    "Target": rec.get("Target"),
                                    "Stop Loss": rec.get("Stop Loss")
                                })
                        else:
                            if "Buy" in rec.get("Intraday", ""):
                                intraday_results.append({
                                    "Symbol": symbol,
                                    "Score": rec.get("Score"),
                                    "Current Price": rec.get("Current Price"),
                                    "Target": rec.get("Target"),
                                    "Stop Loss": rec.get("Stop Loss")
                                })
                
                if intraday_results:
                    intraday_df = pd.DataFrame(intraday_results).sort_values("Score", ascending=False).head(3)
                    st.success("✅ Found intraday opportunities!")
                    
                    for _, row in intraday_df.iterrows():
                        st.info(f"**{row['Symbol']}** - Entry: ₹{row['Current Price']:.2f}, "
                               f"Target: ₹{row['Target']:.2f}, SL: ₹{row['Stop Loss']:.2f}")
                else:
                    st.warning("No strong intraday setups found currently.")
    
    with col3:
        if st.button("📜 View Historical Performance", use_container_width=True):
            conn = sqlite3.connect('stock_picks.db')
            history_df = pd.read_sql_query(
                "SELECT * FROM daily_picks ORDER BY date DESC LIMIT 50", 
                conn
            )
            conn.close()
            
            if not history_df.empty:
                st.subheader("📊 Recent Pick Performance")
                
                # Performance summary
                total_picks = len(history_df)
                buy_picks = history_df[history_df['recommendation'] == 'Buy'] if 'recommendation' in history_df else pd.DataFrame()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Picks", total_picks)
                with col2:
                    st.metric("Buy Signals", len(buy_picks))
                with col3:
                    avg_score = history_df['score'].mean() if 'score' in history_df else 0
                    st.metric("Avg Score", f"{avg_score:.1f}")
                
                # Show recent picks
                st.dataframe(
                    history_df[['date', 'symbol', 'score', 'recommendation', 'current_price', 'target']].head(10),
                    use_container_width=True
                )
            else:
                st.info("No historical data available yet.")
    
    # Individual Stock Analysis Section
    if st.session_state.symbol and st.session_state.data is not None:
        st.markdown("---")
        st.header(f"📈 Detailed Analysis: {st.session_state.symbol}")
        
        data = st.session_state.data
        recommendations = st.session_state.recommendations
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current = recommendations.get('Current Price', 0)
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current
            change = ((current - prev_close) / prev_close * 100) if prev_close else 0
            st.metric(
                "Current Price",
                f"₹{current:.2f}",
                f"{change:+.2f}%"
            )
        
        with col2:
            st.metric("Buy At", f"₹{recommendations.get('Buy At', 'N/A')}")
        
        with col3:
            st.metric("Stop Loss", f"₹{recommendations.get('Stop Loss', 'N/A')}")
        
        with col4:
            st.metric("Target", f"₹{recommendations.get('Target', 'N/A')}")
        
        # Technical Indicators
        st.subheader("📊 Technical Indicators")
        
        indicators_data = []
        
        # Collect indicator values
        indicator_list = [
            ("RSI", data['RSI'].iloc[-1] if 'RSI' in data.columns else None, 30, 70),
            ("MACD", data['MACD'].iloc[-1] if 'MACD' in data.columns else None, None, None),
            ("ADX", data['ADX'].iloc[-1] if 'ADX' in data.columns else None, 25, None),
            ("ATR", data['ATR'].iloc[-1] if 'ATR' in data.columns else None, None, None),
            ("CMF", data['CMF'].iloc[-1] if 'CMF' in data.columns else None, -0.1, 0.1)
        ]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        for idx, (name, value, low_thresh, high_thresh) in enumerate(indicator_list):
            if value is not None and pd.notna(value):
                value = float(value)
                
                # Determine status
                if low_thresh and high_thresh:
                    if value < low_thresh:
                        status = "🟢 Oversold"
                    elif value > high_thresh:
                        status = "🔴 Overbought"
                    else:
                        status = "⚪ Neutral"
                elif low_thresh:
                    status = "🟢 Strong" if value > low_thresh else "🔴 Weak"
                else:
                    status = ""
                
                with [col1, col2, col3, col4, col5][idx]:
                    st.metric(
                        tooltip(name, TOOLTIPS.get(name, "")),
                        f"{value:.2f}",
                        status
                    )
        
        # Chart
        st.subheader("📈 Price Chart with Signals")
        
        fig = px.line(data.reset_index(), x='Date', y='Close', 
                     title=f"{st.session_state.symbol} - Price Movement")
        
        # Add Bollinger Bands
        if 'Upper_Band' in data.columns:
            fig.add_scatter(x=data.index, y=data['Upper_Band'], 
                          name='Upper BB', line=dict(dash='dash', color='gray'))
            fig.add_scatter(x=data.index, y=data['Lower_Band'], 
                          name='Lower BB', line=dict(dash='dash', color='gray'))
        
        # Add moving averages
        if 'SMA_20' in data.columns:
            fig.add_scatter(x=data.index, y=data['SMA_20'], 
                          name='SMA 20', line=dict(color='orange'))
        if 'SMA_50' in data.columns:
            fig.add_scatter(x=data.index, y=data['SMA_50'], 
                          name='SMA 50', line=dict(color='blue'))
        
        fig.update_layout(height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Backtest Section
        st.subheader("🔬 Backtest Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Swing Trading Backtest", use_container_width=True):
                with st.spinner("Running backtest..."):
                    backtest_results = backtest_stock(data, st.session_state.symbol, "Swing")
                    st.session_state.backtest_swing = backtest_results
        
        with col2:
            if st.button("Run Intraday Backtest", use_container_width=True):
                with st.spinner("Running backtest..."):
                    backtest_results = backtest_stock(data, st.session_state.symbol, "Intraday")
                    st.session_state.backtest_intraday = backtest_results
        
        # Display backtest results
        for strategy in ["swing", "intraday"]:
            if f"backtest_{strategy}" in st.session_state:
                results = st.session_state[f"backtest_{strategy}"]
                
                st.markdown(f"**{strategy.capitalize()} Strategy Results:**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{results['total_return']:.2f}%")
                    st.metric("Annual Return", f"{results['annual_return']:.2f}%")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                
                with col3:
                    st.metric("Total Trades", results['trades'])
                    st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                
                with col4:
                    st.metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
                    st.metric("Avg Win", f"{results.get('average_win', 0):.2f}%")
                
                # Equity curve
                if 'equity_curve' in results and not results['equity_curve'].empty:
                    fig_equity = px.line(results['equity_curve'], x='Date', y='Equity',
                                        title=f"Equity Curve - {strategy.capitalize()}")
                    fig_equity.update_layout(height=300)
                    st.plotly_chart(fig_equity, use_container_width=True)

def main():
    """Main application entry point"""
    
    # Setup
    warnings.filterwarnings("ignore")
    init_database()
    
    # Sidebar configuration
    st.sidebar.title("🔍 StockGenie Pro")
    st.sidebar.markdown("---")
    
    # Fetch stock list
    with st.spinner("Loading stock data..."):
        stock_list = fetch_nse_stock_list()
    
    if not stock_list:
        st.error("❌ Could not fetch stock list. Please check your connection.")
        return
    
    # Stock selection
    st.sidebar.subheader("📌 Individual Stock Analysis")
    
    selected_symbol = st.sidebar.selectbox(
        "Select Stock",
        stock_list,
        index=0,
        help="Choose a stock for detailed analysis"
    )
    
    # Analysis mode
    st.sidebar.subheader("⚙️ Settings")
    
    recommendation_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Standard", "Adaptive"],
        index=0 if st.session_state.get('recommendation_mode', 'Standard') == "Standard" else 1,
        help="Standard: Traditional indicators | Adaptive: AI-enhanced with regime detection"
    )
    st.session_state.recommendation_mode = recommendation_mode
    
    # Position sizing
    position_size = st.sidebar.slider(
        "Position Size (%)",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Percentage of capital to allocate per trade"
    )
    st.session_state.position_size = position_size / 100
    
    # Analyze button
    if st.sidebar.button("📊 Analyze Stock", use_container_width=True):
        with st.spinner(f"Analyzing {selected_symbol}..."):
            try:
                # Fetch and analyze data
                data = fetch_stock_data_with_auth(selected_symbol)
                
                if not data.empty:
                    analyzed_data = analyze_stock(data)
                    
                    if recommendation_mode == "Adaptive":
                        recommendations = adaptive_recommendation(analyzed_data, selected_symbol)
                    else:
                        recommendations = generate_recommendations(analyzed_data, selected_symbol)
                    
                    # Update session state
                    st.session_state.symbol = selected_symbol
                    st.session_state.data = analyzed_data
                    st.session_state.recommendations = recommendations
                    
                    st.success(f"✅ Analysis complete for {selected_symbol}")
                else:
                    st.warning(f"⚠️ No data available for {selected_symbol}")
                    
            except Exception as e:
                st.error(f"❌ Error analyzing {selected_symbol}: {str(e)}")
    
    # Display dashboard
    display_dashboard()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tips:**\n"
        "- Use Adaptive mode for AI-powered analysis\n"
        "- Check multiple timeframes before trading\n"
        "- Always use stop-loss orders\n"
        "- Past performance doesn't guarantee future results"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("StockGenie Pro v2.0 | Market data may be delayed")

if __name__ == "__main__":
    main()
