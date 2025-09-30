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
import plotly.graph_objects as go
import time
import requests
import io
import random
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
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import Queue, Empty
import json
from pydantic import BaseModel, validator, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from numba import jit
import asyncio
import aiohttp
from datetime import timezone

load_dotenv()
warnings.filterwarnings("ignore")

# ========================= ENHANCED CONFIGURATION =========================

@dataclass
class TradingConfig:
    """Enhanced trading configuration with validation"""
    DEFAULT_RISK_PER_TRADE: float = 0.02
    MAX_POSITION_SIZE_PCT: float = 0.25
    DEFAULT_STOP_LOSS_ATR_MULT: float = 2.0
    CACHE_TTL: int = 86400
    MAX_WORKERS: int = 3
    BATCH_SIZE: int = 10
    INITIAL_CAPITAL: float = 30000
    COMMISSION: float = 0.001
    
    # Rate limiting configuration
    RATE_LIMIT_CALLS: int = 5
    RATE_LIMIT_PERIOD: int = 60
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: int = 4
    RETRY_MAX_WAIT: int = 60
    
    # Alert configuration
    ALERT_CHECK_INTERVAL: int = 60  # seconds
    MAX_ALERTS_PER_SYMBOL: int = 5
    
    # Performance configuration
    USE_NUMBA: bool = True
    VECTORIZE_OPERATIONS: bool = True
    MAX_CACHE_SIZE: int = 1000000000  # 1GB

config = TradingConfig()

# ========================= ENHANCED LOGGING =========================

class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for better log visibility"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(level=logging.INFO):
    """Enhanced logging setup with colored output and file rotation"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler for detailed logs
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        'stockgenie.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ========================= ENHANCED EXCEPTIONS =========================

class StockAnalysisError(Exception):
    """Base exception for stock analysis errors"""
    pass

class DataFetchError(StockAnalysisError):
    """Exception for data fetching errors"""
    pass

class ValidationError(StockAnalysisError):
    """Exception for data validation errors"""
    pass

class RateLimitError(StockAnalysisError):
    """Exception for rate limit errors"""
    pass

class APIError(StockAnalysisError):
    """Exception for API errors"""
    pass

# ========================= DATA VALIDATION MODELS =========================

class PriceLevel(str, Enum):
    """Price level types"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT = "pivot"

class SignalStrength(str, Enum):
    """Signal strength levels"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class StockDataValidator(BaseModel):
    """Pydantic model for stock data validation"""
    symbol: str = Field(..., min_length=1, max_length=20)
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    date: datetime
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= Low')
        return v
    
    @validator('close')
    def close_within_range(cls, v, values):
        if 'high' in values and 'low' in values:
            if not values['low'] <= v <= values['high']:
                raise ValueError('Close must be between Low and High')
        return v

# ========================= THREAD-SAFE COMPONENTS =========================

class ThreadSafeCache:
    """Thread-safe caching implementation"""
    
    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._access_count = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Thread-safe get operation"""
        with self._lock:
            if key in self._cache:
                self._access_count[key] = self._access_count.get(key, 0) + 1
                cache_entry = self._cache[key]
                
                # Check if expired
                if isinstance(cache_entry, dict) and 'expires_at' in cache_entry:
                    if cache_entry['expires_at'] and time.time() > cache_entry['expires_at']:
                        del self._cache[key]
                        return None
                    return cache_entry.get('value')
                return cache_entry
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Thread-safe set operation with TTL support"""
        with self._lock:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self._max_size:
                # Remove least recently used item
                if self._access_count:
                    lru_key = min(self._access_count, key=self._access_count.get)
                    if lru_key in self._cache:
                        del self._cache[lru_key]
                    if lru_key in self._access_count:
                        del self._access_count[lru_key]
            
            self._cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl if ttl else None
            }
            self._access_count[key] = 0
    
    def delete(self, key: str) -> None:
        """Thread-safe delete operation"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_count:
                del self._access_count[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()

class ThreadSafeRateLimiter:
    """Enhanced thread-safe rate limiter with exponential backoff"""
    
    def __init__(self, calls: int = 10, period: int = 60):
        self.calls = calls
        self.period = period
        self.call_times = deque()
        self._lock = threading.RLock()
        self._backoff_until = {}
        
    def acquire(self, identifier: str = "default") -> bool:
        """Try to acquire permission to make a call"""
        with self._lock:
            now = time.time()
            
            # Check if in backoff period
            if identifier in self._backoff_until:
                if now < self._backoff_until[identifier]:
                    return False
                else:
                    del self._backoff_until[identifier]
            
            # Remove old calls
            while self.call_times and self.call_times[0] < now - self.period:
                self.call_times.popleft()
            
            # Check rate limit
            if len(self.call_times) >= self.calls:
                # Calculate backoff
                backoff_time = self.period - (now - self.call_times[0]) + 1
                self._backoff_until[identifier] = now + backoff_time
                return False
            
            self.call_times.append(now)
            return True
    
    def wait_if_needed(self, identifier: str = "default") -> None:
        """Wait if rate limited"""
        while not self.acquire(identifier):
            time.sleep(0.1)

# Global instances
cache = ThreadSafeCache(max_size=10000)
rate_limiter = ThreadSafeRateLimiter(calls=config.RATE_LIMIT_CALLS, period=config.RATE_LIMIT_PERIOD)

# ========================= ENHANCED ERROR HANDLING =========================

def handle_api_errors(func):
    """Decorator for comprehensive API error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif e.response and e.response.status_code == 401:
                raise APIError(f"Authentication failed: {e}")
            elif e.response and e.response.status_code >= 500:
                raise APIError(f"Server error: {e}")
            else:
                raise APIError(f"HTTP error: {e}")
        except requests.exceptions.ConnectionError as e:
            raise DataFetchError(f"Connection error: {e}")
        except requests.exceptions.Timeout as e:
            raise DataFetchError(f"Request timeout: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise StockAnalysisError(f"Unexpected error: {e}")
    return wrapper

@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT),
    retry=retry_if_exception_type((RateLimitError, DataFetchError)),
    reraise=True
)
@handle_api_errors
def fetch_with_retry(url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None) -> requests.Response:
    """Fetch data with retry logic and error handling"""
    
    # Wait for rate limit
    rate_limiter.wait_if_needed(url)
    
    # Set default headers
    if headers is None:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
    
    # Make request
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    
    return response

# ========================= FIXED NUMBA FUNCTIONS =========================

@jit(nopython=True)
def calculate_rsi_optimized(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Numba-optimized RSI calculation"""
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)
    
    deltas = np.diff(prices)
    rsi = np.full(len(prices), np.nan)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        rsi[period] = 100 if avg_gain > 0 else 50
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    for i in range(period + 1, len(prices)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi[i] = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

@jit(nopython=True)
def calculate_ema_optimized(prices: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized EMA calculation"""
    ema = np.full(len(prices), np.nan)
    if len(prices) < period:
        return ema
    
    ema[period - 1] = np.mean(prices[:period])
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema

def convert_masked_to_regular_array(arr):
    """Convert MaskedArray to regular numpy array"""
    if isinstance(arr, np.ma.MaskedArray):
        # Fill masked values with the mean of non-masked values
        if arr.count() > 0:
            filled_value = arr.mean()
            return arr.filled(fill_value=filled_value)
        else:
            return arr.data
    return arr

# ========================= COMPLETE ALERT SYSTEM =========================

class AlertType(Enum):
    """Alert types enumeration"""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    VOLUME_SPIKE = "volume_spike"
    BREAKOUT = "breakout"
    DIVERGENCE = "divergence"
    STOP_LOSS_HIT = "stop_loss_hit"
    TARGET_REACHED = "target_reached"

class Alert:
    """Individual alert class"""
    
    def __init__(self, symbol: str, alert_type: AlertType, threshold: float, 
                 message: str, callback: Optional[callable] = None):
        self.id = hashlib.md5(f"{symbol}{alert_type}{threshold}{time.time()}".encode()).hexdigest()[:8]
        self.symbol = symbol
        self.alert_type = alert_type
        self.threshold = threshold
        self.message = message
        self.callback = callback
        self.created_at = datetime.now()
        self.triggered = False
        self.triggered_at = None
        self.trigger_count = 0
        
    def check_condition(self, current_value: float) -> bool:
        """Check if alert condition is met"""
        if self.alert_type == AlertType.PRICE_ABOVE:
            return current_value > self.threshold
        elif self.alert_type == AlertType.PRICE_BELOW:
            return current_value < self.threshold
        elif self.alert_type == AlertType.RSI_OVERSOLD:
            return current_value < self.threshold
        elif self.alert_type == AlertType.RSI_OVERBOUGHT:
            return current_value > self.threshold
        return False
    
    def trigger(self, current_value: float) -> None:
        """Trigger the alert"""
        self.triggered = True
        self.triggered_at = datetime.now()
        self.trigger_count += 1
        
        alert_message = f"🔔 ALERT: {self.symbol} - {self.message} (Current: {current_value:.2f}, Threshold: {self.threshold:.2f})"
        logger.info(alert_message)
        
        if self.callback:
            try:
                self.callback(self, current_value)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Store in session state for UI display
        if 'alerts_triggered' not in st.session_state:
            st.session_state.alerts_triggered = []
        st.session_state.alerts_triggered.append({
            'time': self.triggered_at,
            'symbol': self.symbol,
            'message': alert_message,
            'type': self.alert_type.value
        })

class AlertMonitor:
    """Complete Alert monitoring system with all methods"""
    
    def __init__(self):
        self.alerts: Dict[str, List[Alert]] = {}
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
    def add_alert(self, alert: Alert) -> str:
        """Add new alert"""
        with self._lock:
            if alert.symbol not in self.alerts:
                self.alerts[alert.symbol] = []
            
            # Check max alerts per symbol
            if len(self.alerts[alert.symbol]) >= config.MAX_ALERTS_PER_SYMBOL:
                # Remove oldest alert
                self.alerts[alert.symbol].pop(0)
            
            self.alerts[alert.symbol].append(alert)
            logger.info(f"Alert added for {alert.symbol}: {alert.message}")
            return alert.id
    
    def remove_alert(self, alert_id: str) -> bool:
        """Remove alert by ID"""
        with self._lock:
            for symbol, alerts in self.alerts.items():
                for alert in alerts:
                    if alert.id == alert_id:
                        alerts.remove(alert)
                        logger.info(f"Alert removed: {alert_id}")
                        return True
            return False
    
    def get_alerts(self, symbol: Optional[str] = None) -> List[Alert]:
        """Get alerts for symbol or all alerts"""
        with self._lock:
            if symbol:
                return self.alerts.get(symbol, []).copy()
            else:
                all_alerts = []
                for alerts in self.alerts.values():
                    all_alerts.extend(alerts)
                return all_alerts
    
    def check_alerts(self, symbol: str, data: pd.DataFrame) -> List[Alert]:
        """Check alerts for a symbol"""
        triggered_alerts = []
        
        with self._lock:
            if symbol not in self.alerts or data.empty:
                return triggered_alerts
            
            # Get current values
            current_price = data['Close'].iloc[-1] if 'Close' in data.columns else None
            current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else None
            current_volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else None
            avg_volume = data['Volume_SMA'].iloc[-1] if 'Volume_SMA' in data.columns else None
            
            for alert in self.alerts[symbol]:
                if alert.triggered and alert.trigger_count >= 3:
                    continue  # Skip if already triggered multiple times
                
                should_trigger = False
                current_value = None
                
                if alert.alert_type in [AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW] and current_price:
                    current_value = current_price
                    should_trigger = alert.check_condition(current_price)
                
                elif alert.alert_type in [AlertType.RSI_OVERSOLD, AlertType.RSI_OVERBOUGHT] and current_rsi:
                    if not pd.isna(current_rsi):
                        current_value = current_rsi
                        should_trigger = alert.check_condition(current_rsi)
                
                elif alert.alert_type == AlertType.VOLUME_SPIKE and current_volume and avg_volume:
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        current_value = volume_ratio
                        should_trigger = volume_ratio > alert.threshold
                
                if should_trigger and current_value is not None:
                    alert.trigger(current_value)
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def start_monitoring(self, check_interval: int = 60):
        """Start background monitoring"""
        if self._monitoring:
            logger.info("Monitoring already running")
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(check_interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
    
    def _monitor_loop(self, check_interval: int):
        """Background monitoring loop"""
        logger.info(f"Monitor loop started with {check_interval}s interval")
        
        while not self._stop_event.is_set():
            try:
                # Get list of symbols to check
                with self._lock:
                    symbols_to_check = list(self.alerts.keys())
                
                if symbols_to_check:
                    logger.debug(f"Checking alerts for {len(symbols_to_check)} symbols")
                    
                    for symbol in symbols_to_check:
                        if self._stop_event.is_set():
                            break
                        
                        try:
                            # Fetch latest data
                            data = fetch_stock_data_with_validation(
                                symbol, 
                                period="1d", 
                                interval="5m"
                            )
                            
                            if not data.empty:
                                # Compute indicators
                                data = compute_indicators_optimized(data, symbol)
                                # Check alerts
                                triggered = self.check_alerts(symbol, data)
                                if triggered:
                                    logger.info(f"Triggered {len(triggered)} alerts for {symbol}")
                        except Exception as e:
                            logger.error(f"Error checking alerts for {symbol}: {e}")
                
                # Wait for next check
                self._stop_event.wait(check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                # Continue monitoring even if there's an error
                time.sleep(5)
        
        logger.info("Monitor loop stopped")
    
    def clear_alerts(self, symbol: Optional[str] = None):
        """Clear alerts for a symbol or all alerts"""
        with self._lock:
            if symbol:
                if symbol in self.alerts:
                    self.alerts[symbol].clear()
                    logger.info(f"Cleared alerts for {symbol}")
            else:
                self.alerts.clear()
                logger.info("Cleared all alerts")

# Global alert monitor instance
alert_monitor = AlertMonitor()

# ========================= API CONFIGURATION =========================

CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
API_KEYS = {
    "Historical": os.getenv("HISTORICAL_API_KEY", "c3C0tMGn"),
    "Trading": os.getenv("TRADING_API_KEY"),
    "Market": os.getenv("MARKET_API_KEY")
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
]

# ========================= DATA FETCHING =========================

@st.cache_data(ttl=86400)
def load_symbol_token_map() -> Dict[str, str]:
    """Load symbol to token mapping with retry logic"""
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = fetch_with_retry(url)
        data = response.json()
        
        symbol_map = {}
        for entry in data:
            if "symbol" in entry and "token" in entry:
                symbol_map[entry["symbol"]] = entry["token"]
        
        logger.info(f"Loaded {len(symbol_map)} symbol-token mappings")
        return symbol_map
        
    except Exception as e:
        logger.error(f"Failed to load symbol token map: {e}")
        st.error(f"⚠️ Failed to load instrument list: {str(e)}")
        return {}

def init_smartapi_client() -> Optional[SmartConnect]:
    """Initialize SmartAPI client with retry logic"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _init_client():
        smart_api = SmartConnect(api_key=API_KEYS["Historical"])
        totp = pyotp.TOTP(TOTP_SECRET)
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
        
        if data['status']:
            logger.info("SmartAPI client initialized successfully")
            return smart_api
        else:
            raise APIError(f"SmartAPI authentication failed: {data.get('message', 'Unknown error')}")
    
    try:
        return _init_client()
    except Exception as e:
        logger.error(f"Failed to initialize SmartAPI after retries: {e}")
        st.error(f"⚠️ Authentication failed: {str(e)}")
        return None

def validate_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLC data - Fixed version"""
    if df.empty:
        raise ValidationError("Empty dataframe")
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing columns: {missing_columns}")
    
    # Validate each row
    invalid_rows = []
    for idx, row in df.iterrows():
        try:
            validator = StockDataValidator(
                symbol="TEMP",
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=int(row['Volume']),
                date=idx if isinstance(idx, datetime) else datetime.now()
            )
        except Exception as e:
            invalid_rows.append(idx)
            logger.warning(f"Invalid row at {idx}: {e}")
    
    # Remove invalid rows
    if invalid_rows:
        df = df.drop(invalid_rows)
        logger.info(f"Removed {len(invalid_rows)} invalid rows")
    
    # Handle outliers - Fixed to convert MaskedArray to regular array
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            winsorized_data = winsorize(df[col].values, limits=[0.001, 0.001])
            # Convert MaskedArray to regular array
            df[col] = convert_masked_to_regular_array(winsorized_data)
    
    return df

@retry(
    stop=stop_after_attempt(config.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT),
    retry=retry_if_exception_type((RateLimitError, DataFetchError))
)
def fetch_stock_data_with_validation(symbol: str, period: str = "2y", 
                                    interval: str = "1d") -> pd.DataFrame:
    """Fetch stock data with validation and retry logic"""
    
    # Check cache first
    cache_key = f"{symbol}_{period}_{interval}_{datetime.now().date()}"
    cached_data = cache.get(cache_key)
    
    if cached_data and isinstance(cached_data, pd.DataFrame):
        logger.debug(f"Cache hit for {symbol}")
        return cached_data
    
    # Rate limit check
    if not rate_limiter.acquire(f"fetch_{symbol}"):
        logger.warning(f"Rate limited for {symbol}, waiting...")
        rate_limiter.wait_if_needed(f"fetch_{symbol}")
    
    try:
        # Ensure symbol format
        if "-EQ" not in symbol:
            symbol = f"{symbol.split('.')[0]}-EQ"
        
        smart_api = init_smartapi_client()
        if not smart_api:
            raise APIError("SmartAPI client initialization failed")
        
        # Calculate date range
        end_date = datetime.now()
        period_map = {
            "2y": timedelta(days=730),
            "1y": timedelta(days=365),
            "1mo": timedelta(days=30),
            "1w": timedelta(days=7),
            "1d": timedelta(days=1)
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
            raise ValidationError(f"Invalid symbol: {symbol}")
        
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
            
            # Validate data
            data = validate_ohlc_data(data)
            
            # Cache the validated data
            cache.set(cache_key, data, ttl=config.CACHE_TTL)
            
            logger.info(f"Successfully fetched and validated data for {symbol}: {len(data)} rows")
            return data
        else:
            error_msg = historical_data.get('message', 'No data received')
            raise DataFetchError(f"No data for {symbol}: {error_msg}")
            
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

# ========================= OPTIMIZED TECHNICAL INDICATORS =========================

def compute_indicators_optimized(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Fixed indicator computation without MaskedArray issues"""
    
    if df.empty or 'Close' not in df.columns:
        return df
    
    df = df.copy()
    
    try:
        # Ensure we're working with regular numpy arrays
        close_prices = np.asarray(df['Close'].values, dtype=np.float64)
        high_prices = np.asarray(df['High'].values, dtype=np.float64)
        low_prices = np.asarray(df['Low'].values, dtype=np.float64)
        
        # Remove any nan values for Numba functions
        close_prices_clean = np.nan_to_num(close_prices, nan=np.nanmean(close_prices))
        
        # Use optimized functions when available
        use_numba = config.USE_NUMBA
        
        # Check if arrays are suitable for Numba (no MaskedArrays)
        if isinstance(close_prices, np.ma.MaskedArray):
            close_prices = convert_masked_to_regular_array(close_prices)
            use_numba = False
            logger.debug("Detected MaskedArray, falling back to standard calculations")
        
        if use_numba and len(close_prices_clean) >= 14:
            try:
                # Optimized RSI
                df['RSI'] = calculate_rsi_optimized(close_prices_clean, 14)
                
                # Optimized EMAs for MACD
                if len(close_prices_clean) >= 26:
                    ema_12 = calculate_ema_optimized(close_prices_clean, 12)
                    ema_26 = calculate_ema_optimized(close_prices_clean, 26)
                    macd_line = ema_12 - ema_26
                    
                    # Clean MACD line for signal calculation
                    macd_clean = np.nan_to_num(macd_line, nan=0)
                    valid_macd = macd_clean[~np.isnan(macd_line)]
                    
                    if len(valid_macd) >= 9:
                        macd_signal = calculate_ema_optimized(valid_macd, 9)
                        df['MACD'] = macd_line
                        df['MACD_signal'] = np.nan
                        # Align the signal properly
                        signal_start = len(df) - len(macd_signal)
                        if signal_start >= 0 and signal_start < len(df):
                            df.iloc[signal_start:, df.columns.get_loc('MACD_signal')] = macd_signal
                    else:
                        df['MACD'] = macd_line
                        df['MACD_signal'] = np.nan
                        
            except Exception as e:
                logger.warning(f"Numba optimization failed, falling back to standard: {e}")
                use_numba = False
        
        # Fallback to pandas/ta if Numba fails or is disabled
        if not use_numba or len(df) < 14:
            if len(df) >= 14:
                try:
                    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
                except Exception as e:
                    logger.warning(f"RSI calculation failed: {e}")
                    df['RSI'] = np.nan
            else:
                df['RSI'] = np.nan
            
            if len(df) >= 26:
                try:
                    macd = ta.trend.MACD(df['Close'])
                    df['MACD'] = macd.macd()
                    df['MACD_signal'] = macd.macd_signal()
                except Exception as e:
                    logger.warning(f"MACD calculation failed: {e}")
                    df['MACD'] = np.nan
                    df['MACD_signal'] = np.nan
            else:
                df['MACD'] = np.nan
                df['MACD_signal'] = np.nan
        
        # Multiple SMAs
        sma_windows = [20, 50, 200]
        for window in sma_windows:
            if len(df) >= window:
                df[f'SMA_{window}'] = df['Close'].rolling(window, min_periods=1).mean()
            else:
                df[f'SMA_{window}'] = np.nan
        
        # Bollinger Bands
        if len(df) >= 20:
            try:
                middle = df['Close'].rolling(window=20, min_periods=1).mean()
                std = df['Close'].rolling(window=20, min_periods=1).std()
                df['Upper_Band'] = middle + (std * 2)
                df['Middle_Band'] = middle
                df['Lower_Band'] = middle - (std * 2)
            except Exception as e:
                logger.warning(f"Bollinger Bands calculation failed: {e}")
                df['Upper_Band'] = df['Middle_Band'] = df['Lower_Band'] = np.nan
        else:
            df['Upper_Band'] = df['Middle_Band'] = df['Lower_Band'] = np.nan
        
        # ATR and ADX
        if len(df) >= 14:
            try:
                df['ATR'] = ta.volatility.AverageTrueRange(
                    df['High'], df['Low'], df['Close'], window=14
                ).average_true_range()
            except Exception as e:
                logger.warning(f"ATR calculation failed: {e}")
                df['ATR'] = np.nan
            
            try:
                df['ADX'] = ta.trend.ADXIndicator(
                    df['High'], df['Low'], df['Close'], window=14
                ).adx()
            except Exception as e:
                logger.warning(f"ADX calculation failed: {e}")
                df['ADX'] = np.nan
        else:
            df['ATR'] = np.nan
            df['ADX'] = np.nan
        
        # Volume indicators
        if 'Volume' in df.columns and len(df) >= 20:
            try:
                df['Volume_SMA'] = df['Volume'].rolling(20, min_periods=1).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
                df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], 1.0)
            except Exception as e:
                logger.warning(f"Volume indicators calculation failed: {e}")
                df['Volume_SMA'] = df['Volume_Ratio'] = np.nan
            
            try:
                df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
                    df['High'], df['Low'], df['Close'], df['Volume'], window=20
                ).chaikin_money_flow()
            except Exception as e:
                logger.warning(f"CMF calculation failed: {e}")
                df['CMF'] = np.nan
        else:
            df['Volume_SMA'] = df['Volume_Ratio'] = df['CMF'] = np.nan
        
        # Stochastic
        if len(df) >= 14:
            try:
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
            except Exception as e:
                logger.warning(f"Stochastic calculation failed: {e}")
                df['Stoch_K'] = df['Stoch_D'] = np.nan
        else:
            df['Stoch_K'] = df['Stoch_D'] = np.nan
        
        # VWAP
        if 'Volume' in df.columns:
            try:
                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                df['VWAP'] = (df['Volume'] * typical_price).cumsum() / df['Volume'].cumsum()
                df['VWAP'] = df['VWAP'].replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                logger.warning(f"VWAP calculation failed: {e}")
                df['VWAP'] = np.nan
        else:
            df['VWAP'] = np.nan
        
        logger.debug(f"Computed indicators for {symbol if symbol else 'unknown'}")
        
    except Exception as e:
        logger.error(f"Error computing indicators: {e}")
        # Return dataframe with basic indicators even if some fail
        for col in ['RSI', 'MACD', 'MACD_signal', 'ATR', 'ADX', 'Upper_Band', 
                   'Middle_Band', 'Lower_Band', 'Volume_SMA', 'Volume_Ratio', 'CMF',
                   'Stoch_K', 'Stoch_D', 'VWAP']:
            if col not in df.columns:
                df[col] = np.nan
    
    return df

# ========================= SIGNAL GENERATION =========================

def generate_signals_with_alerts(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Generate trading signals and check for alert conditions"""
    
    signals = {
        "buy_signals": [],
        "sell_signals": [],
        "alerts": [],
        "score": 0
    }
    
    if df.empty or len(df) < 20:
        return signals
    
    try:
        current_price = df['Close'].iloc[-1]
        buy_score = 0
        sell_score = 0
        
        # Price action signals
        if 'Lower_Band' in df.columns and pd.notna(df['Lower_Band'].iloc[-1]):
            if current_price <= df['Lower_Band'].iloc[-1]:
                signals["buy_signals"].append("Bollinger Band Oversold")
                buy_score += 2
        
        if 'Upper_Band' in df.columns and pd.notna(df['Upper_Band'].iloc[-1]):
            if current_price >= df['Upper_Band'].iloc[-1]:
                signals["sell_signals"].append("Bollinger Band Overbought")
                sell_score += 2
        
        # RSI signals
        if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]):
            rsi = df['RSI'].iloc[-1]
            if rsi < 30:
                signals["buy_signals"].append(f"RSI Oversold ({rsi:.1f})")
                buy_score += 3
            elif rsi < 40:
                buy_score += 1
            elif rsi > 70:
                signals["sell_signals"].append(f"RSI Overbought ({rsi:.1f})")
                sell_score += 3
            elif rsi > 60:
                sell_score += 1
        
        # Volume spike
        if 'Volume_Ratio' in df.columns and pd.notna(df['Volume_Ratio'].iloc[-1]):
            if df['Volume_Ratio'].iloc[-1] > 2.0:
                signals["alerts"].append(f"Volume spike ({df['Volume_Ratio'].iloc[-1]:.1f}x)")
                if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                    buy_score += 1
                else:
                    sell_score += 1
        
        # MACD crossover
        if 'MACD' in df.columns and 'MACD_signal' in df.columns and len(df) > 1:
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_signal'].iloc[-1]
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['MACD_signal'].iloc[-2]
            
            if pd.notna(macd) and pd.notna(signal) and pd.notna(prev_macd) and pd.notna(prev_signal):
                if macd > signal and prev_macd <= prev_signal:
                    signals["buy_signals"].append("MACD Bullish Crossover")
                    buy_score += 2
                elif macd < signal and prev_macd >= prev_signal:
                    signals["sell_signals"].append("MACD Bearish Crossover")
                    sell_score += 2
        
        # Moving average signals
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            
            if pd.notna(sma_20) and pd.notna(sma_50):
                if current_price > sma_20 > sma_50:
                    buy_score += 1
                elif current_price < sma_20 < sma_50:
                    sell_score += 1
        
        # Calculate net score
        signals["score"] = buy_score - sell_score
        
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")
    
    return signals

def adaptive_recommendation_with_monitoring(df: pd.DataFrame, symbol: str, 
                                           account_size: float = 30000) -> Dict[str, Any]:
    """Enhanced adaptive recommendation with alert monitoring"""
    
    try:
        # Compute optimized indicators
        df = compute_indicators_optimized(df, symbol)
        
        if len(df) < 20:
            return {
                "Symbol": symbol,
                "Recommendation": "Hold",
                "Reason": "Insufficient data",
                "Score": 0
            }
        
        # Get current values
        current_price = df['Close'].iloc[-1]
        
        # Generate signals
        signals = generate_signals_with_alerts(df, symbol)
        
        # Check existing alerts
        triggered_alerts = alert_monitor.check_alerts(symbol, df)
        
        # Calculate scores
        net_score = signals["score"]
        
        # Generate recommendation
        if net_score >= 3:
            recommendation = "Strong Buy"
            confidence = min(100, net_score * 15)
        elif net_score >= 2:
            recommendation = "Buy"
            confidence = min(100, net_score * 20)
        elif net_score <= -3:
            recommendation = "Strong Sell"
            confidence = min(100, abs(net_score) * 15)
        elif net_score <= -2:
            recommendation = "Sell"
            confidence = min(100, abs(net_score) * 20)
        else:
            recommendation = "Hold"
            confidence = 50
        
        # Calculate position sizing and risk levels
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and pd.notna(df['ATR'].iloc[-1]) else current_price * 0.02
        
        if "Buy" in recommendation:
            entry_price = current_price * 1.002
            stop_loss = current_price - (atr * 2)
            target = current_price + (atr * 6)
        elif "Sell" in recommendation:
            entry_price = current_price * 0.998
            stop_loss = current_price + (atr * 2)
            target = current_price - (atr * 6)
        else:
            entry_price = None
            stop_loss = None
            target = None
        
        # Position sizing
        if stop_loss:
            risk_per_share = abs(current_price - stop_loss)
            position_size = int((account_size * 0.02) / risk_per_share) if risk_per_share > 0 else 0
        else:
            risk_per_share = None
            position_size = 0
        
        return {
            "Symbol": symbol,
            "Current Price": round(current_price, 2),
            "Entry Price": round(entry_price, 2) if entry_price else None,
            "Stop Loss": round(stop_loss, 2) if stop_loss else None,
            "Target": round(target, 2) if target else None,
            "Recommendation": recommendation,
            "Confidence": confidence,
            "Score": net_score,
            "Buy Signals": signals["buy_signals"],
            "Sell Signals": signals["sell_signals"],
            "Position Size": position_size,
            "Risk Per Share": round(risk_per_share, 2) if risk_per_share else None
        }
        
    except Exception as e:
        logger.error(f"Error in adaptive recommendation for {symbol}: {e}")
        return {
            "Symbol": symbol,
            "Recommendation": "Hold",
            "Reason": f"Error: {str(e)}",
            "Score": 0
        }

# ========================= BATCH PROCESSING =========================

def analyze_batch_stocks(stock_list: List[str], progress_callback: Optional[callable] = None) -> pd.DataFrame:
    """Analyze batch of stocks with progress tracking"""
    
    results = []
    total_stocks = len(stock_list)
    
    for idx, symbol in enumerate(stock_list):
        try:
            # Update progress
            if progress_callback:
                progress = (idx + 1) / total_stocks
                progress_callback(progress)
            
            # Fetch and analyze
            data = fetch_stock_data_with_validation(symbol, period="1mo", interval="1d")
            
            if not data.empty and len(data) >= 20:
                result = adaptive_recommendation_with_monitoring(data, symbol)
                results.append(result)
            else:
                logger.warning(f"Insufficient data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            continue
    
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by score
    results_df = pd.DataFrame(results)
    
    if "Score" in results_df.columns:
        results_df = results_df.sort_values(by="Score", ascending=False)
    
    return results_df

# ========================= SECTORS CONFIGURATION =========================

SECTORS = {
    "Bank": [
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "AXISBANK-EQ"
    ],
    "IT": [
        "TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ"
    ],
    "Pharma": [
        "SUNPHARMA-EQ", "CIPLA-EQ", "DRREDDY-EQ", "LUPIN-EQ", "DIVISLAB-EQ"
    ],
    "Auto": [
        "MARUTI-EQ", "TATAMOTORS-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "HEROMOTOCO-EQ"
    ],
    "FMCG": [
        "HINDUNILVR-EQ", "ITC-EQ", "NESTLEIND-EQ", "BRITANNIA-EQ", "DABUR-EQ"
    ],
    "Metal": [
        "TATASTEEL-EQ", "JSWSTEEL-EQ", "HINDALCO-EQ", "VEDL-EQ", "SAIL-EQ"
    ],
    "Oil & Gas": [
        "RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "GAIL-EQ"
    ],
    "Power": [
        "NTPC-EQ", "POWERGRID-EQ", "ADANIPOWER-EQ", "TATAPOWER-EQ"
    ]
}

# ========================= MAIN APPLICATION =========================

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="StockGenie Pro 2.0",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'alerts_triggered' not in st.session_state:
        st.session_state.alerts_triggered = []
    
    st.title("📊 StockGenie Pro 2.0 - Advanced Trading System")
    st.caption(f"Real-time Analysis | {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Display alerts panel
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔔 Alert Manager")
    
    all_alerts = alert_monitor.get_alerts()
    if all_alerts:
        st.sidebar.info(f"Active Alerts: {len(all_alerts)}")
    else:
        st.sidebar.info("No active alerts")
    
    # Stock selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Stock Selection")
    
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=list(SECTORS.keys()),
        default=["Bank"],
        help="Choose sectors to analyze"
    )
    
    stocks_to_analyze = []
    for sector in selected_sectors:
        stocks_to_analyze.extend(SECTORS.get(sector, []))
    
    # Risk settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Risk Management")
    
    account_size = st.sidebar.number_input(
        "Account Size (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    risk_per_trade = st.sidebar.slider(
        "Risk Per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📈 Analysis", "🏆 Top Picks", "📊 Individual Stock"])
    
    with tab1:
        st.subheader("🔍 Market Analysis")
        
        if st.button("🚀 Run Analysis", use_container_width=True):
            if not stocks_to_analyze:
                st.warning("Please select at least one sector")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Analyzing stocks..."):
                status_text.text("Starting analysis...")
                
                # Analyze stocks
                results_df = analyze_batch_stocks(
                    stocks_to_analyze[:10],  # Limit to 10 for performance
                    progress_callback=lambda p: progress_bar.progress(p)
                )
                
                progress_bar.empty()
                status_text.empty()
                
                if not results_df.empty:
                    st.success(f"✅ Analysis complete! Analyzed {len(results_df)} stocks")
                    
                    # Display results
                    st.dataframe(
                        results_df[['Symbol', 'Current Price', 'Recommendation', 'Score', 'Confidence']],
                        use_container_width=True
                    )
                    
                    # Show detailed analysis for top picks
                    st.subheader("📊 Top Recommendations")
                    
                    top_picks = results_df.nlargest(3, 'Score')
                    
                    for idx, row in top_picks.iterrows():
                        with st.expander(f"📈 {row['Symbol']} - {row['Recommendation']} (Score: {row['Score']})"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Price", f"₹{row['Current Price']}")
                                if row['Entry Price']:
                                    st.metric("Entry Price", f"₹{row['Entry Price']}")
                            
                            with col2:
                                if row['Stop Loss']:
                                    st.metric("Stop Loss", f"₹{row['Stop Loss']}")
                                if row['Target']:
                                    st.metric("Target", f"₹{row['Target']}")
                            
                            with col3:
                                st.metric("Confidence", f"{row['Confidence']}%")
                                if row['Position Size']:
                                    st.metric("Position Size", f"{row['Position Size']} shares")
                            
                            # Show signals
                            if row['Buy Signals']:
                                st.success("**Buy Signals:**")
                                for signal in row['Buy Signals']:
                                    st.write(f"✅ {signal}")
                            
                            if row['Sell Signals']:
                                st.error("**Sell Signals:**")
                                for signal in row['Sell Signals']:
                                    st.write(f"❌ {signal}")
                
                else:
                    st.warning("No analysis results to display")
    
    with tab2:
        st.subheader("🏆 Daily Top Picks")
        
        if st.button("Generate Top Picks", use_container_width=True):
            with st.spinner("Finding top opportunities..."):
                # Get all stocks from selected sectors
                all_stocks = []
                for sector in selected_sectors:
                    all_stocks.extend(SECTORS.get(sector, []))
                
                if all_stocks:
                    # Analyze all stocks
                    results_df = analyze_batch_stocks(all_stocks[:20])  # Limit for performance
                    
                    if not results_df.empty:
                        # Filter for buy recommendations
                        buy_picks = results_df[results_df['Recommendation'].str.contains('Buy', na=False)]
                        
                        if not buy_picks.empty:
                            st.success(f"Found {len(buy_picks)} buy opportunities!")
                            
                            # Display top 5
                            top_5 = buy_picks.nlargest(5, 'Score')
                            
                            for idx, row in top_5.iterrows():
                                score_emoji = "🟢" if row['Score'] > 3 else "🟡" if row['Score'] > 0 else "🔴"
                                
                                st.info(
                                    f"{score_emoji} **{row['Symbol']}** - {row['Recommendation']}\n"
                                    f"Price: ₹{row['Current Price']} | "
                                    f"Target: ₹{row['Target'] if row['Target'] else 'N/A'} | "
                                    f"Stop Loss: ₹{row['Stop Loss'] if row['Stop Loss'] else 'N/A'}"
                                )
                        else:
                            st.warning("No buy opportunities found currently")
                    else:
                        st.warning("No analysis results")
                else:
                    st.warning("Please select sectors first")
    
    with tab3:
        st.subheader("📊 Individual Stock Analysis")
        
        symbol_input = st.text_input(
            "Enter Stock Symbol",
            value="RELIANCE-EQ",
            help="Enter NSE symbol with -EQ suffix (e.g., RELIANCE-EQ)"
        )
        
        if st.button("Analyze Stock", use_container_width=True):
            with st.spinner(f"Analyzing {symbol_input}..."):
                try:
                    # Fetch data
                    data = fetch_stock_data_with_validation(symbol_input, period="1mo", interval="1d")
                    
                    if not data.empty:
                        # Get recommendation
                        recommendation = adaptive_recommendation_with_monitoring(data, symbol_input, account_size)
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"₹{recommendation['Current Price']}")
                        
                        with col2:
                            st.metric("Recommendation", recommendation['Recommendation'])
                        
                        with col3:
                            st.metric("Score", recommendation['Score'])
                        
                        with col4:
                            st.metric("Confidence", f"{recommendation['Confidence']}%")
                        
                        # Risk metrics
                        if recommendation['Stop Loss']:
                            st.subheader("📊 Risk Management")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Entry Price", f"₹{recommendation['Entry Price']}")
                            
                            with col2:
                                st.metric("Stop Loss", f"₹{recommendation['Stop Loss']}")
                            
                            with col3:
                                st.metric("Target", f"₹{recommendation['Target']}")
                            
                            st.info(
                                f"**Position Size:** {recommendation['Position Size']} shares\n"
                                f"**Risk Per Share:** ₹{recommendation['Risk Per Share']}"
                            )
                        
                        # Chart
                        st.subheader("📈 Price Chart")
                        
                        fig = go.Figure()
                        
                        # Add candlestick
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name='OHLC'
                        ))
                        
                        # Add Bollinger Bands if available
                        if 'Upper_Band' in data.columns:
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['Upper_Band'],
                                name='Upper BB',
                                line=dict(color='gray', dash='dash')
                            ))
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['Lower_Band'],
                                name='Lower BB',
                                line=dict(color='gray', dash='dash')
                            ))
                        
                        # Add SMAs
                        for sma in ['SMA_20', 'SMA_50']:
                            if sma in data.columns:
                                fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=data[sma],
                                    name=sma,
                                    line=dict(width=1)
                                ))
                        
                        fig.update_layout(
                            title=f"{symbol_input} - Price Chart",
                            yaxis_title="Price (₹)",
                            xaxis_title="Date",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Indicators subplot
                        if 'RSI' in data.columns:
                            st.subheader("📊 Technical Indicators")
                            
                            fig2 = go.Figure()
                            
                            fig2.add_trace(go.Scatter(
                                x=data.index,
                                y=data['RSI'],
                                name='RSI',
                                line=dict(color='purple')
                            ))
                            
                            # Add RSI levels
                            fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                            
                            fig2.update_layout(
                                title="RSI Indicator",
                                yaxis_title="RSI",
                                xaxis_title="Date",
                                height=300
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    else:
                        st.error(f"No data available for {symbol_input}")
                        
                except Exception as e:
                    st.error(f"Error analyzing {symbol_input}: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This tool is for educational purposes only. "
        "Always do your own research and consult with a financial advisor before making investment decisions."
    )
    st.caption("StockGenie Pro 2.0 | Data may be delayed | Not financial advice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        alert_monitor.stop_monitoring()
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        st.error(f"Application error: {str(e)}")
