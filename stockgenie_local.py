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
from pydantic.dataclasses import dataclass as pydantic_dataclass
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

class TechnicalIndicators(BaseModel):
    """Validated technical indicators"""
    rsi: Optional[float] = Field(None, ge=0, le=100)
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    adx: Optional[float] = Field(None, ge=0, le=100)
    atr: Optional[float] = Field(None, ge=0)
    bollinger_upper: Optional[float] = Field(None, gt=0)
    bollinger_lower: Optional[float] = Field(None, gt=0)
    volume_ratio: Optional[float] = Field(None, ge=0)
    
    @validator('rsi')
    def validate_rsi(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('RSI must be between 0 and 100')
        return v

class AlertCondition(BaseModel):
    """Alert condition model"""
    symbol: str
    condition_type: str = Field(..., pattern="^(above|below|crossover|divergence)$")
    price: Optional[float] = Field(None, gt=0)
    indicator: Optional[str] = None
    indicator_value: Optional[float] = None
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    triggered: bool = False
    triggered_at: Optional[datetime] = None

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
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Thread-safe set operation with TTL support"""
        with self._lock:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self._max_size:
                # Remove least recently used item
                lru_key = min(self._access_count, key=self._access_count.get)
                del self._cache[lru_key]
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
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, value in self._cache.items()
                if value.get('expires_at') and value['expires_at'] < current_time
            ]
            for key in expired_keys:
                self.delete(key)

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
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif e.response.status_code == 401:
                raise APIError(f"Authentication failed: {e}")
            elif e.response.status_code >= 500:
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

# ========================= PERFORMANCE OPTIMIZATIONS =========================

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

def vectorized_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Vectorized Bollinger Bands calculation"""
    middle = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def vectorized_multiple_smas(prices: pd.Series, windows: List[int]) -> pd.DataFrame:
    """Vectorized calculation of multiple SMAs"""
    result = pd.DataFrame(index=prices.index)
    for window in windows:
        result[f'SMA_{window}'] = prices.rolling(window=window, min_periods=1).mean()
    return result

# ========================= MONITORING & ALERTS SYSTEM =========================

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
            self.callback(self, current_value)
        
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
    """Alert monitoring system"""
    
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
            if symbol not in self.alerts:
                return triggered_alerts
            
            current_price = data['Close'].iloc[-1] if not data.empty else None
            current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and not data.empty else None
            current_volume = data['Volume'].iloc[-1] if 'Volume' in data.columns and not data.empty else None
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in data.columns and len(data) >= 20 else None
            
            for alert in self.alerts[symbol]:
                if alert.triggered and alert.trigger_count >= 3:
                    continue  # Skip if already triggered multiple times
                
                should_trigger = False
                current_value = None
                
                if alert.alert_type in [AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW] and current_price:
                    current_value = current_price
                    should_trigger = alert.check_condition(current_price)
                
                elif alert.alert_type in [AlertType.RSI_OVERSOLD, AlertType.RSI_OVERBOUGHT] and current_rsi:
                    current_value = current_rsi
                    should_trigger = alert.check_condition(current_rsi)
                
                elif alert.alert_type == AlertType.VOLUME_SPIKE and current_volume and avg_volume:
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
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, args=(check_interval,))
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
    
    def _monitor_loop(self, check_interval: int):
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                symbols_to_check = list(self.alerts.keys())
                
                for symbol in symbols_to_check:
                    if self._stop_event.is_set():
                        break
                    
                    try:
                        # Fetch latest data
                        data = fetch_stock_data_with_validation(symbol, period="1d", interval="5m")
                        if not data.empty:
                            self.check_alerts(symbol, data)
                    except Exception as e:
                        logger.error(f"Error checking alerts for {symbol}: {e}")
                
                # Wait for next check
                self._stop_event.wait(check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

# Global alert monitor instance
alert_monitor = AlertMonitor()

# ========================= ENHANCED DATA FETCHING =========================

# API Configuration
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
    """Validate and clean OHLC data"""
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
    
    # Handle outliers
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = winsorize(df[col], limits=[0.001, 0.001])
    
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
    
    if cached_data and isinstance(cached_data, dict):
        data = cached_data.get('value')
        if isinstance(data, pd.DataFrame) and not data.empty:
            logger.debug(f"Cache hit for {symbol}")
            return data
    
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
        raise

# ========================= OPTIMIZED TECHNICAL INDICATORS =========================

def compute_indicators_optimized(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Optimized indicator computation with performance enhancements"""
    
    if df.empty or 'Close' not in df.columns:
        return df
    
    df = df.copy()
    
    try:
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        
        # Use optimized functions when available
        if config.USE_NUMBA and len(close_prices) >= 14:
            # Optimized RSI
            df['RSI'] = calculate_rsi_optimized(close_prices, 14)
            
            # Optimized EMAs for MACD
            if len(close_prices) >= 26:
                ema_12 = calculate_ema_optimized(close_prices, 12)
                ema_26 = calculate_ema_optimized(close_prices, 26)
                macd_line = ema_12 - ema_26
                macd_signal = calculate_ema_optimized(macd_line[~np.isnan(macd_line)], 9)
                
                df['MACD'] = macd_line
                df['MACD_signal'] = np.nan
                df.loc[df.index[26+8:], 'MACD_signal'] = macd_signal
        else:
            # Fallback to pandas/ta
            if len(df) >= 14:
                df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            if len(df) >= 26:
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
        
        # Vectorized operations for multiple indicators
        if config.VECTORIZE_OPERATIONS:
            # Multiple SMAs at once
            sma_windows = [20, 50, 200]
            sma_df = vectorized_multiple_smas(df['Close'], sma_windows)
            for col in sma_df.columns:
                df[col] = sma_df[col]
            
            # Bollinger Bands
            if len(df) >= 20:
                upper, middle, lower = vectorized_bollinger_bands(df['Close'], 20, 2)
                df['Upper_Band'] = upper
                df['Middle_Band'] = middle
                df['Lower_Band'] = lower
        else:
            # Standard calculation
            for window in [20, 50, 200]:
                if len(df) >= window:
                    df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
            
            if len(df) >= 20:
                bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
                df['Upper_Band'] = bb.bollinger_hband()
                df['Middle_Band'] = bb.bollinger_mavg()
                df['Lower_Band'] = bb.bollinger_lband()
        
        # ATR and ADX
        if len(df) >= 14:
            df['ATR'] = ta.volatility.AverageTrueRange(
                df['High'], df['Low'], df['Close'], window=14
            ).average_true_range()
            
            df['ADX'] = ta.trend.ADXIndicator(
                df['High'], df['Low'], df['Close'], window=14
            ).adx()
        
        # Volume indicators
        if 'Volume' in df.columns and len(df) >= 20:
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
                df['High'], df['Low'], df['Close'], df['Volume'], window=20
            ).chaikin_money_flow()
        
        # Stochastic
        if len(df) >= 14:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
        
        # VWAP
        if 'Volume' in df.columns:
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        logger.debug(f"Computed indicators for {symbol if symbol else 'unknown'}")
        
    except Exception as e:
        logger.error(f"Error computing indicators: {e}")
    
    return df

# ========================= RECOMMENDATION ENGINE WITH ALERTS =========================

def generate_signals_with_alerts(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Generate trading signals and check for alert conditions"""
    
    signals = {
        "buy_signals": [],
        "sell_signals": [],
        "alerts": []
    }
    
    if df.empty or len(df) < 20:
        return signals
    
    try:
        current_price = df['Close'].iloc[-1]
        
        # Price action signals
        if 'Lower_Band' in df.columns and current_price <= df['Lower_Band'].iloc[-1]:
            signals["buy_signals"].append("Bollinger Band Oversold")
            signals["alerts"].append(Alert(
                symbol, AlertType.PRICE_BELOW, 
                df['Lower_Band'].iloc[-1],
                "Price below Bollinger Lower Band"
            ))
        
        if 'Upper_Band' in df.columns and current_price >= df['Upper_Band'].iloc[-1]:
            signals["sell_signals"].append("Bollinger Band Overbought")
            signals["alerts"].append(Alert(
                symbol, AlertType.PRICE_ABOVE,
                df['Upper_Band'].iloc[-1],
                "Price above Bollinger Upper Band"
            ))
        
        # RSI signals
        if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]):
            rsi = df['RSI'].iloc[-1]
            if rsi < 30:
                signals["buy_signals"].append("RSI Oversold")
                signals["alerts"].append(Alert(
                    symbol, AlertType.RSI_OVERSOLD, 30,
                    f"RSI oversold at {rsi:.1f}"
                ))
            elif rsi > 70:
                signals["sell_signals"].append("RSI Overbought")
                signals["alerts"].append(Alert(
                    symbol, AlertType.RSI_OVERBOUGHT, 70,
                    f"RSI overbought at {rsi:.1f}"
                ))
        
        # Volume spike
        if 'Volume_Ratio' in df.columns and pd.notna(df['Volume_Ratio'].iloc[-1]):
            if df['Volume_Ratio'].iloc[-1] > 2.0:
                signals["alerts"].append(Alert(
                    symbol, AlertType.VOLUME_SPIKE, 2.0,
                    f"Volume spike detected ({df['Volume_Ratio'].iloc[-1]:.1f}x average)"
                ))
        
        # MACD crossover
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_signal'].iloc[-1]
            prev_macd = df['MACD'].iloc[-2] if len(df) > 1 else macd
            prev_signal = df['MACD_signal'].iloc[-2] if len(df) > 1 else signal
            
            if pd.notna(macd) and pd.notna(signal):
                if macd > signal and prev_macd <= prev_signal:
                    signals["buy_signals"].append("MACD Bullish Crossover")
                elif macd < signal and prev_macd >= prev_signal:
                    signals["sell_signals"].append("MACD Bearish Crossover")
        
        # Breakout detection
        if len(df) >= 20:
            recent_high = df['High'].iloc[-20:-1].max()
            recent_low = df['Low'].iloc[-20:-1].min()
            
            if current_price > recent_high:
                signals["buy_signals"].append("Breakout Above Resistance")
                signals["alerts"].append(Alert(
                    symbol, AlertType.BREAKOUT, recent_high,
                    f"Breakout above {recent_high:.2f}"
                ))
            elif current_price < recent_low:
                signals["sell_signals"].append("Breakdown Below Support")
        
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")
    
    return signals

def adaptive_recommendation_with_monitoring(df: pd.DataFrame, symbol: str, 
                                           account_size: float = 30000) -> Dict[str, Any]:
    """Enhanced adaptive recommendation with alert monitoring"""
    
    try:
        # Compute optimized indicators
        df = compute_indicators_optimized(df, symbol)
        
        if len(df) < 50:
            return {
                "Recommendation": "Hold",
                "Reason": "Insufficient data",
                "Alerts": []
            }
        
        # Get current values
        current_price = df['Close'].iloc[-1]
        
        # Generate signals and alerts
        signals = generate_signals_with_alerts(df, symbol)
        
        # Check existing alerts
        triggered_alerts = alert_monitor.check_alerts(symbol, df)
        
        # Calculate scores
        buy_score = len(signals["buy_signals"])
        sell_score = len(signals["sell_signals"])
        net_score = buy_score - sell_score
        
        # Generate recommendation
        if net_score >= 2:
            recommendation = "Buy"
            confidence = min(100, net_score * 20)
        elif net_score <= -2:
            recommendation = "Sell"
            confidence = min(100, abs(net_score) * 20)
        else:
            recommendation = "Hold"
            confidence = 50
        
        # Calculate position sizing and risk levels
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
        
        if recommendation == "Buy":
            entry_price = current_price * 1.002
            stop_loss = current_price - (atr * 2)
            target = current_price + (atr * 6)
            
            # Add stop loss and target alerts
            stop_alert = Alert(
                symbol, AlertType.STOP_LOSS_HIT, stop_loss,
                f"Stop loss alert at {stop_loss:.2f}"
            )
            target_alert = Alert(
                symbol, AlertType.TARGET_REACHED, target,
                f"Target reached at {target:.2f}"
            )
            
            alert_monitor.add_alert(stop_alert)
            alert_monitor.add_alert(target_alert)
            
        elif recommendation == "Sell":
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
            position_size = 0
        
        # Add new alerts from signals
        for alert in signals.get("alerts", []):
            alert_monitor.add_alert(alert)
        
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
            "Risk Per Share": round(risk_per_share, 2) if stop_loss else None,
            "Triggered Alerts": [a.message for a in triggered_alerts],
            "Active Alerts": len(alert_monitor.get_alerts(symbol))
        }
        
    except Exception as e:
        logger.error(f"Error in adaptive recommendation for {symbol}: {e}")
        return {
            "Recommendation": "Hold",
            "Reason": f"Error: {str(e)}",
            "Alerts": []
        }

# ========================= BATCH PROCESSING WITH ERROR HANDLING =========================

async def analyze_stock_async(session: aiohttp.ClientSession, symbol: str) -> Optional[Dict[str, Any]]:
    """Async stock analysis for better performance"""
    try:
        # Fetch data
        data = fetch_stock_data_with_validation(symbol, period="1y", interval="1d")
        
        if data.empty or len(data) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        # Analyze with monitoring
        result = adaptive_recommendation_with_monitoring(data, symbol)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

async def analyze_batch_async(symbols: List[str]) -> List[Dict[str, Any]]:
    """Analyze batch of stocks asynchronously"""
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_stock_async(session, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, dict) and result:
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
        
        return valid_results

def analyze_all_stocks_enhanced(stock_list: List[str], batch_size: int = 10,
                              progress_callback: Optional[callable] = None) -> pd.DataFrame:
    """Enhanced batch analysis with async processing"""
    
    all_results = []
    total_stocks = len(stock_list)
    
    for i in range(0, total_stocks, batch_size):
        batch = stock_list[i:i + batch_size]
        
        try:
            # Use async processing for better performance
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            batch_results = loop.run_until_complete(analyze_batch_async(batch))
            loop.close()
            
            all_results.extend(batch_results)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
        
        # Update progress
        if progress_callback:
            progress = min(1.0, (i + len(batch)) / total_stocks)
            progress_callback(progress)
        
        # Rate limiting between batches
        if i + batch_size < total_stocks:
            time.sleep(1)
    
    if not all_results:
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by score
    results_df = pd.DataFrame(all_results)
    
    if "Score" in results_df.columns:
        results_df = results_df.sort_values(by="Score", ascending=False)
    
    # Filter for actionable recommendations
    if "Recommendation" in results_df.columns:
        actionable = results_df[results_df["Recommendation"].isin(["Buy", "Sell"])]
        if not actionable.empty:
            return actionable.head(10)
    
    return results_df.head(10)

# ========================= SECTORS CONFIGURATION =========================

SECTORS = {
    "Bank": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"
    ],
    "IT": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"
    ],
    "Pharma": [
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "LUPIN.NS", "DIVISLAB.NS"
    ],
    "Auto": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"
    ],
    "Metal": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS"
    ],
    "Oil & Gas": [
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS"
    ],
    "Power": [
        "NTPC.NS", "POWERGRID.NS", "ADANIPOWER.NS", "TATAPOWER.NS"
    ]
}

# ========================= ENHANCED UI COMPONENTS =========================

def display_alerts_panel():
    """Display alerts panel in the UI"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔔 Alert Manager")
    
    # Get all active alerts
    all_alerts = alert_monitor.get_alerts()
    
    if all_alerts:
        st.sidebar.info(f"Active Alerts: {len(all_alerts)}")
        
        # Display recent triggered alerts
        if 'alerts_triggered' in st.session_state:
            recent_alerts = st.session_state.alerts_triggered[-5:]
            if recent_alerts:
                st.sidebar.markdown("**Recent Alerts:**")
                for alert in recent_alerts:
                    st.sidebar.warning(f"• {alert['symbol']}: {alert['message'][:50]}...")
    else:
        st.sidebar.info("No active alerts")
    
    # Add new alert form
    with st.sidebar.expander("➕ Add Alert"):
        alert_symbol = st.text_input("Symbol", key="alert_symbol")
        alert_type = st.selectbox(
            "Alert Type",
            ["Price Above", "Price Below", "RSI Oversold", "RSI Overbought", "Volume Spike"],
            key="alert_type"
        )
        alert_threshold = st.number_input("Threshold", min_value=0.0, key="alert_threshold")
        
        if st.button("Add Alert", key="add_alert_btn"):
            if alert_symbol and alert_threshold > 0:
                alert_type_map = {
                    "Price Above": AlertType.PRICE_ABOVE,
                    "Price Below": AlertType.PRICE_BELOW,
                    "RSI Oversold": AlertType.RSI_OVERSOLD,
                    "RSI Overbought": AlertType.RSI_OVERBOUGHT,
                    "Volume Spike": AlertType.VOLUME_SPIKE
                }
                
                new_alert = Alert(
                    alert_symbol,
                    alert_type_map[alert_type],
                    alert_threshold,
                    f"{alert_type} alert at {alert_threshold}"
                )
                
                alert_id = alert_monitor.add_alert(new_alert)
                st.sidebar.success(f"Alert added: {alert_id}")

def display_risk_metrics(recommendations: Dict[str, Any]):
    """Display risk management metrics"""
    st.subheader("📊 Risk Management")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        position_size = recommendations.get("Position Size", 0)
        st.metric("Position Size", f"{position_size} shares")
    
    with col2:
        risk_per_share = recommendations.get("Risk Per Share", 0)
        st.metric("Risk/Share", f"₹{risk_per_share:.2f}" if risk_per_share else "N/A")
    
    with col3:
        total_risk = position_size * risk_per_share if position_size and risk_per_share else 0
        st.metric("Total Risk", f"₹{total_risk:.2f}")
    
    with col4:
        confidence = recommendations.get("Confidence", 0)
        st.metric("Confidence", f"{confidence}%")

def display_signal_details(recommendations: Dict[str, Any]):
    """Display detailed signal information"""
    st.subheader("📈 Signal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Buy Signals:**")
        buy_signals = recommendations.get("Buy Signals", [])
        if buy_signals:
            for signal in buy_signals:
                st.success(f"✓ {signal}")
        else:
            st.info("No buy signals")
    
    with col2:
        st.markdown("**Sell Signals:**")
        sell_signals = recommendations.get("Sell Signals", [])
        if sell_signals:
            for signal in sell_signals:
                st.error(f"✗ {signal}")
        else:
            st.info("No sell signals")

def display_performance_chart(data: pd.DataFrame, symbol: str):
    """Display enhanced performance chart with indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))
    
    # Add Bollinger Bands
    if 'Upper_Band' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Upper_Band'],
            name='Upper BB', line=dict(color='gray', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Lower_Band'],
            name='Lower BB', line=dict(color='gray', dash='dash')
        ))
    
    # Add moving averages
    for ma in ['SMA_20', 'SMA_50', 'SMA_200']:
        if ma in data.columns:
            color = {'SMA_20': 'orange', 'SMA_50': 'blue', 'SMA_200': 'green'}.get(ma)
            fig.add_trace(go.Scatter(
                x=data.index, y=data[ma],
                name=ma, line=dict(color=color)
            ))
    
    fig.update_layout(
        title=f"{symbol} - Price Chart with Indicators",
        yaxis_title="Price (₹)",
        xaxis_title="Date",
        height=600,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig

# ========================= MAIN APPLICATION =========================

def main():
    """Enhanced main application with monitoring and alerts"""
    
    st.set_page_config(
        page_title="StockGenie Pro 2.0",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'alert_monitor_started' not in st.session_state:
        alert_monitor.start_monitoring(check_interval=60)
        st.session_state.alert_monitor_started = True
    
    st.title("📊 StockGenie Pro 2.0 - Advanced Trading System")
    st.caption(f"Real-time Analysis | {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Display alerts panel
    display_alerts_panel()
    
    # Stock selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Stock Selection")
    
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Sector Analysis", "Individual Stock", "Watchlist"],
        help="Choose analysis mode"
    )
    
    if analysis_mode == "Sector Analysis":
        selected_sectors = st.sidebar.multiselect(
            "Select Sectors",
            options=list(SECTORS.keys()),
            default=["Bank", "IT"],
            help="Choose sectors to analyze"
        )
        
        stocks_to_analyze = []
        for sector in selected_sectors:
            stocks_to_analyze.extend(SECTORS.get(sector, []))
        
    elif analysis_mode == "Individual Stock":
        symbol = st.sidebar.text_input(
            "Enter Symbol",
            value="RELIANCE.NS",
            help="Enter NSE symbol (e.g., RELIANCE.NS)"
        )
        stocks_to_analyze = [symbol] if symbol else []
        
    else:  # Watchlist
        watchlist = st.sidebar.text_area(
            "Enter Symbols (one per line)",
            value="RELIANCE.NS\nTCS.NS\nHDFCBANK.NS",
            help="Enter NSE symbols, one per line"
        )
        stocks_to_analyze = [s.strip() for s in watchlist.split('\n') if s.strip()]
    
    # Analysis settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Settings")
    
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1d", "1w", "1mo", "1y", "2y"],
        index=3,
        help="Select analysis timeframe"
    )
    
    interval = st.sidebar.selectbox(
        "Interval",
        ["5m", "15m", "30m", "1h", "1d"],
        index=4,
        help="Select data interval"
    )
    
    # Risk settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Risk Management")
    
    account_size = st.sidebar.number_input(
        "Account Size (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000,
        help="Your trading account size"
    )
    
    risk_per_trade = st.sidebar.slider(
        "Risk Per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Maximum risk per trade as % of account"
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analysis", "🔍 Scanner", "📊 Backtest", "📉 Risk Dashboard"])
    
    with tab1:
        if st.button("🚀 Run Analysis", use_container_width=True):
            if not stocks_to_analyze:
                st.warning("Please select stocks to analyze")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Analyzing stocks..."):
                results_df = analyze_all_stocks_enhanced(
                    stocks_to_analyze,
                    batch_size=5,
                    progress_callback=lambda p: progress_bar.progress(p)
                )
                
                progress_bar.empty()
                status_text.empty()
                
                if not results_df.empty:
                    st.success(f"✅ Analysis complete! Found {len(results_df)} opportunities")
                    
                    # Display results
                    for idx, row in results_df.iterrows():
                        with st.expander(f"📊 {row['Symbol']} - Score: {row.get('Score', 0)}/10"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Price", f"₹{row.get('Current Price', 'N/A')}")
                                st.metric("Recommendation", row.get('Recommendation', 'N/A'))
                            
                            with col2:
                                st.metric("Entry Price", f"₹{row.get('Entry Price', 'N/A')}")
                                st.metric("Stop Loss", f"₹{row.get('Stop Loss', 'N/A')}")
                            
                            with col3:
                                st.metric("Target", f"₹{row.get('Target', 'N/A')}")
                                st.metric("Position Size", f"{row.get('Position Size', 0)} shares")
                            
                            # Display signals
                            display_signal_details(row)
                            
                            # Display risk metrics
                            display_risk_metrics(row)
                            
                            # Chart button
                            if st.button(f"View Chart for {row['Symbol']}", key=f"chart_{idx}"):
                                chart_data = fetch_stock_data_with_validation(
                                    row['Symbol'], period=timeframe, interval=interval
                                )
                                if not chart_data.empty:
                                    chart_data = compute_indicators_optimized(chart_data, row['Symbol'])
                                    fig = display_performance_chart(chart_data, row['Symbol'])
                                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No trading opportunities found with current criteria")
    
    with tab2:
        st.subheader("🔍 Real-time Stock Scanner")
        
        scan_type = st.selectbox(
            "Scan Type",
            ["Breakout Stocks", "Oversold Bounce", "Volume Surge", "Trend Reversal"]
        )
        
        if st.button("Start Scan", use_container_width=True):
            with st.spinner(f"Scanning for {scan_type}..."):
                # Implement specific scans based on type
                scan_results = []
                
                for symbol in stocks_to_analyze[:20]:  # Limit for performance
                    try:
                        data = fetch_stock_data_with_validation(symbol, "1mo", "1d")
                        if not data.empty:
                            data = compute_indicators_optimized(data, symbol)
                            
                            # Check scan criteria
                            if scan_type == "Breakout Stocks":
                                if len(data) >= 20:
                                    recent_high = data['High'].iloc[-20:-1].max()
                                    if data['Close'].iloc[-1] > recent_high:
                                        scan_results.append({
                                            "Symbol": symbol,
                                            "Price": data['Close'].iloc[-1],
                                            "Breakout Level": recent_high,
                                            "Volume Ratio": data.get('Volume_Ratio', pd.Series([1])).iloc[-1]
                                        })
                            
                            elif scan_type == "Oversold Bounce":
                                if 'RSI' in data.columns:
                                    rsi = data['RSI'].iloc[-1]
                                    if pd.notna(rsi) and rsi < 35:
                                        scan_results.append({
                                            "Symbol": symbol,
                                            "Price": data['Close'].iloc[-1],
                                            "RSI": rsi,
                                            "Distance from SMA20": (data['Close'].iloc[-1] / data.get('SMA_20', pd.Series([data['Close'].iloc[-1]])).iloc[-1] - 1) * 100
                                        })
                    
                    except Exception as e:
                        logger.error(f"Scan error for {symbol}: {e}")
                
                if scan_results:
                    scan_df = pd.DataFrame(scan_results)
                    st.dataframe(scan_df, use_container_width=True)
                else:
                    st.info(f"No stocks found matching {scan_type} criteria")
    
    with tab3:
        st.subheader("📊 Strategy Backtesting")
        
        if stocks_to_analyze:
            selected_stock = st.selectbox("Select Stock for Backtest", stocks_to_analyze)
            
            col1, col2 = st.columns(2)
            
            with col1:
                strategy_type = st.selectbox(
                    "Strategy",
                    ["RSI Mean Reversion", "MACD Crossover", "Bollinger Breakout", "Moving Average Cross"]
                )
            
            with col2:
                backtest_period = st.selectbox(
                    "Backtest Period",
                    ["1mo", "3mo", "6mo", "1y", "2y"],
                    index=3
                )
            
            if st.button("Run Backtest", use_container_width=True):
                with st.spinner("Running backtest..."):
                    # Fetch data for backtest
                    backtest_data = fetch_stock_data_with_validation(
                        selected_stock, period=backtest_period, interval="1d"
                    )
                    
                    if not backtest_data.empty:
                        backtest_data = compute_indicators_optimized(backtest_data, selected_stock)
                        
                        # Simple backtest logic (can be expanded)
                        initial_capital = 100000
                        position = 0
                        cash = initial_capital
                        trades = []
                        
                        for i in range(20, len(backtest_data)):
                            current_price = backtest_data['Close'].iloc[i]
                            
                            # Strategy logic
                            if strategy_type == "RSI Mean Reversion":
                                rsi = backtest_data['RSI'].iloc[i] if 'RSI' in backtest_data.columns else 50
                                
                                if pd.notna(rsi):
                                    if rsi < 30 and position == 0:
                                        # Buy signal
                                        shares = int(cash * 0.95 / current_price)
                                        if shares > 0:
                                            position = shares
                                            cash -= shares * current_price * 1.001  # Commission
                                            trades.append({
                                                "Date": backtest_data.index[i],
                                                "Type": "Buy",
                                                "Price": current_price,
                                                "Shares": shares
                                            })
                                    
                                    elif rsi > 70 and position > 0:
                                        # Sell signal
                                        cash += position * current_price * 0.999  # Commission
                                        trades.append({
                                            "Date": backtest_data.index[i],
                                            "Type": "Sell",
                                            "Price": current_price,
                                            "Shares": position
                                        })
                                        position = 0
                        
                        # Calculate final portfolio value
                        final_value = cash + (position * backtest_data['Close'].iloc[-1] if position > 0 else 0)
                        total_return = ((final_value - initial_capital) / initial_capital) * 100
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Return", f"{total_return:.2f}%")
                        
                        with col2:
                            st.metric("Total Trades", len(trades))
                        
                        with col3:
                            st.metric("Final Value", f"₹{final_value:,.2f}")
                        
                        if trades:
                            st.subheader("Trade History")
                            trades_df = pd.DataFrame(trades)
                            st.dataframe(trades_df, use_container_width=True)
    
    with tab4:
        st.subheader("📉 Risk Management Dashboard")
        
        # Portfolio risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Account Size", f"₹{account_size:,}")
        
        with col2:
            max_risk = account_size * (risk_per_trade / 100)
            st.metric("Max Risk/Trade", f"₹{max_risk:,.2f}")
        
        with col3:
            max_positions = min(10, int(100 / risk_per_trade))
            st.metric("Max Positions", max_positions)
        
        with col4:
            total_exposure = min(account_size, max_positions * max_risk * 5)
            st.metric("Max Exposure", f"₹{total_exposure:,}")
        
        # Risk allocation chart
        risk_data = {
            "Category": ["Available", "At Risk", "Reserved"],
            "Amount": [
                account_size * 0.5,
                account_size * 0.3,
                account_size * 0.2
            ]
        }
        
        fig = px.pie(
            risk_data,
            values="Amount",
            names="Category",
            title="Risk Allocation",
            color_discrete_map={
                "Available": "#00cc44",
                "At Risk": "#ff9900",
                "Reserved": "#0066cc"
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Active alerts summary
        st.subheader("🔔 Active Alerts Summary")
        
        active_alerts = alert_monitor.get_alerts()
        if active_alerts:
            alert_summary = []
            for alert in active_alerts[:10]:
                alert_summary.append({
                    "Symbol": alert.symbol,
                    "Type": alert.alert_type.value,
                    "Threshold": alert.threshold,
                    "Created": alert.created_at.strftime("%H:%M:%S"),
                    "Triggered": "Yes" if alert.triggered else "No"
                })
            
            alert_df = pd.DataFrame(alert_summary)
            st.dataframe(alert_df, use_container_width=True)
        else:
            st.info("No active alerts")
    
    # Footer
    st.markdown("---")
    st.caption("StockGenie Pro 2.0 | Real-time Analysis & Alert System | Data may be delayed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        alert_monitor.stop_monitoring()
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        st.error(f"Application error: {str(e)}")
