import time
import requests
import random
from typing import Dict, List, Optional, Any
from functools import wraps
from config import (
    ENDPOINTS, CACHE_DURATION, USER_AGENTS,
    SMARTAPI_RATE_LIMITS, TELEGRAM_ENABLED, TELEGRAM_CHAT_ID,
    REDIS_ENABLED, REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_PREFIX
)
from cache_backend import get_cache_backend


def retry_on_failure(max_retries=3, delay=1):
    """Decorator to retry failed API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    raise
                except requests.exceptions.RequestException:
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    raise
            return None
        return wrapper
    return decorator


class APIManager:
    def __init__(self):
        # Initialize cache backend (Redis if enabled, otherwise in-memory)
        self.cache = get_cache_backend(
            use_redis=REDIS_ENABLED,
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            prefix=REDIS_PREFIX
        )
        self.last_request_time = 0
        self.request_count = {"second": 0, "minute": 0, "hour": 0}
        self.time_windows = {
            "second": time.time(),
            "minute": time.time(),
            "hour": time.time()
        }
        # Connection pooling for better performance
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get random user agent headers"""
        return {"User-Agent": random.choice(USER_AGENTS)}
    
    def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        # Reset counters if windows expired
        if current_time - self.time_windows["second"] >= 1:
            self.request_count["second"] = 0
            self.time_windows["second"] = current_time
        
        if current_time - self.time_windows["minute"] >= 60:
            self.request_count["minute"] = 0
            self.time_windows["minute"] = current_time
        
        if current_time - self.time_windows["hour"] >= 3600:
            self.request_count["hour"] = 0
            self.time_windows["hour"] = current_time
        
        # Check limits
        if self.request_count["second"] >= SMARTAPI_RATE_LIMITS["requests_per_second"]:
            sleep_time = 1 - (current_time - self.time_windows["second"])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Enforce minimum delay
        time_since_last = current_time - self.last_request_time
        if time_since_last < SMARTAPI_RATE_LIMITS["min_delay_between_requests"]:
            time.sleep(SMARTAPI_RATE_LIMITS["min_delay_between_requests"] - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count["second"] += 1
        self.request_count["minute"] += 1
        self.request_count["hour"] += 1
    
    def _check_cache(self, key: str, duration: int) -> Optional[Any]:
        """Check if cached data is still valid"""
        return self.cache.get(key)
    
    def _set_cache(self, key: str, data: Any, duration: int = 3600):
        """Store data in cache with TTL"""
        self.cache.set(key, data, ttl=duration)
    
    @retry_on_failure(max_retries=3, delay=1)
    def fetch_market_breadth(self) -> Optional[Dict]:
        """Fetch market breadth data"""
        cache_key = "market_breadth"
        cached = self._check_cache(cache_key, CACHE_DURATION["market_breadth"])
        if cached:
            return cached
        
        try:
            self._rate_limit_check()
            response = self.session.get(
                ENDPOINTS["market_breadth"],
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            self._set_cache(cache_key, data, CACHE_DURATION["market_breadth"])
            return data
        except Exception as e:
            print(f"❌ Error fetching market breadth: {e}")
            return None
    
    def fetch_sector_performance(self) -> Optional[Dict]:
        """Fetch sector indices performance"""
        cache_key = "sector_performance"
        cached = self._check_cache(cache_key, CACHE_DURATION["sector_performance"])
        if cached:
            return cached
        
        try:
            self._rate_limit_check()
            response = requests.get(
                ENDPOINTS["sector_performance"],
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            self._set_cache(cache_key, data, CACHE_DURATION["sector_performance"])
            return data
        except Exception as e:
            print(f"❌ Error fetching sector performance: {e}")
            return None
    
    def fetch_support_resistance(
        self, 
        symbols: List[str], 
        timeframe: str = "5min"
    ) -> Optional[Dict]:
        """Fetch support and resistance levels"""
        try:
            self._rate_limit_check()
            # Remove -EQ suffix if present
            clean_symbols = [s.replace("-EQ", "") for s in symbols]
            nse_symbols = [f"NSE_{s}" for s in clean_symbols]
            
            response = requests.post(
                ENDPOINTS["support_resistance"],
                json={
                    "time_frame": timeframe,
                    "stocks": nse_symbols,
                    "user_broker_id": "ZMS"
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching S/R levels: {e}")
            return None
    
    def fetch_candlestick_data(
        self, 
        symbol: str, 
        timeframe: str = "5min"
    ) -> Optional[List]:
        """Fetch candlestick OHLCV data"""
        try:
            self._rate_limit_check()
            clean_symbol = symbol.replace("-EQ", "")
            
            # Fixed URL construction
            base_url = "https://technicalwidget.streak.tech/api/candles/"
            params = f"?stock=NSE:{clean_symbol}&timeFrame={timeframe}&user_id="
            url = base_url + params
            
            response = requests.get(url, timeout=10)
            
            # Handle 400 errors silently (invalid symbol)
            if response.status_code == 400:
                return None
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                # Symbol doesn't exist in NSE - skip silently
                return None
            print(f"❌ HTTP Error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error fetching candlestick data for {symbol}: {e}")
            return None
    
    def fetch_shareholdings(self, symbol: str) -> Optional[Dict]:
        """Fetch shareholding pattern data from Zerodha"""
        cache_key = f"shareholdings_{symbol}"
        cached = self._check_cache(cache_key, CACHE_DURATION["shareholdings"])
        if cached:
            return cached
        
        try:
            self._rate_limit_check()
            clean_symbol = symbol.replace("-EQ", "")
            url = ENDPOINTS["shareholdings"].format(clean_symbol)
            
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            data = response.json()
            self._set_cache(cache_key, data, CACHE_DURATION["shareholdings"])
            return data
        except Exception as e:
            print(f"❌ Error fetching shareholdings for {symbol}: {e}")
            return None
    
    def fetch_financials(self, symbol: str) -> Optional[Dict]:
        """Fetch financial data (P&L, Balance Sheet, Cash Flow) from Zerodha"""
        cache_key = f"financials_{symbol}"
        cached = self._check_cache(cache_key, CACHE_DURATION["financials"])
        if cached:
            return cached
        
        try:
            self._rate_limit_check()
            clean_symbol = symbol.replace("-EQ", "")
            url = ENDPOINTS["financials"].format(clean_symbol)
            
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            data = response.json()
            self._set_cache(cache_key, data, CACHE_DURATION["financials"])
            return data
        except Exception as e:
            print(f"❌ Error fetching financials for {symbol}: {e}")
            return None
    
    def fetch_technical_analysis(self, symbol: str, timeframe: str = "5min") -> Optional[Dict]:
        """Fetch comprehensive technical analysis from Streak
        
        Supported timeframes: 1min, 3min, 5min, 10min, 15min, 30min, hour, day
        
        Returns indicators: RSI, MACD, ADX, CCI, Stochastic, etc.
        """
        cache_key = f"tech_analysis_{symbol}_{timeframe}"
        cached = self._check_cache(cache_key, CACHE_DURATION["technical_analysis"])
        if cached:
            return cached
        
        try:
            self._rate_limit_check()
            clean_symbol = symbol.replace("-EQ", "")
            
            # Construct URL with query parameters
            url = ENDPOINTS["technical_analysis"]
            params = {
                "timeFrame": timeframe,
                "stock": f"NSE:{clean_symbol}",
                "user_id": ""
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            # Handle 400 errors silently (invalid symbol)
            if response.status_code == 400:
                return None
            
            response.raise_for_status()
            data = response.json()
            self._set_cache(cache_key, data, CACHE_DURATION["technical_analysis"])
            return data
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                # Symbol doesn't exist in NSE - skip silently
                return None
            print(f"❌ HTTP Error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error fetching technical analysis for {symbol}: {e}")
            return None

    
    def send_telegram_alert(self, message: str) -> bool:
        """Send alert via Telegram"""
        if not TELEGRAM_ENABLED:
            return False
        
        try:
            response = requests.post(
                ENDPOINTS["telegram"],
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                },
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Error sending Telegram alert: {e}")
            return False
