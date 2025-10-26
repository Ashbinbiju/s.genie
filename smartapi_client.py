"""SmartAPI client for Angel One integration"""

import pyotp
from SmartApi import SmartConnect
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from collections import deque

# Try to import from secure config first, fall back to regular config
try:
    from config_secure import *
except:
    from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for Angel One API calls"""
    def __init__(self, calls_per_second=3, calls_per_minute=180):
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.second_window = deque()
        self.minute_window = deque()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Clean old entries from second window (keep last 1 second)
        while self.second_window and now - self.second_window[0] > 1.0:
            self.second_window.popleft()
        
        # Clean old entries from minute window (keep last 60 seconds)
        while self.minute_window and now - self.minute_window[0] > 60.0:
            self.minute_window.popleft()
        
        # Check if we need to wait for second limit
        if len(self.second_window) >= self.calls_per_second:
            sleep_time = 1.0 - (now - self.second_window[0])
            if sleep_time > 0:
                logger.info(f"⏳ Rate limit: Waiting {sleep_time:.2f}s")
                time.sleep(sleep_time)
                now = time.time()
                # Clean again after sleep
                while self.second_window and now - self.second_window[0] > 1.0:
                    self.second_window.popleft()
        
        # Check if we need to wait for minute limit
        if len(self.minute_window) >= self.calls_per_minute:
            sleep_time = 60.0 - (now - self.minute_window[0])
            if sleep_time > 0:
                logger.warning(f"⏳ Rate limit: Waiting {sleep_time:.2f}s (minute limit)")
                time.sleep(sleep_time)
                now = time.time()
                # Clean again after sleep
                while self.minute_window and now - self.minute_window[0] > 60.0:
                    self.minute_window.popleft()
        
        # Record this call
        self.second_window.append(now)
        self.minute_window.append(now)


class SmartAPIClient:
    def __init__(self):
        self.client_id = CLIENT_ID
        self.password = PASSWORD
        self.totp_secret = TOTP_SECRET
        self.api_key = TRADING_API_KEY
        self.smart_api = None
        # Rate limiters for different API endpoints
        self.login_limiter = RateLimiter(calls_per_second=1, calls_per_minute=60)
        self.candle_limiter = RateLimiter(calls_per_second=3, calls_per_minute=180)
        self.order_limiter = RateLimiter(calls_per_second=20, calls_per_minute=500)
        self.ltp_limiter = RateLimiter(calls_per_second=10, calls_per_minute=500)
        
    def login(self):
        """Login to Angel One SmartAPI"""
        try:
            self.login_limiter.wait_if_needed()
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            
            data = self.smart_api.generateSession(
                clientCode=self.client_id,
                password=self.password,
                totp=totp
            )
            
            if data['status']:
                logger.info("✅ Successfully logged in to Angel One")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return False
    
    def get_historical_data(self, symbol_token, exchange, interval, from_date, to_date):
        """Fetch historical candle data with rate limiting"""
        try:
            # Apply rate limit for getCandleData endpoint
            self.candle_limiter.wait_if_needed()
            
            params = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            data = self.smart_api.getCandleData(params)
            
            if data['status'] and data['data']:
                df = pd.DataFrame(
                    data['data'],
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_positions(self):
        """Get current positions with rate limiting"""
        try:
            # Position API has 1 req/sec limit
            self.login_limiter.wait_if_needed()  # Reuse 1/sec limiter
            return self.smart_api.position()
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
    
    def place_order(self, order_params):
        """Place order with rate limiting"""
        try:
            # Order placement has 20 req/sec, 500/min limit
            self.order_limiter.wait_if_needed()
            order_id = self.smart_api.placeOrder(order_params)
            logger.info(f"✅ Order placed: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None
    
    def get_ltp(self, exchange, trading_symbol, symbol_token):
        """Get Last Traded Price with rate limiting"""
        try:
            # LTP has 10 req/sec, 500/min limit
            self.ltp_limiter.wait_if_needed()
            params = {
                "exchange": exchange,
                "tradingsymbol": trading_symbol,
                "symboltoken": symbol_token
            }
            return self.smart_api.ltpData(params)
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
