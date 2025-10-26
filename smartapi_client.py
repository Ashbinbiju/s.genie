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


# Angel One Error Code Mappings
ERROR_CODES = {
    'AG8001': 'Invalid Token - Please login again',
    'AG8002': 'Token Expired - Please login again',
    'AG8003': 'Token Missing - Please login again',
    'AB8050': 'Invalid Refresh Token',
    'AB8051': 'Refresh Token Expired',
    'AB1000': 'Invalid Email or Password',
    'AB1001': 'Invalid Email',
    'AB1002': 'Invalid Password Length',
    'AB1006': 'Client is blocked for trading',
    'AB1007': 'Login failed - Invalid credentials',
    'AB1010': 'Session Expired - Please login again',
    'AB1011': 'Client not logged in',
    'AB1012': 'Invalid Product Type',
    'AB1013': 'Order not found',
    'AB1014': 'Trade not found',
    'AB1015': 'Holding not found',
    'AB1016': 'Position not found',
    'AB1017': 'Position conversion failed',
    'AB2001': 'Internal Error - Please try after sometime',
    'AB2002': 'ROBO order is blocked',
}


class AngelOneAPIError(Exception):
    """Custom exception for Angel One API errors"""
    def __init__(self, error_code, message):
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{error_code}] {message}")


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
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None
        # Rate limiters for different API endpoints
        self.login_limiter = RateLimiter(calls_per_second=1, calls_per_minute=60)
        self.candle_limiter = RateLimiter(calls_per_second=3, calls_per_minute=180)
        self.order_limiter = RateLimiter(calls_per_second=20, calls_per_minute=500)
        self.ltp_limiter = RateLimiter(calls_per_second=10, calls_per_minute=500)
    
    def _handle_response(self, response, operation="API call"):
        """Handle Angel One API response with proper error checking"""
        try:
            # Check if response has status
            if isinstance(response, dict):
                status = response.get('status', False)
                
                # Handle boolean status
                if isinstance(status, bool):
                    if not status:
                        error_code = response.get('errorcode', 'UNKNOWN')
                        message = response.get('message', 'Unknown error')
                        error_desc = ERROR_CODES.get(error_code, message)
                        logger.error(f"❌ {operation} failed: [{error_code}] {error_desc}")
                        raise AngelOneAPIError(error_code, error_desc)
                    return response.get('data')
                
                # Handle string status (some APIs return "true"/"false")
                elif isinstance(status, str):
                    if status.lower() == 'false':
                        error_code = response.get('errorcode', 'UNKNOWN')
                        message = response.get('message', 'Unknown error')
                        error_desc = ERROR_CODES.get(error_code, message)
                        logger.error(f"❌ {operation} failed: [{error_code}] {error_desc}")
                        raise AngelOneAPIError(error_code, error_desc)
                    return response.get('data')
                
                # Status is True or "true"
                return response.get('data')
            
            # If response is not a dict, return as is
            return response
            
        except AngelOneAPIError:
            raise
        except Exception as e:
            logger.error(f"❌ Error handling {operation} response: {e}")
            raise
        
    def login(self):
        """Login to Angel One SmartAPI with proper error handling"""
        try:
            self.login_limiter.wait_if_needed()
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            
            data = self.smart_api.generateSession(
                clientCode=self.client_id,
                password=self.password,
                totp=totp
            )
            
            # Handle response with proper error checking
            session_data = self._handle_response(data, "Login")
            
            if session_data:
                self.jwt_token = session_data.get('jwtToken')
                self.refresh_token = session_data.get('refreshToken')
                self.feed_token = session_data.get('feedToken')
                logger.info("✅ Successfully logged in to Angel One")
                return True
            return False
            
        except AngelOneAPIError as e:
            logger.error(f"❌ Login error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return False
    
    def get_historical_data(self, symbol_token, exchange, interval, from_date, to_date):
        """Fetch historical candle data with rate limiting and error handling"""
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
            
            response = self.smart_api.getCandleData(params)
            data = self._handle_response(response, "Get Candle Data")
            
            if data:
                df = pd.DataFrame(
                    data,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                return df
            return pd.DataFrame()
            
        except AngelOneAPIError as e:
            # Handle specific errors
            if e.error_code in ['AG8001', 'AG8002', 'AG8003', 'AB1010', 'AB1011']:
                logger.warning(f"⚠️ Session expired or invalid. Please re-login.")
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_positions(self):
        """Get current positions with rate limiting and error handling"""
        try:
            # Position API has 1 req/sec limit
            self.login_limiter.wait_if_needed()
            response = self.smart_api.position()
            return self._handle_response(response, "Get Positions")
        except AngelOneAPIError as e:
            logger.error(f"Error getting positions: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return None
    
    def place_order(self, order_params):
        """Place order with rate limiting and error handling"""
        try:
            # Order placement has 20 req/sec, 500/min limit
            self.order_limiter.wait_if_needed()
            response = self.smart_api.placeOrder(order_params)
            order_data = self._handle_response(response, "Place Order")
            
            if order_data:
                order_id = order_data.get('orderid') or order_data.get('uniqueorderid')
                logger.info(f"✅ Order placed: {order_id}")
                return order_id
            return None
            
        except AngelOneAPIError as e:
            # Handle specific order errors
            if e.error_code == 'AB1012':
                logger.error("❌ Invalid product type. Check order parameters.")
            elif e.error_code == 'AB1006':
                logger.error("❌ Client blocked for trading. Contact broker.")
            elif e.error_code == 'AB2002':
                logger.error("❌ ROBO orders are blocked.")
            else:
                logger.error(f"❌ Order failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return None
    
    def get_ltp(self, exchange, trading_symbol, symbol_token):
        """Get Last Traded Price with rate limiting and error handling"""
        try:
            # LTP has 10 req/sec, 500/min limit
            self.ltp_limiter.wait_if_needed()
            params = {
                "exchange": exchange,
                "tradingsymbol": trading_symbol,
                "symboltoken": symbol_token
            }
            response = self.smart_api.ltpData(params)
            return self._handle_response(response, "Get LTP")
        except AngelOneAPIError as e:
            logger.error(f"Error getting LTP: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting LTP: {e}")
            return None
    
    def search_scrip(self, exchange, search_term):
        """Search for scrip/symbol and get token"""
        try:
            # Search has 1 req/sec limit
            self.login_limiter.wait_if_needed()
            params = {
                "exchange": exchange,
                "searchscrip": search_term
            }
            response = self.smart_api.searchScrip(params)
            return self._handle_response(response, "Search Scrip")
        except AngelOneAPIError as e:
            logger.error(f"Error searching scrip: {e}")
            return None
        except Exception as e:
            logger.error(f"Error searching scrip: {e}")
            return None
    
    def get_ltp_data(self, exchange, trading_symbol, symbol_token):
        """Get LTP data for a symbol (alternative to get_ltp with more details)"""
        try:
            self.ltp_limiter.wait_if_needed()
            params = {
                "exchange": exchange,
                "tradingsymbol": trading_symbol,
                "symboltoken": symbol_token
            }
            response = self.smart_api.ltpData(params)
            return self._handle_response(response, "Get LTP Data")
        except AngelOneAPIError as e:
            logger.error(f"Error getting LTP data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting LTP data: {e}")
            return None
    
    def logout(self):
        """Logout from Angel One with proper cleanup"""
        try:
            self.login_limiter.wait_if_needed()
            params = {"clientcode": self.client_id}
            response = self.smart_api.terminateSession(params)
            self._handle_response(response, "Logout")
            
            # Clear tokens
            self.jwt_token = None
            self.refresh_token = None
            self.feed_token = None
            logger.info("✅ Successfully logged out")
            return True
            
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            return False
