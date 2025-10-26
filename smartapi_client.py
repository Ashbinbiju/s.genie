"""SmartAPI client for Angel One integration"""

import pyotp
from SmartApi import SmartConnect
import pandas as pd
from datetime import datetime, timedelta
import logging

# Try to import from secure config first, fall back to regular config
try:
    from config_secure import *
except:
    from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartAPIClient:
    def __init__(self):
        self.client_id = CLIENT_ID
        self.password = PASSWORD
        self.totp_secret = TOTP_SECRET
        self.api_key = TRADING_API_KEY
        self.smart_api = None
        
    def login(self):
        """Login to Angel One SmartAPI"""
        try:
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
        """Fetch historical candle data"""
        try:
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
        """Get current positions"""
        try:
            return self.smart_api.position()
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
    
    def place_order(self, order_params):
        """Place order"""
        try:
            order_id = self.smart_api.placeOrder(order_params)
            logger.info(f"✅ Order placed: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None
