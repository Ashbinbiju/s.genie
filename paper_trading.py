"""Automated Paper Trading Engine"""

import time
from datetime import datetime
import logging
from supabase_db import PaperTradeDB
from config_secure import TELEGRAM_ENABLED, TELEGRAM_CHAT_ID, TELEGRAM_BOT_TOKEN
import requests

logger = logging.getLogger(__name__)

class PaperTradingEngine:
    """Automated paper trading with signal-based execution"""
    
    def __init__(self, initial_capital=100000):
        self.db = PaperTradeDB()
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.risk_per_trade = 2  # % of capital
        
    def send_telegram_alert(self, message):
        """Send alert to Telegram"""
        if TELEGRAM_ENABLED == "true":
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML"
                }
                requests.post(url, data=data, timeout=10)
            except Exception as e:
                logger.error(f"Telegram alert failed: {e}")
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk management"""
        risk_amount = self.available_capital * (self.risk_per_trade / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
            
        quantity = int(risk_amount / price_risk)
        return max(1, quantity)  # At least 1 share
    
    def check_and_execute_signal(self, symbol, token, exchange, signal_data, current_price):
        """
        Check signal and execute paper trade
        
        signal_data: {
            'buy_signal': bool,
            'sell_signal': bool,
            'buy_strength': int,
            'sell_strength': int,
            'buy_quality': int,
            'sell_quality': int,
            'entry_price': float,
            'stop_loss': float,
            'take_profit': float,
            'rsi': float,
            'trend': str,
        }
        """
        if not self.db.is_connected():
            logger.warning("Database not connected, skipping trade execution")
            return None
        
        # Check for existing open trade
        open_trades = self.db.get_open_trades(symbol)
        
        # BUY Signal Logic
        if signal_data.get('buy_signal') and not open_trades:
            strength = signal_data.get('buy_strength', 0)
            quality = signal_data.get('buy_quality', 0)
            
            # Only trade high-quality signals
            if strength >= 3 and quality >= 2:
                entry = signal_data['entry_price']
                sl = signal_data['stop_loss']
                tp = signal_data['take_profit']
                
                quantity = self.calculate_position_size(entry, sl)
                cost = entry * quantity
                
                if cost <= self.available_capital:
                    trade_data = {
                        'symbol': symbol,
                        'token': token,
                        'exchange': exchange,
                        'signal_type': 'BUY',
                        'entry_price': entry,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'quantity': quantity,
                        'signal_strength': strength,
                        'signal_quality': quality,
                        'rsi': signal_data.get('rsi', 0),
                        'trend': signal_data.get('trend', 'Unknown'),
                        'status': 'OPEN',
                        'entry_time': datetime.now().isoformat()
                    }
                    
                    result = self.db.create_trade(trade_data)
                    if result:
                        self.available_capital -= cost
                        
                        # Send Telegram alert
                        message = f"""
🟢 <b>BUY Signal Executed (Paper Trade)</b>

📊 Symbol: {symbol}
💰 Entry: ₹{entry:.2f}
🛑 Stop Loss: ₹{sl:.2f}
🎯 Take Profit: ₹{tp:.2f}
📦 Quantity: {quantity}
💵 Cost: ₹{cost:.2f}

⚡ Strength: {strength}/4
⭐ Quality: {quality}/4
📈 RSI: {signal_data.get('rsi', 0):.1f}
🎯 Trend: {signal_data.get('trend', 'Unknown')}

Risk:Reward = 1:{abs((tp-entry)/(entry-sl)):.2f}
                        """
                        self.send_telegram_alert(message.strip())
                        logger.info(f"BUY trade executed for {symbol} at ₹{entry}")
                        return result
        
        # SELL Signal Logic (Exit existing position)
        elif signal_data.get('sell_signal') and open_trades:
            for trade in open_trades:
                if trade['signal_type'] == 'BUY':  # Close long position
                    strength = signal_data.get('sell_strength', 0)
                    quality = signal_data.get('sell_quality', 0)
                    
                    if strength >= 2 and quality >= 2:
                        result = self.db.close_trade(
                            trade['id'],
                            current_price,
                            datetime.now().isoformat()
                        )
                        
                        if result:
                            pnl = result['profit_loss']
                            pnl_pct = result['profit_loss_percent']
                            self.available_capital += (trade['entry_price'] * trade['quantity']) + pnl
                            
                            # Send Telegram alert
                            emoji = "🟢" if pnl > 0 else "🔴"
                            message = f"""
{emoji} <b>Position CLOSED (Paper Trade)</b>

📊 Symbol: {symbol}
📥 Entry: ₹{trade['entry_price']:.2f}
📤 Exit: ₹{current_price:.2f}
📦 Quantity: {trade['quantity']}

💰 P&L: ₹{pnl:.2f} ({pnl_pct:+.2f}%)
💵 Capital: ₹{self.available_capital:.2f}

⚡ Exit Strength: {strength}/4
⭐ Exit Quality: {quality}/4
                            """
                            self.send_telegram_alert(message.strip())
                            logger.info(f"Position closed for {symbol} at ₹{current_price}, P&L: ₹{pnl:.2f}")
                            return result
        
        return None
    
    def check_stop_loss_take_profit(self, current_price_dict):
        """
        Check open trades for SL/TP hits
        
        current_price_dict: {'SYMBOL': current_price, ...}
        """
        if not self.db.is_connected():
            return
        
        open_trades = self.db.get_open_trades()
        
        for trade in open_trades:
            symbol = trade['symbol']
            if symbol not in current_price_dict:
                continue
            
            current_price = current_price_dict[symbol]
            entry = trade['entry_price']
            sl = trade['stop_loss']
            tp = trade['take_profit']
            signal_type = trade['signal_type']
            
            hit_reason = None
            
            if signal_type == 'BUY':
                if current_price <= sl:
                    hit_reason = "Stop Loss Hit"
                elif current_price >= tp:
                    hit_reason = "Take Profit Hit"
            elif signal_type == 'SELL':
                if current_price >= sl:
                    hit_reason = "Stop Loss Hit"
                elif current_price <= tp:
                    hit_reason = "Take Profit Hit"
            
            if hit_reason:
                result = self.db.close_trade(
                    trade['id'],
                    current_price,
                    datetime.now().isoformat()
                )
                
                if result:
                    pnl = result['profit_loss']
                    pnl_pct = result['profit_loss_percent']
                    self.available_capital += (entry * trade['quantity']) + pnl
                    
                    # Send Telegram alert
                    emoji = "🎯" if "Take Profit" in hit_reason else "🛑"
                    color_emoji = "🟢" if pnl > 0 else "🔴"
                    message = f"""
{emoji} <b>{hit_reason} (Paper Trade)</b>

📊 Symbol: {symbol}
📥 Entry: ₹{entry:.2f}
📤 Exit: ₹{current_price:.2f}
📦 Quantity: {trade['quantity']}

{color_emoji} P&L: ₹{pnl:.2f} ({pnl_pct:+.2f}%)
💵 Capital: ₹{self.available_capital:.2f}
                    """
                    self.send_telegram_alert(message.strip())
                    logger.info(f"{hit_reason} for {symbol} at ₹{current_price}, P&L: ₹{pnl:.2f}")
    
    def get_account_summary(self):
        """Get current account summary"""
        stats = self.db.get_performance_stats()
        open_trades = self.db.get_open_trades()
        
        return {
            'initial_capital': self.initial_capital,
            'available_capital': self.available_capital,
            'open_trades_count': len(open_trades),
            'total_trades': stats['total_trades'] if stats else 0,
            'win_rate': stats['win_rate'] if stats else 0,
            'total_pnl': stats['total_pnl'] if stats else 0,
        }
