"""Supabase Database Integration for Paper Trading"""

import os
from datetime import datetime
import streamlit as st
try:
    from supabase import create_client, Client
except ImportError:
    pass  # Will be installed

# Supabase Configuration
try:
    SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]
except:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

class PaperTradeDB:
    """Manage paper trading records in Supabase"""
    
    def __init__(self):
        if SUPABASE_URL and SUPABASE_KEY:
            self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        else:
            self.supabase = None
            
    def is_connected(self):
        """Check if database is connected"""
        return self.supabase is not None
    
    def create_trade(self, trade_data):
        """
        Create a new trade record
        
        trade_data: {
            'symbol': str,
            'token': str,
            'exchange': str,
            'signal_type': 'BUY' or 'SELL',
            'entry_price': float,
            'stop_loss': float,
            'take_profit': float,
            'quantity': int,
            'signal_strength': int,
            'signal_quality': int,
            'rsi': float,
            'trend': str,
            'status': 'OPEN',
            'entry_time': datetime
        }
        """
        try:
            if not self.supabase:
                return None
                
            response = self.supabase.table('paper_trades').insert(trade_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error creating trade: {e}")
            return None
    
    def update_trade(self, trade_id, update_data):
        """
        Update an existing trade
        
        update_data: {
            'exit_price': float,
            'exit_time': datetime,
            'status': 'CLOSED',
            'profit_loss': float,
            'profit_loss_percent': float
        }
        """
        try:
            if not self.supabase:
                return None
                
            response = self.supabase.table('paper_trades').update(update_data).eq('id', trade_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error updating trade: {e}")
            return None
    
    def get_open_trades(self, symbol=None):
        """Get all open trades, optionally filtered by symbol"""
        try:
            if not self.supabase:
                return []
                
            query = self.supabase.table('paper_trades').select('*').eq('status', 'OPEN')
            
            if symbol:
                query = query.eq('symbol', symbol)
                
            response = query.execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Error getting open trades: {e}")
            return []
    
    def get_trade_history(self, limit=100, symbol=None):
        """Get trade history, optionally filtered by symbol"""
        try:
            if not self.supabase:
                return []
                
            query = self.supabase.table('paper_trades').select('*').order('entry_time', desc=True).limit(limit)
            
            if symbol:
                query = query.eq('symbol', symbol)
                
            response = query.execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Error getting trade history: {e}")
            return []
    
    def get_performance_stats(self):
        """Get overall performance statistics"""
        try:
            if not self.supabase:
                return None
                
            # Get all closed trades
            response = self.supabase.table('paper_trades').select('*').eq('status', 'CLOSED').execute()
            
            if not response.data:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_profit': 0,
                    'avg_loss': 0
                }
            
            trades = response.data
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
            
            total_pnl = sum(t.get('profit_loss', 0) for t in trades)
            avg_profit = sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.get('profit_loss', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss
            }
        except Exception as e:
            print(f"Error getting performance stats: {e}")
            return None
    
    def close_trade(self, trade_id, exit_price, exit_time=None):
        """Close a trade and calculate P&L"""
        try:
            if not self.supabase:
                return None
                
            # Get trade details
            response = self.supabase.table('paper_trades').select('*').eq('id', trade_id).execute()
            if not response.data:
                return None
                
            trade = response.data[0]
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            signal_type = trade['signal_type']
            
            # Calculate P&L
            if signal_type == 'BUY':
                profit_loss = (exit_price - entry_price) * quantity
            else:  # SELL
                profit_loss = (entry_price - exit_price) * quantity
            
            profit_loss_percent = (profit_loss / (entry_price * quantity)) * 100
            
            # Update trade
            update_data = {
                'exit_price': exit_price,
                'exit_time': exit_time or datetime.now().isoformat(),
                'status': 'CLOSED',
                'profit_loss': profit_loss,
                'profit_loss_percent': profit_loss_percent
            }
            
            return self.update_trade(trade_id, update_data)
        except Exception as e:
            print(f"Error closing trade: {e}")
            return None
