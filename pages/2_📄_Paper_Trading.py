"""Paper Trading Dashboard"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from supabase_db import PaperTradeDB
from paper_trading import PaperTradingEngine

st.set_page_config(
    page_title="Paper Trading Dashboard",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Paper Trading Dashboard")

# Settings file path
SETTINGS_FILE = "paper_trade_settings.json"

# Load settings from file
def load_settings():
    """Load paper trading settings from file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'enabled': False,
        'min_strength': 3,
        'min_quality': 2,
        'risk_per_trade': 2.0
    }

# Save settings to file
def save_settings(settings):
    """Save paper trading settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f)
        return True
    except Exception as e:
        st.error(f"Failed to save settings: {e}")
        return False

# Initialize session state with saved settings
if 'auto_trade_settings' not in st.session_state:
    st.session_state.auto_trade_settings = load_settings()

# Initialize
db = PaperTradeDB()
engine = PaperTradingEngine()

# Check connection
if not db.is_connected():
    st.error("⚠️ Supabase database not connected!")
    st.info("""
    **Setup Required:**
    1. Create a Supabase project at https://supabase.com
    2. Run the SQL schema from `supabase_schema.sql`
    3. Add credentials to Streamlit secrets:
       ```toml
       [supabase]
       SUPABASE_URL = "your-project-url"
       SUPABASE_KEY = "your-anon-key"
       ```
    """)
    st.stop()

st.success("✅ Database Connected")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Account Summary", "📈 Open Positions", "📜 Trade History", "⚙️ Settings"])

with tab1:
    st.subheader("Account Summary")
    
    summary = engine.get_account_summary()
    stats = db.get_performance_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Initial Capital", f"₹{summary['initial_capital']:,.0f}")
        st.metric("Available Capital", f"₹{summary['available_capital']:,.0f}")
    
    with col2:
        total_pnl = summary['total_pnl']
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"{(total_pnl/summary['initial_capital']*100):.2f}%", delta_color=pnl_color)
        st.metric("Open Positions", summary['open_trades_count'])
    
    with col3:
        st.metric("Total Trades", summary['total_trades'])
        st.metric("Win Rate", f"{summary['win_rate']:.1f}%")
    
    with col4:
        if stats:
            st.metric("Winning Trades", stats['winning_trades'])
            st.metric("Losing Trades", stats['losing_trades'])
    
    if stats and stats['total_trades'] > 0:
        st.markdown("---")
        st.subheader("Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Win", f"₹{stats['avg_profit']:,.2f}")
        with col2:
            st.metric("Average Loss", f"₹{stats['avg_loss']:,.2f}")
        with col3:
            profit_factor = abs(stats['avg_profit'] / stats['avg_loss']) if stats['avg_loss'] != 0 else 0
            st.metric("Profit Factor", f"{profit_factor:.2f}")

with tab2:
    st.subheader("📈 Open Positions")
    
    open_trades = db.get_open_trades()
    
    if open_trades:
        df = pd.DataFrame(open_trades)
        
        # Format for display
        display_df = df[[
            'symbol', 'signal_type', 'entry_price', 'stop_loss', 'take_profit',
            'quantity', 'signal_strength', 'signal_quality', 'rsi', 'trend', 'entry_time'
        ]].copy()
        
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['Symbol', 'Type', 'Entry', 'Stop Loss', 'Take Profit', 
                              'Qty', 'Strength', 'Quality', 'RSI', 'Trend', 'Entry Time']
        
        st.dataframe(display_df, use_container_width=True)
        
        st.caption(f"📊 {len(open_trades)} open position(s)")
        
        # Manual close option
        st.markdown("---")
        st.subheader("Manual Close Position")
        
        symbols = [t['symbol'] for t in open_trades]
        selected_symbol = st.selectbox("Select Position", symbols, key="close_symbol")
        exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01, key="exit_price")
        
        if st.button("Close Position", type="primary"):
            trade = next((t for t in open_trades if t['symbol'] == selected_symbol), None)
            if trade and exit_price > 0:
                result = db.close_trade(trade['id'], exit_price)
                if result:
                    st.success(f"✅ Position closed! P&L: ₹{result['profit_loss']:.2f}")
                    st.rerun()
                else:
                    st.error("Failed to close position")
    else:
        st.info("No open positions")

with tab3:
    st.subheader("📜 Trade History")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        filter_symbol = st.selectbox("Filter by Symbol", ["All"] + list(set([t['symbol'] for t in db.get_trade_history()])), key="filter_symbol")
    with col2:
        limit = st.number_input("Limit", min_value=10, max_value=500, value=50, step=10)
    
    history = db.get_trade_history(limit=limit, symbol=None if filter_symbol == "All" else filter_symbol)
    
    if history:
        df = pd.DataFrame(history)
        
        # Format for display
        display_df = df[[
            'symbol', 'signal_type', 'entry_price', 'exit_price', 'quantity',
            'profit_loss', 'profit_loss_percent', 'status', 'entry_time', 'exit_time'
        ]].copy()
        
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = pd.to_datetime(display_df['exit_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
        
        display_df.columns = ['Symbol', 'Type', 'Entry', 'Exit', 'Qty', 
                              'P&L', 'P&L %', 'Status', 'Entry Time', 'Exit Time']
        
        # Color code P&L
        def color_pnl(val):
            if pd.isna(val):
                return ''
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}'
        
        styled_df = display_df.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
        st.dataframe(styled_df, use_container_width=True)
        
        st.caption(f"📊 Showing {len(history)} trade(s)")
        
        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"paper_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trade history")

with tab4:
    st.subheader("⚙️ Paper Trading Settings")
    
    st.markdown("**Automation Settings**")
    
    # Get current settings
    current_settings = st.session_state.auto_trade_settings
    
    auto_trade = st.checkbox(
        "Enable Automated Trading", 
        value=current_settings.get('enabled', False),
        key="auto_trade_checkbox"
    )
    
    if auto_trade:
        st.warning("⚠️ Automated trading is enabled. Signals will be executed automatically based on criteria below.")
        
        min_strength = st.slider(
            "Minimum Signal Strength", 
            0, 4, 
            current_settings.get('min_strength', 3),
            key="strength_slider"
        )
        min_quality = st.slider(
            "Minimum Signal Quality", 
            0, 4, 
            current_settings.get('min_quality', 2),
            key="quality_slider"
        )
        risk_per_trade = st.slider(
            "Risk Per Trade (%)", 
            0.5, 5.0, 
            current_settings.get('risk_per_trade', 2.0), 
            0.5,
            key="risk_slider"
        )
        
        st.info(f"""
        **Current Settings:**
        - Minimum Strength: {min_strength}/4
        - Minimum Quality: {min_quality}/4
        - Risk Per Trade: {risk_per_trade}%
        """)
        
        # Update settings
        new_settings = {
            'enabled': True,
            'min_strength': min_strength,
            'min_quality': min_quality,
            'risk_per_trade': risk_per_trade
        }
        
        # Check if settings changed
        if new_settings != st.session_state.auto_trade_settings:
            st.session_state.auto_trade_settings = new_settings
            if save_settings(new_settings):
                st.success("✅ Settings saved!")
    else:
        st.info("Automated trading is disabled. Signals will only be displayed.")
        new_settings = {
            'enabled': False,
            'min_strength': current_settings.get('min_strength', 3),
            'min_quality': current_settings.get('min_quality', 2),
            'risk_per_trade': current_settings.get('risk_per_trade', 2.0)
        }
        
        # Save disabled state
        if new_settings != st.session_state.auto_trade_settings:
            st.session_state.auto_trade_settings = new_settings
            save_settings(new_settings)
    
    st.markdown("---")
    st.markdown("**Telegram Notifications**")
    
    from config_secure import TELEGRAM_ENABLED, TELEGRAM_CHAT_ID
    
    if TELEGRAM_ENABLED == "true":
        st.success(f"✅ Telegram notifications enabled")
        st.caption(f"Chat ID: {TELEGRAM_CHAT_ID}")
    else:
        st.warning("⚠️ Telegram notifications disabled")
    
    st.markdown("---")
    st.markdown("**Database Connection**")
    st.success("✅ Connected to Supabase")
    
    if st.button("Test Telegram Alert"):
        engine.send_telegram_alert("🧪 <b>Test Alert</b>\n\nPaper Trading System is working correctly!")
        st.success("Alert sent! Check your Telegram.")

st.markdown("---")
st.caption("💡 Paper trading simulates real trades without actual money. Perfect for testing strategies!")
