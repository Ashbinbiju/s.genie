"""Streamlit Dashboard for Intraday Trading System"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from smartapi_client import SmartAPIClient
from indicators import calculate_indicators, generate_signals
from stock_database import STOCKS_BY_SECTOR

# Try to import from secure config first, fall back to regular config
try:
    from config_secure import TRADING_CONFIG, STOCK_SYMBOLS, TIMEFRAME_MAP
except:
    from config import TRADING_CONFIG, STOCK_SYMBOLS, TIMEFRAME_MAP

# Page config
st.set_page_config(
    page_title="Advanced Intraday Trading System",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'smart_api' not in st.session_state:
    st.session_state.smart_api = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = []


def login_to_angel():
    """Login to Angel One SmartAPI"""
    with st.spinner("🔐 Logging in to Angel One..."):
        client = SmartAPIClient()
        if client.login():
            st.session_state.smart_api = client
            st.session_state.logged_in = True
            st.success("✅ Successfully logged in!")
            return True
        else:
            st.error("❌ Login failed")
            return False


def fetch_market_data(symbol, token, exchange, timeframe, days=5):
    """Fetch historical data with proper date range and error handling"""
    try:
        client = st.session_state.smart_api
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
        interval = TIMEFRAME_MAP.get(timeframe, 'THIRTY_MINUTE')
        
        df = client.get_historical_data(token, exchange, interval, from_date, to_date)
        
        if not df.empty:
            # Keep only today's data for intraday
            today = datetime.now().date()
            df = df[df.index.date == today] if not df.empty else df
        
        return df
    except Exception as e:
        # Check if session expired
        error_msg = str(e)
        if any(code in error_msg for code in ['AG8001', 'AG8002', 'AG8003', 'AB1010', 'AB1011']):
            st.session_state.logged_in = False
            st.error("❌ Session expired. Please login again.")
            st.rerun()
        else:
            st.error(f"Error: {e}")
        return pd.DataFrame()


def create_chart(df, symbol):
    """Create candlestick chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} - Price', 'RSI', 'MACD')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], name='EMA 9', line=dict(color='green', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_medium'], name='EMA 21', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], name='EMA 50', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], name='VWAP', line=dict(color='blue', width=2, dash='dot')), row=1, col=1)
    
    # Buy/Sell signals
    buy_signals = df[df['buy_signal']]
    sell_signals = df[df['sell_signal']]
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['low'] * 0.995,
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='green'),
            name='Buy'
        ), row=1, col=1)
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['high'] * 1.005,
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Sell'
        ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_line'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Hist', marker_color='gray'), row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, xaxis_rangeslider_visible=False)
    return fig


# Main UI
st.title("📈 Advanced Intraday Trading System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    if not st.session_state.logged_in:
        if st.button("🔐 Login to Angel One", type="primary"):
            login_to_angel()
    else:
        st.success("✅ Connected to Angel One")
        if st.button("🚪 Logout", type="secondary", use_container_width=True, key="logout_btn"):
            client = st.session_state.smart_api
            if client and client.logout():
                st.session_state.logged_in = False
                st.session_state.smart_api = None
                st.rerun()
    
    st.markdown("---")
    
    # Wishlist and Scan Mode
    scan_mode = st.radio("📊 Trading Mode", ["Single Stock", "Wishlist Scan"], horizontal=True)
    
    if scan_mode == "Single Stock":
        selected_symbol = st.selectbox("📊 Select Stock", list(STOCK_SYMBOLS.keys()))
    else:
        st.markdown("**📋 My Wishlist**")
        if st.session_state.wishlist:
            for i, stock in enumerate(st.session_state.wishlist):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"📊 {stock['name']}")
                with col2:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.wishlist.pop(i)
                        st.rerun()
        else:
            st.info("📝 Wishlist empty - Use quick add below")
        
        # QUICK ADD STOCKS BY SECTOR
        st.markdown("---")
        st.markdown("**⚡ Quick Add Stocks by Sector**")
        
        # Sector selection
        selected_sector = st.selectbox(
            "Choose Sector",
            list(STOCKS_BY_SECTOR.keys()),
            key="sector_select"
        )
        
        # Show stocks from selected sector
        sector_stocks = STOCKS_BY_SECTOR[selected_sector]
        
        # Display as compact grid (3 columns for better fit)
        st.caption(f"📊 {len(sector_stocks)} stocks in {selected_sector}")
        
        # Add All button for the sector
        if st.button(f"➕ Add All {selected_sector}", use_container_width=True, key="add_all_sector"):
            added_count = 0
            for symbol, token in sector_stocks.items():
                if not any(s['token'] == token for s in st.session_state.wishlist):
                    st.session_state.wishlist.append({
                        'name': symbol,
                        'token': token,
                        'exchange': 'NSE'
                    })
                    added_count += 1
            if added_count > 0:
                st.success(f"✅ Added {added_count} stocks from {selected_sector}!")
                st.rerun()
            else:
                st.info("All stocks from this sector are already in wishlist")
        
        # Show individual stock buttons
        cols = st.columns(3)
        for idx, (symbol, token) in enumerate(sector_stocks.items()):
            with cols[idx % 3]:
                if st.button(f"➕ {symbol}", key=f"quick_add_{symbol}_{token}", use_container_width=True):
                    if not any(s['token'] == token for s in st.session_state.wishlist):
                        st.session_state.wishlist.append({
                            'name': symbol,
                            'token': token,
                            'exchange': 'NSE'
                        })
                        st.success(f"✅ Added!")
                        st.rerun()

    
    timeframe = st.selectbox("⏰ Timeframe", ['5min', '15min', '30min'], index=2)
    
    st.markdown("---")
    st.markdown("**🔄 Auto Refresh**")
    auto_refresh = st.checkbox("Enable Auto Refresh")
    
    if auto_refresh:
        refresh_interval = st.slider("Interval (seconds)", 30, 300, 60)
    
    st.markdown("---")
    st.markdown("**⚙️ Debug**")
    show_debug = st.checkbox("Show Debug Info")
    
    st.markdown("---")
    st.markdown("**💡 Tips**")
    st.caption("• Refresh both dashboards to sync")
    st.caption("• Check same timeframe (30min)")
    st.caption("• VWAP resets daily at 9:15 AM")
    st.caption("• Data lag: 1-5 seconds normal")
    
    # Add search feature (optional - for other stocks)
    if st.session_state.logged_in and scan_mode == "Wishlist Scan":
        st.markdown("---")
        st.markdown("**🔍 Search Other Stocks**")
        
        with st.expander("Advanced Search"):
            search_term = st.text_input("Search symbol", placeholder="e.g., WIPRO", key="search_input")
            search_exchange = st.selectbox("Exchange", ["NSE", "BSE"], key="search_exchange")
            
            if st.button("🔍 Search", key="search_btn", use_container_width=True):
                if search_term:
                    with st.spinner(f"Searching..."):
                        try:
                            client = st.session_state.smart_api
                            if hasattr(client, 'search_scrip'):
                                results = client.search_scrip(search_exchange, search_term.upper())
                                
                                if results:
                                    for r in results[:5]:
                                        symbol = r.get('tradingsymbol', '')
                                        token = r.get('symboltoken', '')
                                        if symbol.endswith('-EQ'):
                                            col1, col2 = st.columns([3, 1])
                                            with col1:
                                                st.caption(f"📊 {symbol}")
                                            with col2:
                                                if st.button("➕", key=f"add_{token}"):
                                                    if not any(s['token'] == token for s in st.session_state.wishlist):
                                                        st.session_state.wishlist.append({
                                                            'name': symbol.replace('-EQ', ''),
                                                            'token': token,
                                                            'exchange': search_exchange
                                                        })
                                                        st.rerun()
                                else:
                                    st.warning("No results found")
                        except Exception as e:
                            st.caption(f"⚠️ {str(e)}")


# Main content
if not st.session_state.logged_in:
    st.warning("⚠️ Please login to view data")
    st.stop()

# Handle Single Stock vs Wishlist Scan mode
if scan_mode == "Single Stock":
    # Original single stock view
    symbol_info = STOCK_SYMBOLS[selected_symbol]
    
    # Show debug info if enabled
    if show_debug:
        with st.sidebar:
            st.caption(f"Token: {symbol_info['token']}")
            st.caption(f"Exchange: {symbol_info['exchange']}")
            st.caption(f"API Interval: {TIMEFRAME_MAP.get(timeframe)}")
            
            # Rate limit stats
            if st.session_state.smart_api:
                client = st.session_state.smart_api
                st.markdown("**🚦 Rate Limits:**")
                st.caption(f"Candle: {len(client.candle_limiter.second_window)}/3 per sec")
                st.caption(f"Candle: {len(client.candle_limiter.minute_window)}/180 per min")
    
    # Add data freshness indicator
    col_refresh1, col_refresh2 = st.columns([3, 1])
    with col_refresh1:
        with st.spinner(f"📊 Fetching {selected_symbol}..."):
            df = fetch_market_data(selected_symbol, symbol_info['token'], symbol_info['exchange'], timeframe, days=5)
    with col_refresh2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    if df.empty:
        st.error("❌ No data available. Market might be closed or data unavailable.")
        st.info(f"💡 Try: Different timeframe, Check market hours (9:20-15:30 IST), Or select another stock")
        st.stop()
    
    # Show data info
    st.caption(f"📊 Data: {len(df)} candles | Last updated: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')} IST")
    
    # Fetch and show LTP for comparison
    if show_debug:
        try:
            client = st.session_state.smart_api
            ltp_data = client.get_ltp_data(
                symbol_info['exchange'], 
                selected_symbol + '-EQ',
                symbol_info['token']
            )
            if ltp_data:
                st.caption(f"🔴 Live LTP: ₹{ltp_data.get('ltp', 'N/A')} | Open: ₹{ltp_data.get('open', 'N/A')} | High: ₹{ltp_data.get('high', 'N/A')} | Low: ₹{ltp_data.get('low', 'N/A')}")
        except:
            pass
    
    # Calculate indicators & signals
    df = calculate_indicators(df, TRADING_CONFIG)
    df = generate_signals(df, TRADING_CONFIG)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

else:
    # WISHLIST SCAN MODE
    st.title("📋 Wishlist Scanner - Intraday Signals")
    
    if not st.session_state.wishlist:
        st.warning("⚠️ Your wishlist is empty. Add stocks using the search feature in the sidebar.")
        st.stop()
    
    col_scan1, col_scan2 = st.columns([3, 1])
    with col_scan1:
        st.info(f"🔍 Scanning {len(st.session_state.wishlist)} stocks from your wishlist...")
    with col_scan2:
        if st.button("🔄 Scan Now", use_container_width=True):
            st.rerun()
    
    scan_results = []
    progress_bar = st.progress(0)
    
    for idx, stock in enumerate(st.session_state.wishlist):
        progress_bar.progress((idx + 1) / len(st.session_state.wishlist))
        
        try:
            # Fetch data for each stock
            df = fetch_market_data(
                stock['name'], 
                stock['token'], 
                stock['exchange'], 
                timeframe, 
                days=5
            )
            
            if not df.empty:
                # Calculate indicators & signals
                df = calculate_indicators(df, TRADING_CONFIG)
                df = generate_signals(df, TRADING_CONFIG)
                
                latest = df.iloc[-1]
                
                # Check for active signals
                if latest['buy_signal'] or latest['sell_signal']:
                    scan_results.append({
                        'stock': stock['name'],
                        'token': stock['token'],
                        'exchange': stock['exchange'],
                        'ltp': latest['close'],
                        'signal': 'BUY' if latest['buy_signal'] else 'SELL',
                        'strength': latest['buy_strength'] if latest['buy_signal'] else latest['sell_strength'],
                        'quality': latest['buy_quality'] if latest['buy_signal'] else latest['sell_quality'],
                        'rsi': latest['rsi'],
                        'trend': 'Bullish' if latest['bullish_trend'] else 'Bearish',
                        'last_updated': df.index[-1]
                    })
            
            time.sleep(0.35)  # Rate limiting: ~3 requests per second
            
        except Exception as e:
            st.caption(f"⚠️ Error scanning {stock['name']}: {str(e)}")
    
    progress_bar.empty()
    
    # Display scan results
    st.markdown("---")
    st.subheader("📊 Scan Results")
    
    if scan_results:
        # Separate buy and sell signals
        buy_signals = [r for r in scan_results if r['signal'] == 'BUY']
        sell_signals = [r for r in scan_results if r['signal'] == 'SELL']
        
        # Show buy signals
        if buy_signals:
            st.markdown("### 🟢 BUY Signals")
            for result in sorted(buy_signals, key=lambda x: x['strength'], reverse=True):
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                    with col1:
                        st.markdown(f"**📊 {result['stock']}**")
                        st.caption(f"Token: {result['token']}")
                    with col2:
                        st.metric("LTP", f"₹{result['ltp']:.2f}")
                    with col3:
                        st.metric("Strength", f"{result['strength']:.0f}/4")
                    with col4:
                        st.metric("Quality", f"{result['quality']:.0f}/4")
                    with col5:
                        st.metric("RSI", f"{result['rsi']:.1f}")
                    with col6:
                        st.caption(f"🎯 {result['trend']}")
                    st.caption(f"⏰ Updated: {result['last_updated'].strftime('%H:%M:%S')}")
                    st.markdown("---")
        
        # Show sell signals
        if sell_signals:
            st.markdown("### 🔴 SELL Signals")
            for result in sorted(sell_signals, key=lambda x: x['strength'], reverse=True):
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                    with col1:
                        st.markdown(f"**📊 {result['stock']}**")
                        st.caption(f"Token: {result['token']}")
                    with col2:
                        st.metric("LTP", f"₹{result['ltp']:.2f}")
                    with col3:
                        st.metric("Strength", f"{result['strength']:.0f}/4")
                    with col4:
                        st.metric("Quality", f"{result['quality']:.0f}/4")
                    with col5:
                        st.metric("RSI", f"{result['rsi']:.1f}")
                    with col6:
                        st.caption(f"🎯 {result['trend']}")
                    st.caption(f"⏰ Updated: {result['last_updated'].strftime('%H:%M:%S')}")
                    st.markdown("---")
    else:
        st.info("ℹ️ No active signals found in your wishlist stocks at the moment.")
    
    st.stop()  # Stop execution for scan mode

# Continue with single stock view
prev = df.iloc[-2] if len(df) > 1 else latest

# Metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    change = latest['close'] - prev['close']
    pct = (change / prev['close']) * 100
    st.metric("💰 LTP", f"₹{latest['close']:.2f}", f"{change:+.2f} ({pct:+.2f}%)")

with col2:
    trend = "🟢 Bull" if latest['bullish_trend'] else "🔴 Bear"
    st.metric("📈 Trend", trend)

with col3:
    rsi_color = "🔴" if latest['rsi'] > 70 else "🟢" if latest['rsi'] < 30 else "🟡"
    st.metric(f"{rsi_color} RSI", f"{latest['rsi']:.1f}")

with col4:
    adx_color = "🟢" if latest['adx'] > 20 else "🔴"
    st.metric(f"{adx_color} ADX", f"{latest['adx']:.1f}")

with col5:
    vwap_pos = "Above" if latest['close'] > latest['vwap'] else "Below"
    st.metric("🔵 VWAP", vwap_pos)

with col6:
    vol = "🔊 High" if latest['volume_spike'] else "🔉 Normal"
    st.metric("📊 Volume", vol)

st.markdown("---")

# Signal panel with enhanced details
col1, col2 = st.columns(2)

with col1:
    if latest['buy_signal']:
        sl_price = latest['close'] - latest['atr'] * TRADING_CONFIG['atr_multiplier_sl']
        tp_price = latest['close'] + latest['atr'] * TRADING_CONFIG['atr_multiplier_tp']
        risk_reward = (tp_price - latest['close']) / (latest['close'] - sl_price)
        st.success(f"""
        🟢 **BUY SIGNAL DETECTED!**
        
        **Entry:** ₹{latest['close']:.2f}
        **Stop Loss:** ₹{sl_price:.2f} ({((sl_price - latest['close']) / latest['close'] * 100):.2f}%)
        **Take Profit:** ₹{tp_price:.2f} ({((tp_price - latest['close']) / latest['close'] * 100):.2f}%)
        **Risk:Reward:** 1:{risk_reward:.2f}
        
        **Signal Strength:** {int(latest['buy_strength'])}/4
        **Quality Score:** {int(latest['buy_quality_score'])}/4
        **ATR:** ₹{latest['atr']:.2f}
        """)
    else:
        st.info("ℹ️ No BUY signal at current candle")

with col2:
    if latest['sell_signal']:
        sl_price = latest['close'] + latest['atr'] * TRADING_CONFIG['atr_multiplier_sl']
        tp_price = latest['close'] - latest['atr'] * TRADING_CONFIG['atr_multiplier_tp']
        risk_reward = (latest['close'] - tp_price) / (sl_price - latest['close'])
        st.error(f"""
        🔴 **SELL SIGNAL DETECTED!**
        
        **Entry:** ₹{latest['close']:.2f}
        **Stop Loss:** ₹{sl_price:.2f} ({((sl_price - latest['close']) / latest['close'] * 100):.2f}%)
        **Take Profit:** ₹{tp_price:.2f} ({((tp_price - latest['close']) / latest['close'] * 100):.2f}%)
        **Risk:Reward:** 1:{risk_reward:.2f}
        
        **Signal Strength:** {int(latest['sell_strength'])}/4
        **Quality Score:** {int(latest['sell_quality_score'])}/4
        **ATR:** ₹{latest['atr']:.2f}
        """)
    else:
        st.info("ℹ️ No SELL signal at current candle")

st.markdown("---")

# Chart
st.subheader("📊 Price Chart")
chart = create_chart(df, selected_symbol)
st.plotly_chart(chart, use_container_width=True)

# Technical details
with st.expander("📋 Technical Details & TradingView Comparison"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**EMAs**")
        st.write(f"Fast (9): ₹{latest['ema_fast']:.2f}")
        st.write(f"Medium (21): ₹{latest['ema_medium']:.2f}")
        st.write(f"Slow (50): ₹{latest['ema_slow']:.2f}")
    
    with col2:
        st.markdown("**Momentum**")
        st.write(f"RSI: {latest['rsi']:.2f}")
        st.write(f"MACD: {latest['macd_line']:.4f}")
        st.write(f"Signal: {latest['macd_signal']:.4f}")
    
    with col3:
        st.markdown("**Volatility**")
        st.write(f"ATR: ₹{latest['atr']:.2f}")
        st.write(f"ADX: {latest['adx']:.2f}")
        st.write(f"VWAP: ₹{latest['vwap']:.2f}")
    
    with col4:
        st.markdown("**Trend**")
        st.write(f"Bullish: {'✅' if latest['bullish_trend'] else '❌'}")
        st.write(f"Bearish: {'✅' if latest['bearish_trend'] else '❌'}")
        st.write(f"Volume: {latest['volume']:,.0f}")
    
    st.markdown("---")
    st.markdown("**📊 Compare with TradingView:**")
    st.info(f"""
    ✅ **Quick Check:**
    - Current Price: ₹{latest['close']:.2f}
    - RSI: {latest['rsi']:.2f} (Compare with TV RSI indicator)
    - Trend: {'Bullish' if latest['bullish_trend'] else 'Bearish' if latest['bearish_trend'] else 'Neutral'}
    - VWAP Position: {'Above' if latest['close'] > latest['vwap'] else 'Below'}
    
    💡 **If values differ slightly:**
    - Data lag of 1-5 seconds is normal
    - Different exchanges may have small price variations
    - VWAP resets daily (check if TradingView is also showing daily VWAP)
    - Refresh both to sync to same candle
    """)

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
