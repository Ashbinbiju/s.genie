import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import plotly.express as px
import time
import requests
import io
import logging
import os
import multiprocessing
import retrying
import json

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# API Keys and Constants
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_default_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_default_chat_id")
PERFORMANCE_FILE = "stock_performance.json"

# Tooltips
TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "Breakdown": "Break below support with volume",
    "MACrossover": "Fast MA crosses slow MA",
    "VWAPRejection": "Price rejects VWAP level",
    "RSIReversal": "RSI overbought with trend reversal",
    "BearishFlag": "Bearish flag pattern breakdown",
    "HeadAndShoulders": "Reversal pattern with three peaks",
    "DoubleTopBottom": "Reversal pattern with two peaks/troughs",
    "A_D": "Accumulation/Distribution - Buying/selling pressure",
    "VPT": "Volume-Price Trend - Confirms price trends",
}

def tooltip(label, explanation):
    return f"{label} 📌 ({explanation})"

@st.cache_data(ttl=86400)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, timeout=30)
        nse_data = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in nse_data['SYMBOL']]
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch NSE stock list: {str(e)}. Using fallback list.")
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]

@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=3)
@st.cache_data
def fetch_stock_data_batch(symbols, interval="5m", period="1d"):
    try:
        data = yf.download(symbols, period=period, interval=interval, group_by='ticker', threads=False, prepost=False)
        if data.empty:
            return {}
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
        for col in data.columns:
            if 'Volume' in col:
                data[col] = data[col].astype(np.int32)
            else:
                data[col] = data[col].astype(np.float32)
        return {symbol: data[symbol].dropna() for symbol in symbols if symbol in data.columns and not data[symbol].empty}
    except Exception as e:
        logger.error(f"Error fetching batch data: {str(e)}")
        return {}

@lru_cache(maxsize=100)
def optimize_rsi_window(close_data):
    close = np.array(close_data)
    best_window, best_sharpe = 14, -float('inf')
    if len(close) < 15:
        return best_window
    close = np.where(close == 0, 1e-10, close)
    returns = np.diff(close) / close[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    for window in range(10, 15):
        rsi = ta.momentum.RSIIndicator(pd.Series(close), window=window).rsi().values
        signals = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        strategy_returns = signals[:-1] * returns
        strategy_returns = np.nan_to_num(strategy_returns, nan=0.0, posinf=0.0, neginf=0.0)
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) != 0 else 0
        if sharpe > best_sharpe:
            best_sharpe, best_window = sharpe, window
    return best_window

@lru_cache(maxsize=100)
def optimize_macd_params(close_data):
    close = np.array(close_data)
    best_params, best_sharpe = (12, 26, 9), -float('inf')
    if len(close) < 26:
        return best_params
    close = np.where(close == 0, 1e-10, close)
    returns = np.diff(close) / close[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    for fast, slow, signal in [(12, 26, 9), (10, 20, 5), (14, 24, 7)]:
        macd = ta.trend.MACD(pd.Series(close), window_fast=fast, window_slow=slow, window_sign=signal)
        signals = np.where(macd.macd() > macd.macd_signal(), 1, -1)
        strategy_returns = signals[:-1] * returns
        strategy_returns = np.nan_to_num(strategy_returns, nan=0.0, posinf=0.0, neginf=0.0)
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) != 0 else 0
        if sharpe > best_sharpe:
            best_sharpe, best_params = sharpe, (fast, slow, signal)
    return best_params

def calculate_indicators(data):
    if len(data) < 5:
        return data
    try:
        rsi_window = optimize_rsi_window(tuple(data['Close'].values))
        macd_fast, macd_slow, macd_sign = optimize_macd_params(tuple(data['Close'].values))
    except Exception as e:
        logger.error(f"Error optimizing indicators: {str(e)}")
        return data
    
    windows = {
        'rsi': min(rsi_window, len(data) - 1),
        'macd_slow': min(macd_slow, len(data) - 1),
        'macd_fast': min(macd_fast, len(data) - 1),
        'macd_sign': min(macd_sign, len(data) - 1),
        'bollinger': min(20, len(data) - 1),
        'atr': min(7, len(data) - 1),
        'volume': min(5, len(data) - 1),
        'donchian': min(10, len(data) - 1),
        'ema_fast': min(9, len(data) - 1),
        'ema_slow': min(21, len(data) - 1),
    }

    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=windows['rsi']).rsi()
    macd = ta.trend.MACD(data['Close'], window_slow=windows['macd_slow'], window_fast=windows['macd_fast'], window_sign=windows['macd_sign'])
    data[['MACD', 'MACD_signal']] = pd.DataFrame({'MACD': macd.macd(), 'MACD_signal': macd.macd_signal()}, index=data.index)
    bollinger = ta.volatility.BollingerBands(data['Close'], window=windows['bollinger'])
    data[['Upper_Band', 'Lower_Band']] = pd.DataFrame({
        'Upper_Band': bollinger.bollinger_hband(),
        'Lower_Band': bollinger.bollinger_lband()
    }, index=data.index)
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=windows['atr']).average_true_range()
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['Avg_Volume'] = data['Volume'].rolling(window=windows['volume'], min_periods=1).mean()
    data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 2)
    donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=windows['donchian'])
    data[['Donchian_Upper', 'Donchian_Lower']] = pd.DataFrame({
        'Donchian_Upper': donchian.donchian_channel_hband(),
        'Donchian_Lower': donchian.donchian_channel_lband()
    }, index=data.index)
    data['EMA_Fast'] = ta.trend.EMAIndicator(data['Close'], window=windows['ema_fast']).ema_indicator()
    data['EMA_Slow'] = ta.trend.EMAIndicator(data['Close'], window=windows['ema_slow']).ema_indicator()
    data['A_D'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']).replace(0, np.nan) * data['Volume'].cumsum()
    data['VPT'] = (data['Volume'] * data['Close'].pct_change()).cumsum().fillna(0)
    return data

def calculate_risk_metrics(data):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 5:
        return 0, 0
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = returns.mean() / downside_std if downside_std != 0 else 0
    return sharpe, sortino

def detect_patterns(data):
    patterns = {}
    if len(data) >= 2:
        curr, prev = data.iloc[-1], data.iloc[-2]
        patterns['Engulfing'] = "Bullish" if (curr['Close'] > curr['Open'] and prev['Close'] < prev['Open'] and
                                              curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']) else \
                               "Bearish" if (curr['Close'] < curr['Open'] and prev['Close'] > prev['Open'] and
                                             curr['Close'] < prev['Open'] and curr['Open'] > prev['Close']) else None
    
    if len(data) >= 10:
        recent = data.tail(10)
        pole_drop = (recent['Close'].iloc[0] - recent['Close'].iloc[4]) / recent['Close'].iloc[0] if recent['Close'].iloc[0] != 0 else 0
        patterns['Bearish_Flag'] = pole_drop < -0.02 and (recent['High'].iloc[5:].max() - recent['Low'].iloc[5:].min()) < (recent['Close'].iloc[0] * 0.01) and recent['Close'].iloc[-1] < recent['Low'].iloc[5:].min()
    
    if len(data) >= 20:
        recent = data.tail(20)
        highs, lows = recent['High'], recent['Low']
        mid = 10
        head = highs[mid-2:mid+3].max()
        left_shoulder, right_shoulder = highs[:mid-2].max(), highs[mid+3:].max()
        neckline = max(lows[mid-2:mid+2].min(), lows[:mid-2].min(), lows[mid+3:].min())
        patterns['Head_and_Shoulders'] = "Bearish" if (head > left_shoulder and head > right_shoulder and highs.iloc[-1] < neckline) else None
        
        peaks = highs[highs > highs.shift(1)].nlargest(2)
        if len(peaks) == 2 and abs(peaks.iloc[0] - peaks.iloc[1]) < peaks.mean() * 0.05 and highs.iloc[-1] < lows[lows.index > peaks.index.min()].min():
            patterns['Double_Top_Bottom'] = "Bearish Double Top"
    return patterns

def save_performance(symbol, recommendation, date_str):
    """Save recommendation performance data"""
    performance_data = {}
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, 'r') as f:
            performance_data = json.load(f)
    
    if date_str not in performance_data:
        performance_data[date_str] = {}
    
    performance_data[date_str][symbol] = {
        'date': date_str,
        'current_price': recommendation['Current Price'],
        'buy_at': recommendation['Buy At'],
        'stop_loss': recommendation['Stop Loss'],
        'target': recommendation['Target'],
        'recommendation': recommendation['Intraday'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(performance_data, f, indent=2)

def load_performance_data():
    """Load historical performance data"""
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, 'r') as f:
            return json.load(f)
    return {}

def calculate_performance(symbol, recommendation, current_price):
    """Calculate performance metrics for a recommendation"""
    performance = {
        'symbol': symbol,
        'initial_price': recommendation['current_price'],
        'current_price': current_price,
        'recommendation': recommendation['recommendation'],
        'target': recommendation['target'],
        'stop_loss': recommendation['stop_loss']
    }
    
    if current_price and recommendation['current_price']:
        performance['price_change'] = ((current_price - recommendation['current_price']) / 
                                     recommendation['current_price'] * 100)
        
        if recommendation['recommendation'] == 'Buy':
            performance['target_hit'] = current_price >= recommendation['target']
            performance['stop_hit'] = current_price <= recommendation['stop_loss']
            performance['profit_loss'] = (current_price - recommendation['current_price']) / recommendation['current_price'] * 100
        elif recommendation['recommendation'] == 'Sell':
            performance['target_hit'] = current_price <= recommendation['target']
            performance['stop_hit'] = current_price >= recommendation['stop_loss']
            performance['profit_loss'] = (recommendation['current_price'] - current_price) / recommendation['current_price'] * 100
        else:
            performance['profit_loss'] = 0
            performance['target_hit'] = False
            performance['stop_hit'] = False
    
    return performance

def generate_summary(period='daily'):
    """Generate performance summary for specified period"""
    performance_data = load_performance_data()
    if not performance_data:
        return None
        
    today = date.today()
    summary = {
        'total_recommendations': 0,
        'successful_trades': 0,
        'average_profit_loss': 0,
        'win_rate': 0,
        'stocks': []
    }
    
    for date_str, stocks in performance_data.items():
        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        if period == 'daily' and date_obj != today:
            continue
        elif period == 'weekly' and (today - date_obj).days > 7:
            continue
            
        for symbol, rec in stocks.items():
            # Fetch current price
            data_batch = fetch_stock_data_batch([symbol])
            current_data = data_batch.get(symbol)
            current_price = current_data['Close'].iloc[-1] if current_data is not None and not current_data.empty else None
            
            if current_price:
                perf = calculate_performance(symbol, rec, current_price)
                summary['stocks'].append(perf)
                summary['total_recommendations'] += 1
                summary['average_profit_loss'] += perf['profit_loss']
                
                if perf['target_hit']:
                    summary['successful_trades'] += 1
    
    if summary['total_recommendations'] > 0:
        summary['average_profit_loss'] /= summary['total_recommendations']
        summary['win_rate'] = (summary['successful_trades'] / summary['total_recommendations']) * 100
    
    return summary

def analyze_stock(symbol, data_batch):
    data = data_batch.get(symbol, pd.DataFrame())
    if len(data) < 5:
        return None
    
    data = calculate_indicators(data)
    sharpe, sortino = calculate_risk_metrics(data)
    price_action = detect_patterns(data)
    
    rec = generate_recommendations(data, sharpe, sortino, price_action)
    result = {
        "Symbol": symbol, "Current Price": rec["Current Price"], "Buy At": rec["Buy At"],
        "Stop Loss": rec["Stop Loss"], "Target": rec["Target"], "Intraday": rec["Intraday"],
        "Breakdown": rec["Breakdown"], "MACrossover": rec["MACrossover"], "VWAPRejection": rec["VWAPRejection"],
        "RSIReversal": rec["RSIReversal"], "BearishFlag": rec["BearishFlag"],
        "HeadAndShoulders": rec["HeadAndShoulders"], "DoubleTopBottom": rec["DoubleTopBottom"],
        "Score": rec["Score"], "Sharpe": sharpe, "Sortino": sortino
    }
    
    # Save performance data
    date_str = date.today().strftime('%Y-%m-%d')
    save_performance(symbol, result, date_str)
    
    return result

def generate_recommendations(data, sharpe, sortino, price_action):
    rec = {k: "Hold" for k in ["Intraday", "Breakdown", "MACrossover", "VWAPRejection", "RSIReversal", "BearishFlag", "HeadAndShoulders", "DoubleTopBottom"]}
    rec.update({"Current Price": round(data['Close'].iloc[-1], 2) if not data.empty else None, "Buy At": None, "Stop Loss": None, "Target": None, "Score": 0})
    if data.empty:
        return rec
    
    buy_score, sell_score = 0, 0
    last_close = data['Close'].iloc[-1]
    
    if 'Donchian_Lower' in data and data['Volume_Spike'].iloc[-1]:
        rec["Breakdown"] = "Sell" if last_close < data['Donchian_Lower'].iloc[-1] else "Buy"
        sell_score += 2 if rec["Breakdown"] == "Sell" else 0
        buy_score += 1 if rec["Breakdown"] == "Buy" else 0

    if 'EMA_Fast' in data and len(data) >= 2:
        rec["MACrossover"] = "Buy" if data['EMA_Fast'].iloc[-1] > data['EMA_Slow'].iloc[-1] and data['EMA_Fast'].iloc[-2] <= data['EMA_Slow'].iloc[-2] else \
                            "Sell" if data['EMA_Fast'].iloc[-1] < data['EMA_Slow'].iloc[-1] and data['EMA_Fast'].iloc[-2] >= data['EMA_Slow'].iloc[-2] else "Hold"
        buy_score += 2 if rec["MACrossover"] == "Buy" else 0
        sell_score += 2 if rec["MACrossover"] == "Sell" else 0

    if 'VWAP' in data and len(data) >= 3:
        vwap = data['VWAP'].iloc[-1]
        rec["VWAPRejection"] = "Sell" if data['Close'].iloc[-2] > vwap and data['Close'].iloc[-1] < vwap and data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1] else \
                              "Buy" if data['Close'].iloc[-2] < vwap and data['Close'].iloc[-1] > vwap and data['Volume'].iloc[-1] > data['Avg_Volume'].iloc[-1] else "Hold"
        sell_score += 2 if rec["VWAPRejection"] == "Sell" else 0
        buy_score += 2 if rec["VWAPRejection"] == "Buy" else 0

    if 'RSI' in data and 'MACD' in data:
        rec["RSIReversal"] = "Sell" if data['RSI'].iloc[-1] > 70 and data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1] else \
                            "Buy" if data['RSI'].iloc[-1] < 30 and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] else "Hold"
        sell_score += 2 if rec["RSIReversal"] == "Sell" else 0
        buy_score += 2 if rec["RSIReversal"] == "Buy" else 0

    rec["BearishFlag"] = "Sell" if price_action.get('Bearish_Flag') else "Hold"
    sell_score += 2 if rec["BearishFlag"] == "Sell" else 0

    rec["HeadAndShoulders"] = "Sell" if price_action.get('Head_and_Shoulders') == "Bearish" else "Hold"
    sell_score += 2 if rec["HeadAndShoulders"] == "Sell" else 0

    rec["DoubleTopBottom"] = "Sell" if price_action.get('Double_Top_Bottom') == "Bearish Double Top" else "Hold"
    sell_score += 2 if rec["DoubleTopBottom"] == "Sell" else 0

    if price_action.get('Engulfing') == "Bullish":
        buy_score += 1
    elif price_action.get('Engulfing') == "Bearish":
        sell_score += 1

    if sharpe > 1 and sortino > 1:
        buy_score += 1
    elif sharpe < 0 or sortino < 0:
        sell_score += 1

    rec["Intraday"] = "Buy" if buy_score > sell_score + 2 else "Sell" if sell_score > buy_score + 2 else "Hold"
    rec["Buy At"] = round(last_close * 0.99, 2) if 'RSI' in data and data['RSI'].iloc[-1] < 30 else rec["Current Price"]
    rec["Stop Loss"] = round(last_close - (1.5 * data['ATR'].iloc[-1]), 2) if 'ATR' in data else None
    rec["Target"] = round(last_close + 2 * (last_close - rec["Stop Loss"]), 2) if rec["Stop Loss"] else None
    rec["Score"] = max(0, min(buy_score - sell_score, 7))
    return rec

def analyze_batch(stock_batch, data_batch):
    results = []
    max_workers = min(multiprocessing.cpu_count(), 10)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_stock, symbol, data_batch): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    return results

@st.cache_data
def analyze_all_stocks(stock_list, batch_size=25, price_range=None):
    if st.session_state.get('cancel_operation', False):
        return pd.DataFrame()
    
    results = []
    total_items = len(stock_list)
    progress_bar = st.progress(0)
    loading_text = st.empty()
    start_time = time.time()

    for i in range(0, total_items, batch_size):
        if st.session_state.get('cancel_operation', False):
            break
        batch = stock_list[i:i + batch_size]
        data_batch = fetch_stock_data_batch(batch)
        batch_results = analyze_batch(batch, data_batch)
        results.extend(batch_results)
        processed_items = min(i + batch_size, total_items)
        progress_value = processed_items / total_items
        progress_bar.progress(progress_value)
        elapsed = time.time() - start_time
        eta = int((elapsed / processed_items) * (total_items - processed_items)) if processed_items > 0 else 0
        loading_text.text(f"Processing {processed_items}/{total_items} (ETA: {eta}s)")

    progress_bar.empty()
    loading_text.empty()
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    if price_range:
        results_df = results_df[results_df['Current Price'].notna() & results_df['Current Price'].between(price_range[0], price_range[1])]
    return results_df.sort_values(by="Score", ascending=False).head(5)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{"7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps"}/sendMessage"
    payload = {"chat_id": "-1002411670969", "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5).raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")

def display_dashboard(symbol=None, data=None, recommendations=None, NSE_STOCKS=None):
    st.title("📊 StockGenie Pro - Intraday Analysis")
    price_range = st.sidebar.slider("Price Range (₹)", 0, 10000, (100, 1000))

    # Add summary selection
    summary_type = st.sidebar.selectbox("View Performance Summary", ["None", "Daily", "Weekly"])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Top Intraday Picks"):
            st.session_state.cancel_operation = False
            cancel_button = st.button("❌ Cancel")
            if cancel_button:
                st.session_state.cancel_operation = True
            results_df = analyze_all_stocks(NSE_STOCKS, price_range=price_range)
            if not results_df.empty:
                st.subheader("🏆 Top 5 Intraday Stocks")
                telegram_msg = f"*Top 5 Intraday Stocks ({datetime.now().strftime('%d %b %Y')})*\nChat ID: {TELEGRAM_CHAT_ID}\n\n"
                for _, row in results_df.iterrows():
                    with st.expander(f"{row['Symbol']} - Score: {row['Score']}"):
                        st.write(f"Price: ₹{row['Current Price']:.2f}, Buy At: ₹{row['Buy At']:.2f}, "
                                 f"Stop Loss: ₹{row['Stop Loss']:.2f}, Target: ₹{row['Target']:.2f}, "
                                 f"Intraday: {row['Intraday']}, Telegram Chat ID: {TELEGRAM_CHAT_ID}")
                    telegram_msg += f"*{row['Symbol']}*: ₹{row['Current Price']:.2f} - {row['Intraday']}\n"
                send_telegram_message(telegram_msg)

    # Display performance summary
    if summary_type != "None":
        period = summary_type.lower()
        summary = generate_summary(period)
        if summary:
            st.subheader(f"{summary_type} Performance Summary")
            st.write(f"Total Recommendations: {summary['total_recommendations']}")
            st.write(f"Successful Trades: {summary['successful_trades']}")
            st.write(f"Average Profit/Loss: {summary['average_profit_loss']:.2f}%")
            st.write(f"Win Rate: {summary['win_rate']:.2f}%")
            
            # Display detailed performance for each stock
            if summary['stocks']:
                st.subheader("Detailed Performance")
                for stock in summary['stocks']:
                    with st.expander(f"{stock['symbol']}"):
                        st.write(f"Initial Price: ₹{stock['initial_price']:.2f}")
                        st.write(f"Current Price: ₹{stock['current_price']:.2f}")
                        st.write(f"Recommendation: {stock['recommendation']}")
                        st.write(f"Price Change: {stock['price_change']:.2f}%")
                        st.write(f"Target Hit: {stock['target_hit']}")
                        st.write(f"Stop Loss Hit: {stock['stop_hit']}")
                        st.write(f"Profit/Loss: {stock['profit_loss']:.2f}%")

    if symbol and data is not None and recommendations is not None:
        st.header(f"📋 {symbol.split('.')[0]} Analysis")
        st.write(f"Price: ₹{recommendations['Current Price']:.2f}, Intraday: {recommendations['Intraday']}, Telegram Chat ID: {TELEGRAM_CHAT_ID}")
        fig = px.line(data, y=['Close', 'VWAP'], title="Price & VWAP (5m)")
        st.plotly_chart(fig)

def main():
    if 'cancel_operation' not in st.session_state:
        st.session_state.cancel_operation = False
    NSE_STOCKS = fetch_nse_stock_list()
    symbol = st.sidebar.selectbox("Stock:", [""] + NSE_STOCKS)
    
    if symbol:
        data_batch = fetch_stock_data_batch([symbol])
        data = data_batch.get(symbol)
        if data is not None and not data.empty:
            data = calculate_indicators(data)
            sharpe, sortino = calculate_risk_metrics(data)
            price_action = detect_patterns(data)
            recommendations = generate_recommendations(data, sharpe, sortino, price_action)
            display_dashboard(symbol, data, recommendations, NSE_STOCKS)
        else:
            st.error(f"❌ No data for {symbol}")
    else:
        display_dashboard(None, None, None, NSE_STOCKS)

if __name__ == "__main__":
    main()