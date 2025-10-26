"""Technical indicators matching Pine Script EXACTLY"""

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def calculate_supertrend(df, period, multiplier):
    """Calculate Supertrend indicator"""
    hl2 = (df['high'] + df['low']) / 2
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(1, len(df)):
        if pd.isna(supertrend.iloc[i-1]):
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
    
    return supertrend, direction


def calculate_indicators(df, config):
    """Calculate all technical indicators"""
    df = df.copy()
    
    # EMAs
    df['ema_fast'] = EMAIndicator(df['close'], window=config['ema_fast']).ema_indicator()
    df['ema_medium'] = EMAIndicator(df['close'], window=config['ema_medium']).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=config['ema_slow']).ema_indicator()
    
    # RSI
    df['rsi'] = RSIIndicator(df['close'], window=config['rsi_length']).rsi()
    
    # MACD
    macd = MACD(df['close'], window_fast=config['macd_fast'], window_slow=config['macd_slow'], window_sign=config['macd_signal'])
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(window=config['volume_ma_length']).mean()
    df['volume_spike'] = df['volume'] > (df['volume_ma'] * config['volume_multiplier'])
    
    # ATR
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=config['atr_length']).average_true_range()
    
    # VWAP (reset daily for intraday)
    df['date'] = df.index.date
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = df.groupby('date').apply(
        lambda x: (typical_price.loc[x.index] * x['volume']).cumsum() / x['volume'].cumsum()
    ).reset_index(level=0, drop=True)
    df = df.drop('date', axis=1)
    
    # ADX (with error handling for insufficient data)
    try:
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=config['adx_length'])
        df['adx'] = adx.adx()
    except:
        df['adx'] = 20.0  # Default neutral value
    
    # Supertrend
    supertrend_line, supertrend_dir = calculate_supertrend(
        df, 
        config['supertrend_period'], 
        config['supertrend_multiplier']
    )
    df['supertrend_line'] = supertrend_line
    df['supertrend_direction'] = supertrend_dir
    
    # Trends
    df['bullish_trend'] = (df['ema_fast'] > df['ema_medium']) & (df['ema_medium'] > df['ema_slow'])
    df['bearish_trend'] = (df['ema_fast'] < df['ema_medium']) & (df['ema_medium'] < df['ema_slow'])
    
    return df


def generate_signals(df, config):
    """Generate buy/sell signals - EXACT Pine Script logic"""
    df = df.copy()
    
    # ========================================
    # BUY SIGNALS (matching Pine Script lines 227-230)
    # ========================================
    
    # buySignal1: EMA fast crosses above medium, RSI > 50, price > VWAP
    df['buy_signal_1'] = (
        (df['ema_fast'] > df['ema_medium']) &
        (df['ema_fast'].shift(1) <= df['ema_medium'].shift(1)) &  # crossover
        (df['rsi'] > 50) &
        (df['close'] > df['vwap'])
    )
    
    # buySignal2: MACD crosses above signal, RSI < overbought, bullish trend
    df['buy_signal_2'] = (
        (df['macd_line'] > df['macd_signal']) &
        (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) &  # crossover
        (df['rsi'] < config['rsi_overbought']) &
        df['bullish_trend']
    )
    
    # buySignal3: Price crosses above Supertrend, RSI > 40, volume spike
    df['buy_signal_3'] = (
        (df['close'] > df['supertrend_line']) &
        (df['close'].shift(1) <= df['supertrend_line'].shift(1)) &  # crossover
        (df['rsi'] > 40) &
        df['volume_spike']
    )
    
    # buySignal4: RSI oversold recovery - RSI crosses above 30, price > ema medium
    df['buy_signal_4'] = (
        (df['rsi'] < config['rsi_oversold']) &
        (df['rsi'] > 30) &
        (df['rsi'].shift(1) <= 30) &  # crossover
        (df['close'] > df['ema_medium'])
    )
    
    # Buy strength: Count number of signals (0-4)
    df['buy_strength'] = (
        df['buy_signal_1'].astype(int) + 
        df['buy_signal_2'].astype(int) + 
        df['buy_signal_3'].astype(int) + 
        df['buy_signal_4'].astype(int)
    )
    
    # ========================================
    # SELL SIGNALS (matching Pine Script lines 233-236)
    # ========================================
    
    # sellSignal1: EMA fast crosses below medium, RSI < 50, price < VWAP
    df['sell_signal_1'] = (
        (df['ema_fast'] < df['ema_medium']) &
        (df['ema_fast'].shift(1) >= df['ema_medium'].shift(1)) &  # crossunder
        (df['rsi'] < 50) &
        (df['close'] < df['vwap'])
    )
    
    # sellSignal2: MACD crosses below signal, RSI > oversold, bearish trend
    df['sell_signal_2'] = (
        (df['macd_line'] < df['macd_signal']) &
        (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)) &  # crossunder
        (df['rsi'] > config['rsi_oversold']) &
        df['bearish_trend']
    )
    
    # sellSignal3: Price crosses below Supertrend, RSI < 60, volume spike
    df['sell_signal_3'] = (
        (df['close'] < df['supertrend_line']) &
        (df['close'].shift(1) >= df['supertrend_line'].shift(1)) &  # crossunder
        (df['rsi'] < 60) &
        df['volume_spike']
    )
    
    # sellSignal4: RSI overbought reversal - RSI crosses below 70, price < ema medium
    df['sell_signal_4'] = (
        (df['rsi'] > config['rsi_overbought']) &
        (df['rsi'] < 70) &
        (df['rsi'].shift(1) >= 70) &  # crossunder
        (df['close'] < df['ema_medium'])
    )
    
    # Sell strength: Count number of signals (0-4)
    df['sell_strength'] = (
        df['sell_signal_1'].astype(int) + 
        df['sell_signal_2'].astype(int) + 
        df['sell_signal_3'].astype(int) + 
        df['sell_signal_4'].astype(int)
    )
    
    # ========================================
    # QUALITY SCORES (matching Pine Script lines 253-271)
    # ========================================
    
    # Buy Quality Score (0-4)
    df['buy_quality_score'] = (
        (df['bullish_trend'] | (df['ema_fast'] > df['ema_medium'])).astype(int) +
        (df['macd_line'] > df['macd_signal']).astype(int) +
        ((df['rsi'] > 40) & (df['rsi'] < config['rsi_overbought'])).astype(int) +
        ((df['close'] > df['vwap']) | (df['close'] > df['ema_medium'])).astype(int)
    )
    
    # Sell Quality Score (0-4)
    df['sell_quality_score'] = (
        (df['bearish_trend'] | (df['ema_fast'] < df['ema_medium'])).astype(int) +
        (df['macd_line'] < df['macd_signal']).astype(int) +
        ((df['rsi'] < 60) & (df['rsi'] > config['rsi_oversold'])).astype(int) +
        ((df['close'] < df['vwap']) | (df['close'] < df['ema_medium'])).astype(int)
    )
    
    # ========================================
    # FINAL SIGNAL GENERATION (matching Pine Script lines 278-279)
    # ========================================
    
    # Market condition filters
    strong_trend = df['adx'] > config['min_adx']
    not_choppy = ~config['avoid_choppy'] | strong_trend
    late_entry = (df['close'] - df['ema_medium']).abs() > (df['atr'] * 1.5)
    
    # Buy Signal (line 278)
    df['buy_signal'] = (
        (df['buy_strength'] >= config['min_signal_strength']) &
        (df['bullish_trend'] | (df['ema_fast'] > df['ema_medium'])) &
        not_choppy &
        ~late_entry &
        (df['buy_quality_score'] >= config['min_trade_quality'])
    )
    
    # Sell Signal (line 279)
    df['sell_signal'] = (
        (df['sell_strength'] >= config['min_signal_strength']) &
        (df['bearish_trend'] | (df['ema_fast'] < df['ema_medium'])) &
        not_choppy &
        ~late_entry &
        (df['sell_quality_score'] >= config['min_trade_quality'])
    )
    
    return df
