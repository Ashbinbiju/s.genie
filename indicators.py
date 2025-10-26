"""Technical indicators matching Pine Script"""

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


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
    
    # Trends
    df['bullish_trend'] = (df['ema_fast'] > df['ema_medium']) & (df['ema_medium'] > df['ema_slow'])
    df['bearish_trend'] = (df['ema_fast'] < df['ema_medium']) & (df['ema_medium'] < df['ema_slow'])
    
    return df


def generate_signals(df, config):
    """Generate buy/sell signals"""
    df = df.copy()
    
    # Buy signals
    df['buy_signal_1'] = (
        (df['ema_fast'] > df['ema_fast'].shift(1)) &
        (df['ema_fast'].shift(1) <= df['ema_medium'].shift(1)) &
        (df['rsi'] > 50) &
        (df['close'] > df['vwap'])
    )
    
    df['buy_signal_2'] = (
        (df['macd_line'] > df['macd_signal']) &
        (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) &
        (df['rsi'] < config['rsi_overbought']) &
        df['bullish_trend']
    )
    
    df['buy_strength'] = df['buy_signal_1'].astype(int) + df['buy_signal_2'].astype(int)
    
    # Sell signals
    df['sell_signal_1'] = (
        (df['ema_fast'] < df['ema_fast'].shift(1)) &
        (df['ema_fast'].shift(1) >= df['ema_medium'].shift(1)) &
        (df['rsi'] < 50) &
        (df['close'] < df['vwap'])
    )
    
    df['sell_signal_2'] = (
        (df['macd_line'] < df['macd_signal']) &
        (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)) &
        (df['rsi'] > config['rsi_oversold']) &
        df['bearish_trend']
    )
    
    df['sell_strength'] = df['sell_signal_1'].astype(int) + df['sell_signal_2'].astype(int)
    
    # Quality scores
    df['buy_quality_score'] = (
        (df['bullish_trend'] | (df['ema_fast'] > df['ema_medium'])).astype(int) +
        (df['macd_line'] > df['macd_signal']).astype(int) +
        ((df['rsi'] > 40) & (df['rsi'] < config['rsi_overbought'])).astype(int) +
        (df['close'] > df['vwap']).astype(int)
    )
    
    df['sell_quality_score'] = (
        (df['bearish_trend'] | (df['ema_fast'] < df['ema_medium'])).astype(int) +
        (df['macd_line'] < df['macd_signal']).astype(int) +
        ((df['rsi'] < 60) & (df['rsi'] > config['rsi_oversold'])).astype(int) +
        (df['close'] < df['vwap']).astype(int)
    )
    
    # Final signals
    df['buy_signal'] = (
        (df['buy_strength'] >= config['min_signal_strength']) &
        (df['bullish_trend'] | (df['ema_fast'] > df['ema_medium'])) &
        (df['buy_quality_score'] >= config['min_trade_quality'])
    )
    
    df['sell_signal'] = (
        (df['sell_strength'] >= config['min_signal_strength']) &
        (df['bearish_trend'] | (df['ema_fast'] < df['ema_medium'])) &
        (df['sell_quality_score'] >= config['min_trade_quality'])
    )
    
    return df
