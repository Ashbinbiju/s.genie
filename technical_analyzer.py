import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from api_manager import APIManager
from config import TECHNICAL_CONFIG

class TechnicalAnalyzer:
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
        self.config = TECHNICAL_CONFIG
    
    def _candles_to_df(self, candles: List) -> pd.DataFrame:
        """Convert candlestick data to DataFrame"""
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = self.config["rsi_period"]
        
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        fast = self.config["macd_fast"]
        slow = self.config["macd_slow"]
        signal = self.config["macd_signal"]
        
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate ADX (Average Directional Index)"""
        if period is None:
            period = self.config["adx_period"]
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def analyze_stock(self, symbol: str, timeframe: str = "5min") -> Optional[Dict]:
        """Complete technical analysis for a stock using Streak API"""
        # Use the comprehensive technical analysis API
        tech_data = self.api.fetch_technical_analysis(symbol, timeframe)
        
        if not tech_data or tech_data.get("status") != 1:
            # Fallback to manual calculation if API fails
            return self._analyze_stock_manual(symbol, timeframe)
        
        # Extract data from Streak API response
        current_price = tech_data.get("close", 0)
        rsi = tech_data.get("rsi")
        macd = tech_data.get("macd")
        macd_hist = tech_data.get("macdHist")
        adx = tech_data.get("adx")
        
        # Get recommendation signals
        rec_rsi = tech_data.get("rec_rsi", 0)  # -1, 0, 1
        rec_macd = tech_data.get("rec_macd", 0)
        rec_adx = tech_data.get("rec_adx", 0)
        
        # Overall signal state (-1 = bearish, 0 = neutral, 1 = bullish)
        state = tech_data.get("state", 0)
        
        # Win/Loss statistics
        win_signals = tech_data.get("win_signals", 0)
        loss_signals = tech_data.get("loss_signals", 0)
        win_pct = tech_data.get("win_pct", 0)
        
        # Generate signals
        signals = {
            "rsi": "bullish" if rec_rsi > 0 else ("bearish" if rec_rsi < 0 else "neutral"),
            "macd": "bullish" if rec_macd > 0 else ("bearish" if rec_macd < 0 else "neutral"),
            "trend": "strong" if adx and adx > self.config["adx_strong_trend"] else "weak",
            "overall": "bullish" if state > 0 else ("bearish" if state < 0 else "neutral")
        }
        
        # Calculate score based on API signals and historical performance
        score = self._calculate_score_from_api(tech_data, signals, win_pct)
        
        # Calculate volume ratio from API data if available
        volume = tech_data.get("volume", 0)
        avg_volume = tech_data.get("avgVolume", 0)
        volume_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": float(current_price),
            "rsi": float(rsi) if rsi else None,
            "macd": float(macd) if macd else None,
            "macd_hist": float(macd_hist) if macd_hist else None,
            "adx": float(adx) if adx else None,
            "volume_ratio": volume_ratio,
            "signals": signals,
            "score": score,
            "win_rate": round(win_pct * 100, 1) if win_pct else None,
            "total_signals": win_signals + loss_signals,
            "state": state
        }
    
    def _analyze_stock_manual(self, symbol: str, timeframe: str = "5min") -> Optional[Dict]:
        """Fallback: Manual technical analysis when API fails"""
        candles = self.api.fetch_candlestick_data(symbol, timeframe)
        if not candles or len(candles) < 50:
            return None
        
        df = self._candles_to_df(candles)
        
        # Calculate indicators
        df["rsi"] = self.calculate_rsi(df)
        macd_data = self.calculate_macd(df)
        df["macd"] = macd_data["macd"]
        df["macd_signal"] = macd_data["signal"]
        df["macd_histogram"] = macd_data["histogram"]
        df["adx"] = self.calculate_adx(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Volume analysis
        avg_volume = df["volume"].tail(20).mean()
        volume_ratio = latest["volume"] / avg_volume if avg_volume > 0 else 0
        
        # Generate signals
        signals = self._generate_signals(latest, prev, volume_ratio)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": float(latest["close"]),
            "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else None,
            "macd": float(latest["macd"]) if pd.notna(latest["macd"]) else None,
            "macd_signal": float(latest["macd_signal"]) if pd.notna(latest["macd_signal"]) else None,
            "adx": float(latest["adx"]) if pd.notna(latest["adx"]) else None,
            "volume_ratio": round(volume_ratio, 2),
            "signals": signals,
            "score": self._calculate_score(signals, latest, volume_ratio)
        }
    
    def _generate_signals(self, latest, prev, volume_ratio) -> Dict:
        """Generate trading signals"""
        signals = {
            "rsi": "neutral",
            "macd": "neutral",
            "trend": "neutral",
            "volume": "low"
        }
        
        # RSI signal
        if pd.notna(latest["rsi"]):
            if latest["rsi"] < self.config["rsi_oversold"]:
                signals["rsi"] = "oversold"
            elif latest["rsi"] > self.config["rsi_overbought"]:
                signals["rsi"] = "overbought"
        
        # MACD signal
        if pd.notna(latest["macd"]) and pd.notna(prev["macd"]):
            if latest["macd"] > latest["macd_signal"] and prev["macd"] <= prev["macd_signal"]:
                signals["macd"] = "bullish_crossover"
            elif latest["macd"] < latest["macd_signal"] and prev["macd"] >= prev["macd_signal"]:
                signals["macd"] = "bearish_crossover"
        
        # Trend strength (ADX)
        if pd.notna(latest["adx"]):
            if latest["adx"] > self.config["adx_strong_trend"]:
                signals["trend"] = "strong"
            else:
                signals["trend"] = "weak"
        
        # Volume signal
        if volume_ratio > self.config["volume_threshold"]:
            signals["volume"] = "high"
        
        return signals
    
    def _calculate_score(self, signals: Dict, latest, volume_ratio: float) -> float:
        """Calculate overall technical score (0-100)"""
        score = 50  # Base score
        
        # RSI contribution
        if signals["rsi"] == "oversold":
            score += 15
        elif signals["rsi"] == "overbought":
            score -= 15
        
        # MACD contribution
        if signals["macd"] == "bullish_crossover":
            score += 20
        elif signals["macd"] == "bearish_crossover":
            score -= 20
        
        # Trend contribution
        if signals["trend"] == "strong":
            score += 10
        
        # Volume contribution
        if signals["volume"] == "high":
            score += 5
        
        return max(0, min(100, score))
    
    def _calculate_score_from_api(self, tech_data: Dict, signals: Dict, win_pct: float) -> float:
        """Calculate score from Streak API technical data"""
        score = 50  # Base score
        
        # State contribution (overall market signal)
        state = tech_data.get("state", 0)
        score += state * 15  # -15 to +15 based on bearish/bullish
        
        # RSI contribution
        rec_rsi = tech_data.get("rec_rsi", 0)
        score += rec_rsi * 10  # -10 to +10
        
        # MACD contribution
        rec_macd = tech_data.get("rec_macd", 0)
        score += rec_macd * 15  # -15 to +15
        
        # ADX/Trend strength
        if signals["trend"] == "strong":
            score += 10
        
        # Historical win rate contribution
        if win_pct:
            # Add bonus for high win rate
            if win_pct > 0.6:  # 60%+ win rate
                score += 10
            elif win_pct < 0.4:  # Below 40% win rate
                score -= 10
        
        # Additional technical indicators
        rec_ao = tech_data.get("rec_ao", 0)  # Awesome Oscillator
        rec_cci = tech_data.get("rec_cci", 0)  # CCI
        rec_stochastic = tech_data.get("rec_stochastic_k", 0)
        
        # Sum of other indicators (each contributes less)
        other_indicators = rec_ao + rec_cci + rec_stochastic
        score += other_indicators * 3  # Small contribution
        
        return max(0, min(100, score))
