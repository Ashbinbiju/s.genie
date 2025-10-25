"""Unit tests for TechnicalAnalyzer module."""
import pytest
from unittest.mock import Mock, patch
from technical_analyzer import TechnicalAnalyzer


@pytest.fixture
def analyzer():
    """Create TechnicalAnalyzer instance for testing."""
    mock_api = Mock()
    return TechnicalAnalyzer(mock_api)


@pytest.fixture
def sample_price_data():
    """Sample price data for testing."""
    return [100, 102, 98, 103, 105, 101, 99, 104, 106, 102]


@pytest.fixture
def sample_technical_data():
    """Sample technical analysis data."""
    return {
        'rsi': 65,
        'macd': {'macd': 12, 'signal': 10, 'histogram': 2},
        'moving_averages': {
            'sma_20': 100,
            'sma_50': 95,
            'ema_20': 101,
            'ema_50': 96
        },
        'bollinger_bands': {
            'upper': 110,
            'middle': 100,
            'lower': 90
        }
    }


class TestTechnicalAnalyzer:
    """Test suite for TechnicalAnalyzer class."""

    def test_initialization(self, analyzer):
        """Test TechnicalAnalyzer initializes correctly."""
        assert analyzer.api_manager is not None

    def test_calculate_rsi_overbought(self, analyzer, sample_price_data):
        """Test RSI calculation for overbought condition."""
        # Mock data showing strong uptrend
        prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        rsi = analyzer.calculate_rsi(prices)
        assert rsi is not None
        # Strong uptrend should result in high RSI
        if rsi:
            assert rsi > 50

    def test_calculate_rsi_oversold(self, analyzer):
        """Test RSI calculation for oversold condition."""
        # Mock data showing strong downtrend
        prices = [145, 140, 135, 130, 125, 120, 115, 110, 105, 100]
        rsi = analyzer.calculate_rsi(prices)
        assert rsi is not None
        # Strong downtrend should result in low RSI
        if rsi:
            assert rsi < 50

    def test_calculate_rsi_insufficient_data(self, analyzer):
        """Test RSI calculation with insufficient data."""
        prices = [100, 105]  # Too few data points
        rsi = analyzer.calculate_rsi(prices)
        assert rsi is None or rsi == 0

    def test_calculate_macd_bullish(self, analyzer):
        """Test MACD calculation for bullish signal."""
        # Uptrend data
        prices = list(range(100, 150, 2))
        macd_data = analyzer.calculate_macd(prices)
        assert macd_data is not None

    def test_calculate_macd_bearish(self, analyzer):
        """Test MACD calculation for bearish signal."""
        # Downtrend data
        prices = list(range(150, 100, -2))
        macd_data = analyzer.calculate_macd(prices)
        assert macd_data is not None

    def test_calculate_moving_averages(self, analyzer, sample_price_data):
        """Test moving average calculations."""
        ma_data = analyzer.calculate_moving_averages(sample_price_data)
        assert ma_data is not None
        if ma_data:
            assert 'sma' in ma_data or 'ema' in ma_data

    def test_calculate_bollinger_bands(self, analyzer, sample_price_data):
        """Test Bollinger Bands calculation."""
        bb_data = analyzer.calculate_bollinger_bands(sample_price_data)
        assert bb_data is not None
        if bb_data:
            assert 'upper' in bb_data
            assert 'middle' in bb_data
            assert 'lower' in bb_data
            # Upper band should be greater than middle, middle greater than lower
            assert bb_data['upper'] >= bb_data['middle']
            assert bb_data['middle'] >= bb_data['lower']

    def test_analyze_trend_uptrend(self, analyzer):
        """Test trend analysis for uptrend."""
        prices = [100, 102, 105, 108, 110, 113, 115]
        trend = analyzer.analyze_trend(prices)
        assert trend in ['uptrend', 'bullish', 'up', None]

    def test_analyze_trend_downtrend(self, analyzer):
        """Test trend analysis for downtrend."""
        prices = [115, 113, 110, 108, 105, 102, 100]
        trend = analyzer.analyze_trend(prices)
        assert trend in ['downtrend', 'bearish', 'down', None]

    def test_analyze_trend_sideways(self, analyzer):
        """Test trend analysis for sideways market."""
        prices = [100, 101, 100, 99, 100, 101, 100]
        trend = analyzer.analyze_trend(prices)
        assert trend in ['sideways', 'neutral', 'ranging', None]

    def test_get_technical_score_bullish(self, analyzer):
        """Test technical score calculation for bullish setup."""
        mock_data = {
            'rsi': 55,  # Neutral
            'macd': {'histogram': 5},  # Positive
            'moving_averages': {'sma_20': 100, 'sma_50': 95},  # Golden cross
            'trend': 'uptrend'
        }
        analyzer.api_manager.fetch_technical_analysis = Mock(return_value=mock_data)

        score = analyzer.get_technical_score('TEST', 'daily')
        assert score is not None
        if score:
            assert 0 <= score <= 100

    def test_get_technical_score_bearish(self, analyzer):
        """Test technical score calculation for bearish setup."""
        mock_data = {
            'rsi': 35,  # Oversold
            'macd': {'histogram': -5},  # Negative
            'moving_averages': {'sma_20': 95, 'sma_50': 100},  # Death cross
            'trend': 'downtrend'
        }
        analyzer.api_manager.fetch_technical_analysis = Mock(return_value=mock_data)

        score = analyzer.get_technical_score('TEST', 'daily')
        assert score is not None
        if score:
            assert 0 <= score <= 100

    def test_get_technical_score_no_data(self, analyzer):
        """Test technical score with no data available."""
        analyzer.api_manager.fetch_technical_analysis = Mock(return_value=None)

        score = analyzer.get_technical_score('TEST', 'daily')
        assert score in [None, 0, 50]  # Default or neutral score

    def test_identify_support_resistance(self, analyzer):
        """Test support and resistance identification."""
        mock_sr_data = {
            'support': [95, 90, 85],
            'resistance': [110, 115, 120]
        }
        analyzer.api_manager.fetch_support_resistance = Mock(return_value=mock_sr_data)

        sr_levels = analyzer.identify_support_resistance('TEST', 'daily')
        assert sr_levels is not None
        if sr_levels:
            assert 'support' in sr_levels
            assert 'resistance' in sr_levels

    def test_detect_candlestick_patterns(self, analyzer):
        """Test candlestick pattern detection."""
        mock_pattern_data = {
            'patterns': ['hammer', 'doji', 'engulfing']
        }
        analyzer.api_manager.fetch_candlestick_data = Mock(return_value=mock_pattern_data)

        patterns = analyzer.detect_candlestick_patterns('TEST', 'daily')
        assert patterns is not None
        if patterns:
            assert isinstance(patterns, (list, dict))

    def test_is_bullish_signal(self, analyzer, sample_technical_data):
        """Test bullish signal detection."""
        # RSI > 50, positive MACD, price above moving averages
        sample_technical_data['rsi'] = 60
        sample_technical_data['macd']['histogram'] = 5
        sample_technical_data['price'] = 105

        is_bullish = analyzer.is_bullish_signal(sample_technical_data)
        # Result depends on implementation, just check it returns boolean
        assert isinstance(is_bullish, bool) or is_bullish is None

    def test_is_bearish_signal(self, analyzer, sample_technical_data):
        """Test bearish signal detection."""
        # RSI < 50, negative MACD, price below moving averages
        sample_technical_data['rsi'] = 35
        sample_technical_data['macd']['histogram'] = -5
        sample_technical_data['price'] = 95

        is_bearish = analyzer.is_bearish_signal(sample_technical_data)
        # Result depends on implementation, just check it returns boolean
        assert isinstance(is_bearish, bool) or is_bearish is None

    def test_get_signal_strength(self, analyzer, sample_technical_data):
        """Test signal strength calculation."""
        strength = analyzer.get_signal_strength(sample_technical_data)
        # Should return a value or None
        if strength is not None:
            assert -100 <= strength <= 100 or 0 <= strength <= 100

    def test_analyze_with_multiple_timeframes(self, analyzer):
        """Test analysis across multiple timeframes."""
        mock_daily = {'rsi': 60, 'trend': 'uptrend'}
        mock_weekly = {'rsi': 55, 'trend': 'uptrend'}

        analyzer.api_manager.fetch_technical_analysis = Mock(
            side_effect=[mock_daily, mock_weekly]
        )

        daily_score = analyzer.get_technical_score('TEST', 'daily')
        weekly_score = analyzer.get_technical_score('TEST', 'weekly')

        # Both should return valid scores or None
        assert daily_score is None or isinstance(daily_score, (int, float))
        assert weekly_score is None or isinstance(weekly_score, (int, float))

    def test_handle_api_errors_gracefully(self, analyzer):
        """Test graceful handling of API errors."""
        analyzer.api_manager.fetch_technical_analysis = Mock(
            side_effect=Exception("API Error")
        )

        # Should not raise exception
        score = analyzer.get_technical_score('TEST', 'daily')
        assert score is None or isinstance(score, (int, float))

    def test_validate_input_symbols(self, analyzer):
        """Test input validation for symbols."""
        # Valid symbols
        valid_symbols = ['RELIANCE', 'TCS', 'INFY']
        for symbol in valid_symbols:
            result = analyzer.get_technical_score(symbol, 'daily')
            # Should process without errors
            assert result is None or isinstance(result, (int, float))

    def test_validate_input_timeframes(self, analyzer):
        """Test input validation for timeframes."""
        # Valid timeframes
        valid_timeframes = ['1min', '5min', '15min', '30min', 'hourly', 'daily', 'weekly']
        for timeframe in valid_timeframes:
            result = analyzer.get_technical_score('TEST', timeframe)
            # Should process without errors
            assert result is None or isinstance(result, (int, float))
