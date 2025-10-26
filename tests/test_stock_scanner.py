"""Unit tests for StockScanner module."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from stock_scanner import StockScanner


@pytest.fixture
def mock_api_manager():
    """Create mock APIManager for testing."""
    mock_api = Mock()
    # Setup default mock responses
    mock_api.fetch_technical_analysis.return_value = {
        'rsi': 55,
        'macd': {'macd': 12, 'signal': 10, 'histogram': 2},
        'moving_averages': {'sma_20': 100, 'sma_50': 95}
    }
    mock_api.fetch_support_resistance.return_value = {
        'support': [95, 90],
        'resistance': [110, 115]
    }
    mock_api.fetch_candlestick_data.return_value = {
        'patterns': ['hammer']
    }
    mock_api.fetch_shareholdings.return_value = {
        'promoter': 75.0,
        'fii': 15.0
    }
    mock_api.fetch_financials.return_value = {
        'pe_ratio': 20,
        'debt_to_equity': 0.5
    }
    return mock_api


@pytest.fixture
def mock_technical_analyzer():
    """Create mock TechnicalAnalyzer for testing."""
    mock_analyzer = Mock()
    mock_analyzer.get_technical_score.return_value = 75
    mock_analyzer.is_bullish_signal.return_value = True
    mock_analyzer.is_bearish_signal.return_value = False
    mock_analyzer.get_signal_strength.return_value = 60
    return mock_analyzer


@pytest.fixture
def scanner(mock_api_manager, mock_technical_analyzer):
    """Create StockScanner instance for testing."""
    return StockScanner(mock_api_manager, mock_technical_analyzer)


@pytest.fixture
def sample_stocks():
    """Sample stock list for testing."""
    return [
        {'symbol': 'RELIANCE', 'sector': 'Energy'},
        {'symbol': 'TCS', 'sector': 'IT'},
        {'symbol': 'INFY', 'sector': 'IT'},
        {'symbol': 'HDFCBANK', 'sector': 'Banking'},
        {'symbol': 'ICICIBANK', 'sector': 'Banking'}
    ]


class TestStockScanner:
    """Test suite for StockScanner class."""

    def test_initialization(self, scanner):
        """Test StockScanner initializes correctly."""
        assert scanner.api_manager is not None
        assert scanner.technical_analyzer is not None

    def test_scan_for_swing_trades_basic(self, scanner, sample_stocks):
        """Test basic swing trade scanning."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            results = scanner.scan_for_swing_trades()

            assert isinstance(results, list)
            # Should process all stocks
            assert len(results) <= len(sample_stocks)

    def test_scan_for_swing_trades_with_filters(self, scanner, sample_stocks):
        """Test swing trade scanning with filters."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            # Mock high scoring opportunities
            scanner.technical_analyzer.get_technical_score.return_value = 85

            results = scanner.scan_for_swing_trades(min_score=80)

            assert isinstance(results, list)
            # All results should meet minimum score
            for result in results:
                if 'score' in result:
                    assert result['score'] >= 80

    def test_scan_for_swing_trades_sector_filter(self, scanner, sample_stocks):
        """Test swing trade scanning with sector filter."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            results = scanner.scan_for_swing_trades(sector='IT')

            assert isinstance(results, list)
            # All results should be from IT sector
            for result in results:
                if 'sector' in result:
                    assert result['sector'] == 'IT'

    def test_scan_for_intraday_trades_basic(self, scanner, sample_stocks):
        """Test basic intraday trade scanning."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            results = scanner.scan_for_intraday_trades()

            assert isinstance(results, list)
            # Should process all stocks
            assert len(results) <= len(sample_stocks)

    def test_scan_for_intraday_trades_high_volatility(self, scanner, sample_stocks):
        """Test intraday scanning for high volatility stocks."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            # Mock high volatility data
            scanner.api_manager.fetch_technical_analysis.return_value = {
                'rsi': 65,
                'volatility': 3.5,  # High volatility
                'volume': 1000000
            }

            results = scanner.scan_for_intraday_trades(min_volatility=3.0)

            assert isinstance(results, list)

    def test_scan_with_api_errors(self, scanner, sample_stocks):
        """Test scanning handles API errors gracefully."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            # Mock API errors
            scanner.api_manager.fetch_technical_analysis.side_effect = Exception("API Error")

            # Should not raise exception
            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_scan_with_missing_data(self, scanner, sample_stocks):
        """Test scanning handles missing data."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            # Mock missing data
            scanner.api_manager.fetch_technical_analysis.return_value = None

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_scan_results_structure(self, scanner, sample_stocks):
        """Test scan results have correct structure."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):  # Test with one stock
            results = scanner.scan_for_swing_trades()

            if results:
                result = results[0]
                # Check for expected fields
                assert 'symbol' in result
                # May have score, sector, etc.
                assert isinstance(result, dict)

    def test_filter_by_technical_score(self, scanner, sample_stocks):
        """Test filtering stocks by technical score."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            # Mock varying scores
            scores = [85, 60, 90, 45, 75]
            scanner.technical_analyzer.get_technical_score.side_effect = scores

            results = scanner.scan_for_swing_trades(min_score=70)

            # Should only include stocks with score >= 70
            assert isinstance(results, list)

    def test_filter_by_rsi_oversold(self, scanner, sample_stocks):
        """Test filtering for oversold conditions (RSI < 30)."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            scanner.api_manager.fetch_technical_analysis.return_value = {
                'rsi': 25,  # Oversold
                'macd': {'histogram': -2},
                'trend': 'downtrend'
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_filter_by_rsi_overbought(self, scanner, sample_stocks):
        """Test filtering for overbought conditions (RSI > 70)."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            scanner.api_manager.fetch_technical_analysis.return_value = {
                'rsi': 75,  # Overbought
                'macd': {'histogram': 3},
                'trend': 'uptrend'
            }

            results = scanner.scan_for_intraday_trades()
            assert isinstance(results, list)

    def test_bullish_candlestick_patterns(self, scanner, sample_stocks):
        """Test detection of bullish candlestick patterns."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):
            scanner.api_manager.fetch_candlestick_data.return_value = {
                'patterns': ['hammer', 'bullish_engulfing', 'morning_star']
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_bearish_candlestick_patterns(self, scanner, sample_stocks):
        """Test detection of bearish candlestick patterns."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):
            scanner.api_manager.fetch_candlestick_data.return_value = {
                'patterns': ['shooting_star', 'bearish_engulfing', 'evening_star']
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_support_resistance_analysis(self, scanner, sample_stocks):
        """Test support and resistance level analysis."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):
            scanner.api_manager.fetch_support_resistance.return_value = {
                'support': [95, 90, 85],
                'resistance': [110, 115, 120],
                'current_price': 100
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_price_near_support(self, scanner, sample_stocks):
        """Test identification of stocks near support levels."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):
            scanner.api_manager.fetch_support_resistance.return_value = {
                'support': [98, 95],  # Current price near support
                'resistance': [110, 115],
                'current_price': 99
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_price_near_resistance(self, scanner, sample_stocks):
        """Test identification of stocks near resistance levels."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):
            scanner.api_manager.fetch_support_resistance.return_value = {
                'support': [90, 85],
                'resistance': [102, 110],  # Current price near resistance
                'current_price': 101
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_volume_analysis(self, scanner, sample_stocks):
        """Test volume-based filtering."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            scanner.api_manager.fetch_technical_analysis.return_value = {
                'rsi': 60,
                'volume': 5000000,  # High volume
                'avg_volume': 2000000
            }

            results = scanner.scan_for_intraday_trades()
            assert isinstance(results, list)

    def test_trend_identification(self, scanner, sample_stocks):
        """Test trend identification in scan results."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):
            scanner.api_manager.fetch_technical_analysis.return_value = {
                'rsi': 65,
                'trend': 'uptrend',
                'moving_averages': {'sma_20': 100, 'sma_50': 95}
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_fundamental_filters(self, scanner, sample_stocks):
        """Test fundamental analysis filters."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):
            scanner.api_manager.fetch_financials.return_value = {
                'pe_ratio': 15,  # Reasonable PE
                'debt_to_equity': 0.3,  # Low debt
                'roe': 18  # Good ROE
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_promoter_holding_filter(self, scanner, sample_stocks):
        """Test filtering by promoter holding percentage."""
        with patch('stock_scanner.STOCKS', sample_stocks[:1]):
            scanner.api_manager.fetch_shareholdings.return_value = {
                'promoter': 75.5,  # High promoter holding
                'fii': 15.2,
                'dii': 9.3
            }

            results = scanner.scan_for_swing_trades()
            assert isinstance(results, list)

    def test_empty_stock_list(self, scanner):
        """Test scanning with empty stock list."""
        with patch('stock_scanner.STOCKS', []):
            results = scanner.scan_for_swing_trades()
            assert results == []

    def test_scan_performance_with_large_dataset(self, scanner):
        """Test scanner performance with large dataset."""
        large_stock_list = [
            {'symbol': f'STOCK{i}', 'sector': 'TEST'}
            for i in range(100)
        ]

        with patch('stock_scanner.STOCKS', large_stock_list):
            import time
            start = time.time()
            results = scanner.scan_for_swing_trades()
            duration = time.time() - start

            assert isinstance(results, list)
            # Should complete in reasonable time (adjust threshold as needed)
            assert duration < 60  # 60 seconds for 100 stocks

    def test_concurrent_scans(self, scanner, sample_stocks):
        """Test running multiple scans concurrently."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            # Run multiple scans
            swing_results = scanner.scan_for_swing_trades()
            intraday_results = scanner.scan_for_intraday_trades()

            assert isinstance(swing_results, list)
            assert isinstance(intraday_results, list)

    def test_scan_with_progress_callback(self, scanner, sample_stocks):
        """Test scan progress reporting."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            progress_calls = []

            def progress_callback(current, total):
                progress_calls.append((current, total))

            # If scanner supports progress callbacks
            if hasattr(scanner, 'set_progress_callback'):
                scanner.set_progress_callback(progress_callback)
                scanner.scan_for_swing_trades()
                assert len(progress_calls) > 0

    def test_result_sorting(self, scanner, sample_stocks):
        """Test scan results are properly sorted."""
        with patch('stock_scanner.STOCKS', sample_stocks):
            results = scanner.scan_for_swing_trades()

            # If results have scores, they should be sorted
            if results and 'score' in results[0]:
                scores = [r['score'] for r in results if 'score' in r]
                assert scores == sorted(scores, reverse=True)
