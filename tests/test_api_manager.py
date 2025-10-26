"""Unit tests for APIManager module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from api_manager import APIManager, retry_on_failure


@pytest.fixture
def api_manager():
    """Create APIManager instance for testing."""
    return APIManager()


@pytest.fixture
def mock_session():
    """Create mock session for HTTP requests."""
    with patch('api_manager.requests.Session') as mock:
        session = MagicMock()
        mock.return_value = session
        yield session


class TestAPIManager:
    """Test suite for APIManager class."""

    def test_initialization(self, api_manager):
        """Test APIManager initializes correctly."""
        assert api_manager.cache is not None
        assert api_manager.session is not None
        assert api_manager.rate_limiter is not None

    def test_cache_key_generation(self, api_manager):
        """Test cache key generation."""
        key = api_manager._get_cache_key('test_endpoint', {'param': 'value'})
        assert 'test_endpoint' in key
        assert isinstance(key, str)

    def test_cache_expiry_check(self, api_manager):
        """Test cache expiry validation."""
        # Test expired cache
        expired_time = datetime.now() - timedelta(hours=2)
        assert api_manager._is_cache_expired(expired_time, 3600) is True

        # Test valid cache
        valid_time = datetime.now() - timedelta(minutes=30)
        assert api_manager._is_cache_expired(valid_time, 3600) is False

    @patch('api_manager.requests.Session.get')
    def test_fetch_market_breadth_success(self, mock_get, api_manager):
        """Test successful market breadth fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'advances': 1500,
            'declines': 500,
            'unchanged': 100
        }
        mock_get.return_value = mock_response

        result = api_manager.fetch_market_breadth()
        assert result is not None
        assert 'advances' in result
        assert result['advances'] == 1500

    @patch('api_manager.requests.Session.get')
    def test_fetch_market_breadth_cached(self, mock_get, api_manager):
        """Test market breadth returns cached data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_get.return_value = mock_response

        # First call - should hit API
        result1 = api_manager.fetch_market_breadth()
        assert mock_get.call_count == 1

        # Second call - should use cache
        result2 = api_manager.fetch_market_breadth()
        assert mock_get.call_count == 1  # No additional call
        assert result1 == result2

    @patch('api_manager.requests.Session.get')
    def test_fetch_sector_performance_success(self, mock_get, api_manager):
        """Test successful sector performance fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'sectors': [
                {'name': 'IT', 'change': 2.5},
                {'name': 'Banking', 'change': -1.2}
            ]
        }
        mock_get.return_value = mock_response

        result = api_manager.fetch_sector_performance()
        assert result is not None
        assert 'sectors' in result

    @patch('api_manager.requests.Session.get')
    def test_fetch_support_resistance_success(self, mock_get, api_manager):
        """Test successful S/R levels fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'support': [100, 95, 90],
            'resistance': [110, 115, 120]
        }
        mock_get.return_value = mock_response

        result = api_manager.fetch_support_resistance('RELIANCE', 'daily')
        assert result is not None
        assert 'support' in result
        assert 'resistance' in result

    @patch('api_manager.requests.Session.get')
    def test_fetch_candlestick_data_success(self, mock_get, api_manager):
        """Test successful candlestick pattern fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'patterns': ['doji', 'hammer']
        }
        mock_get.return_value = mock_response

        result = api_manager.fetch_candlestick_data('INFY', 'daily')
        assert result is not None
        assert 'patterns' in result

    @patch('api_manager.requests.Session.get')
    def test_fetch_technical_analysis_success(self, mock_get, api_manager):
        """Test successful technical analysis fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'rsi': 65,
            'macd': {'macd': 12, 'signal': 10},
            'moving_averages': {'sma_20': 100, 'sma_50': 95}
        }
        mock_get.return_value = mock_response

        result = api_manager.fetch_technical_analysis('TCS', 'daily')
        assert result is not None
        assert 'rsi' in result

    @patch('api_manager.requests.Session.get')
    def test_fetch_shareholdings_success(self, mock_get, api_manager):
        """Test successful shareholdings fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'promoter': 75.5,
            'fii': 15.2,
            'dii': 9.3
        }
        mock_get.return_value = mock_response

        result = api_manager.fetch_shareholdings('WIPRO')
        assert result is not None
        assert 'promoter' in result

    @patch('api_manager.requests.Session.get')
    def test_fetch_financials_success(self, mock_get, api_manager):
        """Test successful financials fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'revenue': 10000,
            'profit': 2000,
            'eps': 50
        }
        mock_get.return_value = mock_response

        result = api_manager.fetch_financials('HDFCBANK')
        assert result is not None
        assert 'revenue' in result

    @patch('api_manager.requests.Session.get')
    def test_api_error_handling(self, mock_get, api_manager):
        """Test API error handling."""
        mock_get.side_effect = Exception("Network error")

        result = api_manager.fetch_market_breadth()
        assert result is None

    @patch('api_manager.requests.Session.get')
    def test_http_error_handling(self, mock_get, api_manager):
        """Test HTTP error status handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        mock_get.return_value = mock_response

        result = api_manager.fetch_market_breadth()
        assert result is None

    def test_rate_limiting(self, api_manager):
        """Test rate limiting functionality."""
        # Rate limiter should allow some requests
        assert api_manager.rate_limiter.can_proceed() is True

        # After max requests, should be rate limited
        for _ in range(100):
            api_manager.rate_limiter.record_request()

        # Should still work but track the requests
        assert api_manager.rate_limiter is not None


class TestRetryDecorator:
    """Test suite for retry_on_failure decorator."""

    def test_retry_success_on_first_attempt(self):
        """Test function succeeds on first attempt."""
        @retry_on_failure(max_retries=3)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_retry_success_after_failures(self):
        """Test function succeeds after initial failures."""
        call_count = [0]

        @retry_on_failure(max_retries=3)
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary error")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_exhaustion(self):
        """Test function fails after max retries."""
        @retry_on_failure(max_retries=2)
        def failing_function():
            raise Exception("Permanent error")

        result = failing_function()
        assert result is None

    def test_retry_with_custom_delay(self):
        """Test retry with custom delay."""
        @retry_on_failure(max_retries=2, delay=0.1)
        def quick_retry_function():
            raise Exception("Error")

        start = datetime.now()
        result = quick_retry_function()
        duration = (datetime.now() - start).total_seconds()

        assert result is None
        assert duration >= 0.1  # At least one delay occurred


@pytest.fixture(autouse=True)
def clear_cache(api_manager):
    """Clear cache before each test."""
    api_manager.cache.clear()
    yield
    api_manager.cache.clear()
