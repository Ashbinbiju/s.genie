# 📈 StockGenie Pro - Advanced Stock Scanner

A production-ready stock analysis platform for swing and intraday trading with real-time scanning, intelligent caching, and professional-grade features.

## ✨ Key Features

### 🎯 Core Capabilities
- **Background Scanning** - Scan all 1027 stocks asynchronously without blocking UI
- **Intelligent Caching** - Redis-ready caching with configurable TTL (5-30 min)
- **Pagination** - Handle large datasets with smooth page navigation
- **Real-time Progress** - WebSocket updates during long-running scans
- **Quick & Full Scan Modes** - Choose between 50-stock quick scan or full 1027-stock deep scan
- **Technical Analysis** - RSI, MACD, ADX, Volume, Candlestick patterns via Streak API
- **Support & Resistance** - Automated pivot points and S/R levels
- **Risk Management** - Automatic R:R calculation with configurable thresholds
- **Shareholding Data** - Promoter, FII, DII holdings via Zerodha API
- **Financial Analysis** - P&L, Balance Sheet, Cash Flow, Ratios

### 🌐 Web Interface
- **Modern UI** - Bootstrap 5 with gradient effects and smooth animations
- **WebSocket Live Updates** - Real-time scan progress without polling
- **Responsive Design** - Mobile-first, works on all devices
- **Progress Bars** - Visual feedback during background operations
- **Error Handling** - Graceful degradation with user-friendly messages
- **Rate Limiting** - Built-in protection against API abuse

### 🔒 Security & Production-Ready
- **Environment Variables** - Secure credential management
- **Input Validation** - Sanitized inputs to prevent injection attacks
- **Rate Limiting** - Flask-Limiter for API endpoint protection
- **Structured Logging** - Rotating log files with configurable levels
- **Health Check Endpoint** - `/health` for monitoring systems
- **Connection Pooling** - Optimized HTTP requests with session reuse
- **Retry Logic** - Automatic retry on transient failures

## 📊 Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Frontend  │─────▶│   Flask API  │─────▶│  API Manager│
│ (Templates) │◀─────│   + Socket   │◀─────│ (Caching)   │
└─────────────┘      └──────────────┘      └─────────────┘
                           │                       │
                           ├─────────────┬─────────┤
                           ▼             ▼         ▼
                    ┌─────────┐   ┌──────────┐  ┌────────┐
                    │ Scanner │   │ Analyzer │  │ Logger │
                    └─────────┘   └──────────┘  └────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip package manager
- (Optional) Redis for persistent caching

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Ashbinbiju/s.genie.git
cd s.genie
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy template
cp .env.template .env

# Edit with your credentials (NEVER commit .env!)
nano .env
```

**Required Environment Variables:**
**Required Environment Variables:**

```bash
# Telegram (Optional - for alerts)
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token_from_@BotFather
TELEGRAM_CHAT_ID=your_chat_id

# SmartAPI (Optional - for live trading)
CLIENT_ID=your_smartapi_client_id
PASSWORD=your_password
TOTP_SECRET=your_totp_secret

# Flask Security
FLASK_SECRET_KEY=generate-a-random-secret-key-here
FLASK_DEBUG=false  # ALWAYS false in production
```

⚠️ **SECURITY**: See [SECURITY_NOTICE.md](SECURITY_NOTICE.md) for credential management best practices.

### 3. Run the Application

**Development Mode:**
```bash
python3 app.py
```

**Production Mode (with Gunicorn):**
```bash
pip install gunicorn eventlet
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
```

**Using Run Script:**
```bash
chmod +x run.sh
./run.sh
```

### 4. Access the Dashboard

Open your browser to:
- http://localhost:5000 - Main Dashboard
- http://localhost:5000/swing-trading - Swing Opportunities
- http://localhost:5000/intraday-trading - Intraday Opportunities
- http://localhost:5000/health - Health Check

## 📖 Usage Guide

### Quick Scan (Fast - 50 stocks)
1. Visit `/swing-trading` or `/intraday-trading`
2. Click **"Quick Scan"** button
3. Wait ~60 seconds for results
4. Browse paginated results (20 per page)

### Full Scan (Comprehensive - 1027 stocks)
1. Click **"Full Scan (All Stocks)"** button
2. Progress bar shows live updates
3. Scan runs in background (~15-20 min)
4. You can navigate away and come back
5. Results cached for 30 minutes

### API Endpoints

#### Swing Trading Scan
```bash
# Quick scan (50 stocks)
curl "http://localhost:5000/api/swing-scan?page=1&limit=20"

# Full scan (all stocks)
curl "http://localhost:5000/api/swing-scan?full_scan=true"

# Force rescan (bypass cache)
curl "http://localhost:5000/api/swing-scan?force_rescan=true"
```

#### Scan Status
```bash
curl "http://localhost:5000/api/scan-status"
```

#### Single Stock Analysis
```bash
# Get complete stock data
curl "http://localhost:5000/api/stock-complete/RELIANCE-EQ?timeframe=day"

# Get shareholdings
curl "http://localhost:5000/api/shareholdings/RELIANCE-EQ"

# Get financials
curl "http://localhost:5000/api/financials/RELIANCE-EQ"
```

## 🧪 Testing

```bash
# Run unit tests (when available)
pytest

# Run linter
flake8 app.py api_manager.py --max-line-length=120

# Test API endpoints
python test_apis.py
```

## 📁 Project Structure

```
s.genie/
├── app.py                  # Main Flask application
├── api_manager.py          # API client with caching & rate limiting
├── technical_analyzer.py   # Technical indicator calculations
├── stock_scanner.py        # Stock scanning logic
├── market_analyzer.py      # Market health analysis
├── alert_manager.py        # Telegram alert system
├── logger.py               # Structured logging setup
├── validators.py           # Input validation utilities
├── config.py               # Configuration management
├── sectors.py              # Industry classifications
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── swing_trading.html
│   └── intraday_trading.html
├── logs/                   # Application logs (auto-created)
├── .env.template           # Environment variable template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## ⚙️ Configuration

### Scan Configuration (in `app.py`)
```python
SCAN_CONFIG = {
    "quick_scan_limit": 50,  # Stocks in quick scan
    "batch_size": 20,        # Stocks per batch
    "full_scan": False       # Toggle for full scan
}
```

### Trading Parameters (in `config.py`)
```python
TRADING_CONFIG = {
    "swing": {
        "timeframe": "day",
        "min_score": 50,
        "risk_reward_ratio": 1.0,
        "max_positions": 5
    },
    "intraday": {
        "timeframe": "5min",
        "min_score": 50,
        "risk_reward_ratio": 1.0,
        "max_positions": 3
    }
}
```

### Rate Limits (in `config.py`)
```python
SMARTAPI_RATE_LIMITS = {
    "requests_per_second": 3,
    "requests_per_minute": 180,
    "requests_per_hour": 5000,
    "min_delay_between_requests": 0.35
}
```

## 🔧 Advanced Features

### Background Scanning
Scans run in separate threads with progress tracking:
- Batched processing (20 stocks at a time)
- WebSocket progress updates every batch
- Thread-safe cache updates with locks
- Graceful error handling

### Caching Strategy
- **In-memory cache** with TTL expiration
- **Connection pooling** for API requests
- **Retry logic** on transient failures
- **Ready for Redis** (abstracted cache interface)

### Logging
Structured logging to both console and rotating files:
```python
from logger import logger

logger.info("Scan completed")
logger.warning("Rate limit approaching")
logger.error("API error", exc_info=True)
```

### Input Validation
All user inputs are validated and sanitized:
```python
from validators import validate_symbol, validate_timeframe

symbol = validate_symbol("RELIANCE-EQ")  # Sanitized
timeframe = validate_timeframe("5min")   # Validated
```

## 📚 Documentation

- **[API Architecture](API_ARCHITECTURE.txt)** - Detailed API integration docs
- **[Caching & Pagination Guide](CACHING_PAGINATION_GUIDE.md)** - Advanced features
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Recent improvements
- **[Security Notice](SECURITY_NOTICE.md)** - Credential management

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Workflow
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install flake8 pylint pytest

# Run linters
flake8 *.py --max-line-length=120

# Run tests
pytest

# Start development server
python3 app.py
```

## 📄 License

This project is private and proprietary. All rights reserved.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Ashbinbiju/s.genie/issues)
- **Documentation**: See `/docs` folder
- **Health Check**: http://localhost:5000/health

## 🙏 Acknowledgments

- **Streak API** - Technical analysis data
- **Zerodha** - Shareholding and financial data
- **Bootstrap** - UI framework
- **Flask** - Web framework
- **Socket.IO** - Real-time communications

---

**⚠️ Disclaimer**: This tool is for educational and informational purposes only. Not financial advice. Trade at your own risk.

./run.sh
```
Then open `http://localhost:5000` in your browser.

**Option B: Command Line**
```bash
python main.py
```

**Option C: Direct Python**
```bash
python app.py
```

## 📊 Usage Guide

### Web Dashboard

1. **Dashboard** (`/`)
   - Market health overview
   - Top swing & intraday opportunities
   - Quick stats and charts

2. **Market Analysis** (`/market-analysis`)
   - Detailed market breadth
   - Sector performance
   - Trending indices
   - Auto-refresh every 5 minutes

3. **Swing Trading** (`/swing-trading`)
   - Full swing opportunity details
   - R:R ratios and targets
   - Technical scores

4. **Intraday Trading** (`/intraday-trading`)
   - Intraday opportunities
   - Volume analysis
   - Market-hour auto-refresh

5. **Stock Detail** (`/stock-detail/<symbol>`)
   - Live price charts
   - S/R levels
   - Technical indicators
   - Multiple timeframes

### API Endpoints
