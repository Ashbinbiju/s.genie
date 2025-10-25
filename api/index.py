"""
Vercel serverless entry point for StockGenie Pro
This file exports the Flask app for Vercel's WSGI handler
"""
import sys
import os

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime
from api_manager import APIManager
from market_analyzer import MarketAnalyzer
from technical_analyzer import TechnicalAnalyzer
from stock_scanner import StockScanner
from alert_manager import AlertManager
from sectors import SECTORS, INDUSTRY_MAP
from config import FLASK_SECRET_KEY
from logger import logger
from validators import validate_symbol, validate_timeframe, ValidationError

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
app.config['SECRET_KEY'] = FLASK_SECRET_KEY

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Initialize components (without SocketIO for serverless)
api_manager = APIManager()
market_analyzer = MarketAnalyzer(api_manager)
technical_analyzer = TechnicalAnalyzer(api_manager)
scanner = StockScanner(api_manager)
alert_manager = AlertManager(api_manager)

# Build watchlist from all sectors
WATCHLIST = []
seen_stocks = set()
for sector, stocks in INDUSTRY_MAP.items():
    for stock in stocks:
        if stock not in seen_stocks:
            WATCHLIST.append(stock)
            seen_stocks.add(stock)

logger.info(f"Vercel: Loaded {len(WATCHLIST)} stocks from {len(SECTORS)} sectors")

# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/market-analysis')
def market_analysis():
    """Market analysis page"""
    return render_template('market_analysis.html')

@app.route('/swing-trading')
def swing_trading():
    """Swing trading opportunities page"""
    return render_template('swing_trading.html')

@app.route('/intraday-trading')
def intraday_trading():
    """Intraday trading opportunities page"""
    return render_template('intraday_trading.html')

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    """Stock detail page"""
    return render_template('stock_detail.html', symbol=symbol)

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/market-health')
@limiter.limit("30 per minute")
def market_health():
    """Get current market health status"""
    try:
        health = market_analyzer.get_market_health()
        return jsonify(health)
    except Exception as e:
        logger.error(f"Market health error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/swing-scan')
@limiter.limit("20 per minute")
def swing_scan():
    """Scan for swing trading opportunities"""
    try:
        # Get filters from query params
        min_score = float(request.args.get('min_score', 60))
        max_rsi = float(request.args.get('max_rsi', 70))
        min_volume = float(request.args.get('min_volume', 0))
        
        results = scanner.scan_swing_opportunities(
            WATCHLIST[:50],  # Limit for serverless
            min_score=min_score,
            max_rsi=max_rsi,
            min_volume=min_volume
        )
        return jsonify(results)
    except Exception as e:
        logger.error(f"Swing scan error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/intraday-scan')
@limiter.limit("20 per minute")
def intraday_scan():
    """Scan for intraday trading opportunities"""
    try:
        min_score = float(request.args.get('min_score', 60))
        max_rsi = float(request.args.get('max_rsi', 70))
        min_volume = float(request.args.get('min_volume', 0))
        
        results = scanner.scan_intraday_opportunities(
            WATCHLIST[:50],  # Limit for serverless
            min_score=min_score,
            max_rsi=max_rsi,
            min_volume=min_volume
        )
        return jsonify(results)
    except Exception as e:
        logger.error(f"Intraday scan error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>')
@limiter.limit("30 per minute")
def get_stock_data(symbol):
    """Get comprehensive stock data"""
    try:
        validate_symbol(symbol)
        
        # Fetch data
        quote = api_manager.fetch_quote(symbol)
        technical = technical_analyzer.analyze_stock(symbol)
        patterns = technical_analyzer.detect_candlestick_patterns(symbol)
        
        return jsonify({
            'symbol': symbol,
            'quote': quote,
            'technical': technical,
            'patterns': patterns,
            'timestamp': datetime.now().isoformat()
        })
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Stock data error for {symbol}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/chart')
@limiter.limit("30 per minute")
def get_chart_data(symbol):
    """Get historical chart data"""
    try:
        validate_symbol(symbol)
        timeframe = request.args.get('timeframe', '1d')
        validate_timeframe(timeframe)
        
        data = api_manager.fetch_historical_data(symbol, timeframe)
        return jsonify(data)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Chart data error for {symbol}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/sectors')
@limiter.limit("30 per minute")
def get_sectors():
    """Get sector performance"""
    try:
        performance = api_manager.fetch_sector_performance()
        return jsonify(performance)
    except Exception as e:
        logger.error(f"Sector data error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'environment': 'vercel-serverless'
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {e}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return jsonify({'error': 'Internal server error'}), 500

# Export the app for Vercel
# Vercel will look for an 'app' variable in this file
# This is the WSGI application that Vercel will run
