from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import threading
import time
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

app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*")

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Initialize components
api_manager = APIManager()
market_analyzer = MarketAnalyzer(api_manager)
technical_analyzer = TechnicalAnalyzer(api_manager)
scanner = StockScanner(api_manager)
alert_manager = AlertManager(api_manager)

# Build watchlist from all sectors in INDUSTRY_MAP
# Collect all unique stocks across all sectors
WATCHLIST = []
seen_stocks = set()
for sector, stocks in INDUSTRY_MAP.items():
    for stock in stocks:
        if stock not in seen_stocks:
            WATCHLIST.append(stock)
            seen_stocks.add(stock)

print(f"📊 Loaded {len(WATCHLIST)} stocks from {len(SECTORS)} sectors")
print(f"📁 Sectors: {', '.join(SECTORS[:5])}{'...' if len(SECTORS) > 5 else ''}")

logger.info(f"Loaded {len(WATCHLIST)} stocks from {len(SECTORS)} sectors")

# Advanced cache for results with pagination
cache = {
    "market_health": None,
    "swing_opportunities": [],
    "intraday_opportunities": [],
    "last_update": None,
    "scan_in_progress": False,
    "scan_status": "idle",
    "scan_progress": 0,  # Percentage 0-100
    "scanned_count": 0,
    "total_count": 0,
    "scan_type": None,  # 'swing' or 'intraday'
    "scan_start_time": None
}

# Scanning configuration
SCAN_CONFIG = {
    "quick_scan_limit": 50,  # Quick scan for immediate results
    "batch_size": 20,  # Process stocks in batches
    "full_scan": False  # Toggle for full scan
}

# Lock for thread-safe cache updates
cache_lock = threading.Lock()

@app.route('/health')
@limiter.exempt
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "cache": "ok" if cache else "empty",
            "scan_status": cache.get("scan_status", "unknown"),
            "watchlist_size": len(WATCHLIST)
        }
    }), 200

@app.route('/')
def index():
    """Dashboard home page"""
    return render_template('index.html')

@app.route('/market-analysis')
def market_analysis():
    """Market analysis page"""
    return render_template('market_analysis.html')

@app.route('/swing-trading')
def swing_trading():
    """Swing trading page"""
    return render_template('swing_trading.html')

@app.route('/intraday-trading')
def intraday_trading():
    """Intraday trading page"""
    return render_template('intraday_trading.html')

@app.route('/stock-detail/<symbol>')
def stock_detail(symbol):
    """Single stock detail page"""
    return render_template('stock_detail.html', symbol=symbol)

@app.route('/api/market-health')
def get_market_health():
    """API endpoint for market health"""
    try:
        health = market_analyzer.get_market_health()
        bullish_sectors = market_analyzer.get_bullish_sectors()
        trending_indices = market_analyzer.get_trending_indices()
        
        cache["market_health"] = health
        cache["last_update"] = datetime.now().isoformat()
        
        return jsonify({
            "success": True,
            "data": {
                "health": health,
                "bullish_sectors": bullish_sectors,
                "trending_indices": trending_indices
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def background_scan_worker(scan_type='swing', symbols=None):
    """Background worker for scanning stocks"""
    with cache_lock:
        if cache["scan_in_progress"]:
            logger.warning("Scan already in progress, skipping new scan request")
            return

        cache["scan_in_progress"] = True
        cache["scan_status"] = "running"
        cache["scan_type"] = scan_type
        cache["scan_start_time"] = datetime.now()
        cache["scanned_count"] = 0

    if symbols is None:
        symbols = WATCHLIST

    cache["total_count"] = len(symbols)

    try:
        batch_size = SCAN_CONFIG["batch_size"]
        opportunities = []

        logger.info(f"Starting {scan_type} scan for {len(symbols)} stocks...")

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]

            # Scan batch
            if scan_type == 'swing':
                batch_results = scanner.scan_for_swing_trades(batch)
            else:  # intraday
                batch_results = scanner.scan_for_intraday_trades(batch)

            opportunities.extend(batch_results)

            # Update progress
            with cache_lock:
                cache["scanned_count"] = min(i + batch_size, len(symbols))
                cache["scan_progress"] = int((cache["scanned_count"] / cache["total_count"]) * 100)

            # Emit progress via WebSocket
            socketio.emit('scan_progress', {
                'type': scan_type,
                'current': cache["scanned_count"],
                'total': cache["total_count"],
                'percent': cache["scan_progress"],
                'opportunities_found': len(opportunities)
            })

            logger.info(f"Progress: {cache['scanned_count']}/{cache['total_count']} ({cache['scan_progress']}%) - Found {len(opportunities)} opportunities")

            # Small delay to avoid overwhelming the system
            time.sleep(0.5)

        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Update cache
        with cache_lock:
            if scan_type == 'swing':
                cache["swing_opportunities"] = opportunities
            else:
                cache["intraday_opportunities"] = opportunities

            cache["last_update"] = datetime.now()
            cache["scan_in_progress"] = False
            cache["scan_status"] = "completed"
            cache["scan_progress"] = 100

        # Emit completion
        socketio.emit('scan_complete', {
            'type': scan_type,
            'total_opportunities': len(opportunities),
            'scan_duration': (datetime.now() - cache["scan_start_time"]).total_seconds()
        })

        logger.info(f"{scan_type.upper()} scan completed: {len(opportunities)} opportunities found")

    except Exception as e:
        logger.error(f"Error in background scan: {str(e)}", exc_info=True)
        with cache_lock:
            cache["scan_in_progress"] = False
            cache["scan_status"] = "error"

        socketio.emit('scan_error', {
            'type': scan_type,
            'error': str(e)
        })

@app.route('/api/swing-scan')
def swing_scan():
    """Get swing trading opportunities with pagination"""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    full_scan = request.args.get('full_scan', 'false').lower() == 'true'
    force_rescan = request.args.get('force_rescan', 'false').lower() == 'true'
    
    # Validate pagination params
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:
        limit = 20
    
    # Check if scan is in progress
    if cache["scan_in_progress"]:
        # Return current progress status
        opportunities = cache.get("swing_opportunities", [])
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_data = opportunities[start_idx:end_idx]
        
        return jsonify({
            "success": True,
            "scan_in_progress": True,
            "scan_status": cache["scan_status"],
            "scan_progress": cache["scan_progress"],
            "scanned_count": cache["scanned_count"],
            "total_count": cache["total_count"],
            "data": paginated_data,
            "count": len(paginated_data),
            "pagination": {
                "page": page,
                "limit": limit,
                "total_items": len(opportunities),
                "total_pages": (len(opportunities) + limit - 1) // limit if opportunities else 0,
                "has_next": end_idx < len(opportunities),
                "has_prev": page > 1
            },
            "message": "Scan in progress, showing partial results..."
        })
    
    # Check cache validity (5 min for quick scan, 30 min for full scan)
    cache_valid_seconds = 1800 if full_scan else 300
    cache_valid = (
        cache["last_update"] 
        and cache["swing_opportunities"]
        and (datetime.now() - cache["last_update"]).seconds < cache_valid_seconds
        and not force_rescan
    )
    
    # Start background scan if needed
    if not cache_valid or (full_scan and len(cache.get("swing_opportunities", [])) < 100):
        symbols = WATCHLIST if full_scan else WATCHLIST[:SCAN_CONFIG["quick_scan_limit"]]
        thread = threading.Thread(target=background_scan_worker, args=('swing', symbols), daemon=True)
        thread.start()
        
        # Return immediate response
        return jsonify({
            "success": True,
            "scan_started": True,
            "scan_type": "full" if full_scan else "quick",
            "total_stocks": len(symbols),
            "message": "Scan started in background. Results will be available shortly.",
            "scan_in_progress": True,
            "data": [],
            "count": 0
        })
    
    # Return paginated cached results
    opportunities = cache["swing_opportunities"]
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = opportunities[start_idx:end_idx]
    
    return jsonify({
        "success": True,
        "data": paginated_data,
        "count": len(paginated_data),
        "pagination": {
            "page": page,
            "limit": limit,
            "total_items": len(opportunities),
            "total_pages": (len(opportunities) + limit - 1) // limit,
            "has_next": end_idx < len(opportunities),
            "has_prev": page > 1
        },
        "scanned": cache.get("scanned_count", len(opportunities)),
        "total_stocks": len(WATCHLIST),
        "last_update": cache["last_update"].isoformat() if cache["last_update"] else None,
        "scan_in_progress": False
    })

@app.route('/api/intraday-scan')
def intraday_scan():
    """Get intraday trading opportunities with pagination"""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    full_scan = request.args.get('full_scan', 'false').lower() == 'true'
    force_rescan = request.args.get('force_rescan', 'false').lower() == 'true'
    
    # Validate pagination params
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:
        limit = 20
    
    # Check if scan is in progress
    if cache["scan_in_progress"] and cache["scan_type"] == 'intraday':
        # Return current progress status
        opportunities = cache.get("intraday_opportunities", [])
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_data = opportunities[start_idx:end_idx]
        
        return jsonify({
            "success": True,
            "scan_in_progress": True,
            "scan_status": cache["scan_status"],
            "scan_progress": cache["scan_progress"],
            "scanned_count": cache["scanned_count"],
            "total_count": cache["total_count"],
            "data": paginated_data,
            "count": len(paginated_data),
            "pagination": {
                "page": page,
                "limit": limit,
                "total_items": len(opportunities),
                "total_pages": (len(opportunities) + limit - 1) // limit if opportunities else 0,
                "has_next": end_idx < len(opportunities),
                "has_prev": page > 1
            },
            "message": "Scan in progress, showing partial results..."
        })
    
    # Check cache validity (3 min for quick scan, 15 min for full scan)
    cache_valid_seconds = 900 if full_scan else 180
    cache_valid = (
        cache["last_update"] 
        and cache["intraday_opportunities"]
        and (datetime.now() - cache["last_update"]).seconds < cache_valid_seconds
        and not force_rescan
    )
    
    # Start background scan if needed
    if not cache_valid or (full_scan and len(cache.get("intraday_opportunities", [])) < 100):
        symbols = WATCHLIST if full_scan else WATCHLIST[:SCAN_CONFIG["quick_scan_limit"]]
        thread = threading.Thread(target=background_scan_worker, args=('intraday', symbols), daemon=True)
        thread.start()
        
        # Return immediate response
        return jsonify({
            "success": True,
            "scan_started": True,
            "scan_type": "full" if full_scan else "quick",
            "total_stocks": len(symbols),
            "message": "Scan started in background. Results will be available shortly.",
            "scan_in_progress": True,
            "data": [],
            "count": 0
        })
    
    # Return paginated cached results
    opportunities = cache["intraday_opportunities"]
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_data = opportunities[start_idx:end_idx]
    
    return jsonify({
        "success": True,
        "data": paginated_data,
        "count": len(paginated_data),
        "pagination": {
            "page": page,
            "limit": limit,
            "total_items": len(opportunities),
            "total_pages": (len(opportunities) + limit - 1) // limit,
            "has_next": end_idx < len(opportunities),
            "has_prev": page > 1
        },
        "scanned": cache.get("scanned_count", len(opportunities)),
        "total_stocks": len(WATCHLIST),
        "last_update": cache["last_update"].isoformat() if cache["last_update"] else None,
        "scan_in_progress": False
    })

@app.route('/api/scan-status')
def scan_status():
    """Get current scan status"""
    return jsonify({
        "success": True,
        "scan_in_progress": cache["scan_in_progress"],
        "scan_status": cache["scan_status"],
        "scan_type": cache.get("scan_type"),
        "scan_progress": cache["scan_progress"],
        "scanned_count": cache["scanned_count"],
        "total_count": cache["total_count"],
        "scan_start_time": cache["scan_start_time"].isoformat() if cache.get("scan_start_time") else None,
        "last_update": cache["last_update"].isoformat() if cache.get("last_update") else None,
        "swing_opportunities_count": len(cache.get("swing_opportunities", [])),
        "intraday_opportunities_count": len(cache.get("intraday_opportunities", []))
    })

@app.route('/api/scan-config', methods=['GET', 'POST'])
def scan_config():
    """Get or update scan configuration"""
    if request.method == 'POST':
        data = request.get_json()
        if 'quick_scan_limit' in data:
            SCAN_CONFIG['quick_scan_limit'] = max(10, min(200, int(data['quick_scan_limit'])))
        if 'batch_size' in data:
            SCAN_CONFIG['batch_size'] = max(5, min(50, int(data['batch_size'])))
        
        return jsonify({
            "success": True,
            "message": "Configuration updated",
            "config": SCAN_CONFIG
        })
    
    return jsonify({
        "success": True,
        "config": SCAN_CONFIG,
        "total_stocks": len(WATCHLIST),
        "sectors": len(SECTORS)
    })

@app.route('/api/stock-analysis/<symbol>')
@limiter.limit("30 per minute")
def stock_analysis(symbol):
    """API endpoint for single stock analysis"""
    try:
        # Validate symbol
        try:
            symbol = validate_symbol(symbol)
        except ValidationError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        
        # Validate timeframe
        timeframe = request.args.get('timeframe', '5min')
        try:
            timeframe = validate_timeframe(timeframe)
        except ValidationError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        
        logger.info(f"Analyzing stock: {symbol}, timeframe: {timeframe}")
        
        analysis = technical_analyzer.analyze_stock(symbol, timeframe)
        
        if analysis:
            # Get S/R levels
            sr_data = api_manager.fetch_support_resistance([symbol], timeframe)
            if sr_data and "data" in sr_data:
                sr_key = f"NSE_{symbol.replace('-EQ', '')}"
                analysis["support_resistance"] = sr_data["data"].get(sr_key, {})
            
            return jsonify({"success": True, "data": analysis})
        else:
            return jsonify({"success": False, "error": "Unable to analyze stock"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/candlestick/<symbol>')
def get_candlestick(symbol):
    """API endpoint for candlestick data"""
    try:
        timeframe = request.args.get('timeframe', '5min')
        candles = api_manager.fetch_candlestick_data(symbol, timeframe)
        
        if candles:
            return jsonify({"success": True, "data": candles})
        else:
            return jsonify({"success": False, "error": "No data available"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/watchlist')
def get_watchlist():
    """API endpoint to get watchlist"""
    return jsonify({"success": True, "data": WATCHLIST})

@app.route('/api/sectors')
def get_sectors():
    """API endpoint to get all sectors and their stocks"""
    return jsonify({
        "success": True, 
        "data": {
            "sectors": SECTORS,
            "industry_map": INDUSTRY_MAP,
            "total_sectors": len(SECTORS),
            "total_stocks": len(WATCHLIST)
        }
    })

@app.route('/api/sector/<sector_name>')
def get_sector_stocks(sector_name):
    """API endpoint to get stocks for a specific sector"""
    sector_upper = sector_name.upper()
    if sector_upper in INDUSTRY_MAP:
        return jsonify({
            "success": True,
            "data": {
                "sector": sector_upper,
                "stocks": INDUSTRY_MAP[sector_upper],
                "count": len(INDUSTRY_MAP[sector_upper])
            }
        })
    else:
        return jsonify({"success": False, "error": f"Sector '{sector_name}' not found"}), 404

@app.route('/api/shareholdings/<symbol>')
def get_shareholdings(symbol):
    """API endpoint to get shareholding pattern"""
    try:
        data = api_manager.fetch_shareholdings(symbol)
        if data:
            return jsonify({"success": True, "data": data})
        else:
            return jsonify({"success": False, "error": "No shareholdings data available"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/financials/<symbol>')
def get_financials(symbol):
    """API endpoint to get financial data (P&L, Balance Sheet, Cash Flow)"""
    try:
        data = api_manager.fetch_financials(symbol)
        if data:
            return jsonify({"success": True, "data": data})
        else:
            return jsonify({"success": False, "error": "No financial data available"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/technical-analysis/<symbol>')
def get_technical_analysis(symbol):
    """API endpoint to get comprehensive technical analysis from Streak"""
    try:
        timeframe = request.args.get('timeframe', '5min')
        data = api_manager.fetch_technical_analysis(symbol, timeframe)
        if data:
            return jsonify({"success": True, "data": data})
        else:
            return jsonify({"success": False, "error": "No technical analysis available"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/stock-complete/<symbol>')
def get_stock_complete(symbol):
    """API endpoint to get complete stock data (technical + financials + shareholdings)"""
    try:
        timeframe = request.args.get('timeframe', '5min')
        
        # Get all data in parallel (conceptually)
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # Technical analysis
        tech_analysis = technical_analyzer.analyze_stock(symbol, timeframe)
        if tech_analysis:
            result["technical"] = tech_analysis
        
        # Support/Resistance
        sr_data = api_manager.fetch_support_resistance([symbol], timeframe)
        if sr_data:
            sr_key = f"NSE_{symbol.replace('-EQ', '')}"
            result["support_resistance"] = sr_data.get("data", {}).get(sr_key, {})
        
        # Financials
        financials = api_manager.fetch_financials(symbol)
        if financials:
            result["financials"] = financials
        
        # Shareholdings
        shareholdings = api_manager.fetch_shareholdings(symbol)
        if shareholdings:
            result["shareholdings"] = shareholdings
        
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info('Client connected')
    emit('connection_response', {
        'status': 'connected',
        'scan_in_progress': cache['scan_in_progress'],
        'scan_status': cache['scan_status'],
        'scan_progress': cache['scan_progress']
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_update')
def handle_update_request(data):
    """Handle real-time update request"""
    update_type = data.get('type', 'all')
    
    if update_type == 'market' or update_type == 'all':
        health = market_analyzer.get_market_health()
        emit('market_update', health)
    
    if update_type == 'swing' or update_type == 'all':
        emit('swing_update', {
            "opportunities": cache["swing_opportunities"],
            "count": len(cache["swing_opportunities"])
        })
    
    if update_type == 'intraday' or update_type == 'all':
        emit('intraday_update', {
            "opportunities": cache["intraday_opportunities"],
            "count": len(cache["intraday_opportunities"])
        })
    
    if update_type == 'scan_status':
        emit('scan_status_update', {
            'scan_in_progress': cache['scan_in_progress'],
            'scan_status': cache['scan_status'],
            'scan_type': cache.get('scan_type'),
            'scan_progress': cache['scan_progress'],
            'scanned_count': cache['scanned_count'],
            'total_count': cache['total_count']
        })

@socketio.on('start_scan')
def handle_start_scan(data):
    """Handle manual scan trigger from client"""
    scan_type = data.get('type', 'swing')  # 'swing' or 'intraday'
    full_scan = data.get('full_scan', False)
    
    if cache['scan_in_progress']:
        emit('scan_error', {
            'message': 'Scan already in progress',
            'scan_type': cache.get('scan_type')
        })
        return
    
    symbols = WATCHLIST if full_scan else WATCHLIST[:SCAN_CONFIG['quick_scan_limit']]
    thread = threading.Thread(target=background_scan_worker, args=(scan_type, symbols), daemon=True)
    thread.start()
    
    emit('scan_started', {
        'scan_type': scan_type,
        'full_scan': full_scan,
        'total_stocks': len(symbols)
    })


def background_scanner():
    """Background task to scan periodically"""
    while True:
        try:
            # Scan every 5 minutes
            time.sleep(300)

            # Update market health
            health = market_analyzer.get_market_health()
            socketio.emit('market_update', health, namespace='/')

            # Scan for opportunities during market hours
            now = datetime.now()
            if 9 <= now.hour < 16:  # Market hours
                opportunities = scanner.scan_for_intraday_trades(WATCHLIST)
                socketio.emit('intraday_update', {"opportunities": opportunities}, namespace='/')
        except Exception as e:
            logger.error(f"Background scanner error: {e}", exc_info=True)

if __name__ == '__main__':
    # Start background scanner in a separate thread
    scanner_thread = threading.Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    
    # Run the app (debug=False to avoid reloader issues)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
