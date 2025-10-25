#!/bin/bash

cd /workspaces/s.genie

echo "🧪 StockGenie Pro - API Test Suite"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Creating .env file with your credentials..."
    
    cat > .env << 'EOF'
# SmartAPI (AngelOne) Credentials
CLIENT_ID=AAAG399109
PASSWORD=1503
TOTP_SECRET=OLRQ3CYBLPN2XWQPHLKMB7WEKI

# API Keys
HISTORICAL_API_KEY=c3C0tMGn
TRADING_API_KEY=ruseeaBq
MARKET_API_KEY=PflRFXyd

# Telegram Alerts
TELEGRAM_ENABLED=true
TELEGRAM_CHAT_ID=-1002411670969
TELEGRAM_BOT_TOKEN=7902319450:AAFPNcUyk9F6Sesy-h6SQnKHC_Yr6Uqk9ps

# Supabase (Paper Trading) - Optional
SUPABASE_URL=
SUPABASE_KEY=

# Flask Configuration
FLASK_SECRET_KEY=stockgenie-pro-secret-key-2025
FLASK_DEBUG=true
EOF
    echo "✅ .env file created"
    echo ""
fi

# Install dependencies quietly
echo "📦 Installing dependencies..."
pip install -q requests python-dotenv pandas numpy 2>/dev/null

echo ""
echo "🚀 Running API tests..."
echo ""

# Run the test
python3 test_apis.py

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✅ All tests passed! Ready to launch application."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    read -p "Start the web application now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🚀 Starting StockGenie Pro..."
        chmod +x run.sh
        ./run.sh
    else
        echo ""
        echo "To start later, run: ./run.sh"
    fi
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ❌ Some tests failed. Check errors above."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Common issues:"
    echo "  • Check internet connection"
    echo "  • Verify API endpoints are accessible"
    echo "  • Market might be closed (some APIs need live data)"
    echo ""
fi

exit $exit_code
