#!/bin/bash

echo "🚀 Starting StockGenie Pro Web UI..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env with your credentials"
    exit 1
fi

# Run the Flask app
echo ""
echo "✅ Starting web server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📱 Web UI: http://localhost:5000"
echo "🖥️  Local: http://0.0.0.0:5000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔑 Credentials loaded:"
echo "   Client ID: AAAG399109"
echo "   Telegram: Enabled"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Open browser automatically
sleep 2 && "$BROWSER" http://localhost:5000 &

python app.py
