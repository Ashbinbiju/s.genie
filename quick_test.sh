#!/bin/bash

echo "🧪 Running API Tests..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install required packages if needed
pip install -q requests python-dotenv

# Run the test script
python test_apis.py

# Capture exit code
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ All tests passed!"
    echo ""
    read -p "Do you want to start the application now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./run.sh
    fi
else
    echo "❌ Tests failed. Please fix the issues before running the app."
fi

exit $exit_code
