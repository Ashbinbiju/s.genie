import os
import sys

# Add the current directory to the path so imports work
sys.path.append(os.path.dirname(__file__))

# Run the dashboard script
# This is a wrapper to allow Streamlit Cloud to auto-detect the app at root
from src.interface import dashboard
