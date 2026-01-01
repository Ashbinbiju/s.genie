import streamlit as st
import sys
import importlib
from src.optimization import chips

# Force reload module
importlib.reload(chips)
print("Reloaded chips module")
