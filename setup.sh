#!/bin/bash
echo "Installing smartapi-python..."
pip install smartapi-python==1.4.8
echo "Downloading Spacy model..."
python -m spacy download en_core_web_sm
echo "Listing installed packages..."
pip list
