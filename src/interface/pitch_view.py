import streamlit as st
import pandas as pd

def get_pitch_style():
    return """
    <style>
    .pitch-row {
        display: flex;
        justify-content: center;
        gap: 20px;
        z-index: 1; /* Above lines */
        flex: 1;
        align-items: center;
        width: 100%; /* Ensure it spans full width */
    }
    .pitch-container {
        position: relative;
        background: linear-gradient(180deg, #1e7e34 0%, #28a745 50%, #1e7e34 100%);
        border: 2px solid white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        height: 600px;
        align-items: center; /* Center rows horizontally */
    }
    /* Rest of CSS remains similar but minified/cleaned */
    .player-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 6px;
        width: 90px; /* Slightly smaller to fit 5 defenders */
        padding: 4px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
        transition: transform 0.2s;
        cursor: pointer;
    }
    
    .player-card:hover {
        transform: scale(1.05);
        z-index: 10;
        border-color: #3b82f6; 
    }

    .player-shirt {
        font-size: 24px;
        margin-bottom: 4px;
    }
    
    .player-name {
        font-weight: bold;
        font-size: 13px;
        color: #111;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .player-points {
        background-color: #38003c; /* FPL Purpleish */
        color: white;
        font-size: 12px;
        font-weight: bold;
        border-radius: 4px;
        padding: 2px 6px;
        margin-top: 4px;
        display: inline-block;
    }
    
    .bench-container {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        display: flex;
        justify-content: center;
        gap: 15px;
    }
    </style>
    """

def get_player_card_html(player):
    # Determine Shirt Icon based on position? Or just generic
    # 1=GK, 2=DEF, 3=MID, 4=FWD
    pos_map = {1: '🧤', 2: '🛡️', 3: '⚙️', 4: '⚡'}
    icon = pos_map.get(player['element_type'], '👕')
    
    xp = f"XP: {player['predicted_points']:.1f}"
    
    return f"""<div class="player-card">
    <div class="player-shirt">{icon}</div>
    <div class="player-name">{player['web_name']}</div>
    <div class="player-points">{xp}</div>
</div>"""

def render_pitch_view(starters, bench):
    # CSS
    st.markdown(get_pitch_style(), unsafe_allow_html=True)
    
    # Rows
    gks = starters[starters['element_type'] == 1]
    defs = starters[starters['element_type'] == 2]
    mids = starters[starters['element_type'] == 3]
    fwds = starters[starters['element_type'] == 4]
    
    # Build Pitch HTML
    html = '<div class="pitch-container">'
    html += '<div class="pitch-line"></div>'
    html += '<div class="pitch-circle"></div>'
    
    # GK Row
    html += '<div class="pitch-row">'
    for _, p in gks.iterrows():
        html += get_player_card_html(p)
    html += '</div>'
    
    # DEF Row
    html += '<div class="pitch-row">'
    for _, p in defs.iterrows():
        html += get_player_card_html(p)
    html += '</div>'
    
    # MID Row
    html += '<div class="pitch-row">'
    for _, p in mids.iterrows():
        html += get_player_card_html(p)
    html += '</div>'
    
    # FWD Row
    html += '<div class="pitch-row">'
    for _, p in fwds.iterrows():
        html += get_player_card_html(p)
    html += '</div>'
    
    html += '</div>' # End pitch container
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Bench
    st.subheader(f"Bench (XP: {bench['predicted_points'].sum():.1f})")
    
    bench_html = '<div class="bench-container">'
    for _, p in bench.iterrows():
        bench_html += get_player_card_html(p)
    bench_html += '</div>'
    
    st.markdown(bench_html, unsafe_allow_html=True)

