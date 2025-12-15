import streamlit as st
import pandas as pd
import requests

# Cache for image validation checks to avoid repeated network calls
# Key: photo_id, Value: bool (True if valid, False if 404)
IMAGE_CACHE = {}

def get_pitch_style():
    return """
    <style>
    .pitch-row {
        display: flex;
        justify-content: space-evenly;
        gap: 10px;
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
        min-height: 850px; /* Allow expansion for taller cards */
        align-items: center; /* Center rows horizontally */
    }
    /* Rest of CSS remains similar but minified/cleaned */
    .player-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        width: 110px; /* Wider to fit details */
        padding: 6px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
        transition: transform 0.2s;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2px;
        height: 160px; /* Increased height for name wrapping */
        justify-content: space-between;
    }
    
    .player-card:hover {
        transform: scale(1.05);
        z-index: 10;
        border-color: #3b82f6; 
    }

    .player-shirt {
        font-size: 28px;
        margin-bottom: 2px;
        line-height: 1;
    }
    
    .player-name {
        font-weight: bold;
        font-size: 13px;
        color: #111;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
    }

    .player-info {
        font-size: 11px;
        color: #555;
        white-space: nowrap;
    }
    
    .player-points {
        background-color: #38003c; /* FPL Purpleish */
        color: white;
        font-size: 12px;
        font-weight: bold;
        border-radius: 4px;
        padding: 2px 8px;
        margin-top: 4px;
        display: inline-block;
        width: 100%;
    }
    
    .bench-container {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap; /* Allow wrapping if bench is full */
    }
    </style>
    """

# Known missing photos or special overrides
# 714: Woltemade (Confirmed missing/broken)
MANUAL_MISSING = {'714', '541065', 'default'}

def check_image_exists(photo_id):
    """
    Checks if a player photo exists on the Premier League server.
    Uses caching to minimize latency.
    """
    # 1. Manual Override Checks FIRST (bypasses cache)
    if photo_id in MANUAL_MISSING:
        # Update cache to ensure consistency but return False immediately
        if 'image_validity_cache' in st.session_state:
            st.session_state.image_validity_cache[photo_id] = False
        return False
        
    # Initialize cache in session state if needed
    if 'image_validity_cache' not in st.session_state:
        st.session_state.image_validity_cache = {}
        
    if photo_id in st.session_state.image_validity_cache:
        return st.session_state.image_validity_cache[photo_id]
    
    url = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{photo_id}.png"
    try:
        response = requests.head(url, timeout=1.0)
        # Check status and ensure content isn't empty
        is_valid = response.status_code == 200 and int(response.headers.get('content-length', 0)) > 1000
    except:
        is_valid = False
        
    st.session_state.image_validity_cache[photo_id] = is_valid
    return is_valid

def get_player_card_html(player, is_new=False):
    p_type = player['element_type']
    
    # 1. Normalize ID
    photo_raw = str(player.get('photo', 'default')).replace('.jpg', '').replace('.png', '').replace('p', '')
    
    # 2. Determine Image URL via Server-Side Check
    # Default to shirt
    img_url = "https://fantasy.premierleague.com/img/shirts/standard/shirt_0.png"
    
    if photo_raw.isdigit():
        # Only use the real photo if we confirm it exists (Status 200)
        if check_image_exists(photo_raw):
            img_url = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{photo_raw}.png"
    
    badge_html = ""
    if is_new:
        badge_html = '<div style="position: absolute; top: -5px; right: -5px; background: #28a745; color: white; border-radius: 50%; width: 20px; height: 20px; font-size: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid white; z-index: 5;">IN</div>'
    
    next_opp = player.get('next_opponent', '-')
    if next_opp != '-':
        next_opp = f"vs {next_opp}"

    # 3. Clean HTML (No onerror needed since we validated the URL)
    return f"""<div class="player-card" style="position: relative;">{badge_html}
<div style="display: flex; justify-content: center; margin-bottom: 4px; height: 60px; align-items: flex-end;">
<img src="{img_url}" style="width: auto; height: 60px; object-fit: contain;">
</div>
<div class="player-name" style="white-space: normal; line-height: 1.2; height: 32px; display: flex; align-items: center; justify-content: center;">{player['web_name']}</div>
<div class="player-info">
{next_opp} <br/>
£{player['price']:.1f}
</div>
<div class="player-points" style="background-color: {'#e02424' if player['minutes_prob'] < 0.6 else '#38003c'}">
{player['predicted_points']:.1f} XP
</div>
</div>"""

def render_pitch_view(starters, bench, new_transfers=None):
    if new_transfers is None: new_transfers = []
    
    # CSS
    st.markdown(get_pitch_style(), unsafe_allow_html=True)
    
    # Rows
    gks = starters[starters['element_type'] == 1]
    defs = starters[starters['element_type'] == 2]
    mids = starters[starters['element_type'] == 3]
    fwds = starters[starters['element_type'] == 4]
    
    # Helper to clean up loop
    def add_row(players):
        html_row = '<div class="pitch-row">'
        for _, p in players.iterrows():
            is_new = p['id'] in new_transfers
            html_row += get_player_card_html(p, is_new)
        html_row += '</div>'
        return html_row

    # Build Pitch HTML
    html = '<div class="pitch-container">'
    html += '<div class="pitch-line"></div>'
    html += '<div class="pitch-circle"></div>'
    
    # Rows
    html += add_row(gks)
    html += add_row(defs)
    html += add_row(mids)
    html += add_row(fwds)
    
    html += '</div>' # End pitch container
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Bench
    st.subheader(f"Bench (XP: {bench['predicted_points'].sum():.1f})")
    
    bench_html = '<div class="bench-container">'
    for _, p in bench.iterrows():
        is_new = p['id'] in new_transfers
        bench_html += get_player_card_html(p, is_new)
    bench_html += '</div>'
    
    st.markdown(bench_html, unsafe_allow_html=True)

