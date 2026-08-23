import os
import sys

import streamlit as st
import requests

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.season import (shirt_url, player_photo_url, canon_position,
                              ELEMENT_TYPE_TO_POSITION)

# Known-bad photo ids: the PL server returns a placeholder rather than a 404 for these.
MANUAL_MISSING = {'714', '541065', '4470313', 'default', '219847', '4444565'}

IMAGE_CACHE_KEY = 'img_valid_cache_v4'


def get_pitch_style():
    return """
    <style>
    .pitch-row {
        display: flex;
        justify-content: space-evenly;
        gap: 10px;
        z-index: 1;
        flex: 1;
        align-items: center;
        width: 100%;
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
        height: auto;
        min-height: 850px;
        align-items: center;
        padding-bottom: 40px;
    }
    .player-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        width: 110px;
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
        height: 160px;
        justify-content: space-between;
    }
    .player-card:hover {
        transform: scale(1.05);
        z-index: 10;
        border-color: #3b82f6;
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
        background-color: #38003c;
        color: white;
        font-size: 12px;
        font-weight: bold;
        border-radius: 4px;
        padding: 2px 8px;
        margin-top: 4px;
        display: inline-block;
        width: 100%;
    }
    .pos-badge {
        display: inline-block;
        font-size: 9px;
        font-weight: 700;
        line-height: 1.4;
        letter-spacing: 0.3px;
        border-radius: 3px;
        padding: 0 4px;
        margin-right: 3px;
        color: #fff;
        vertical-align: middle;
    }
    .pos-GK  { background-color: #eab308; color: #1a1a1a; }
    .pos-DEF { background-color: #0d9488; }
    .pos-MID { background-color: #2563eb; }
    .pos-FWD { background-color: #7c3aed; }
    .bench-container {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
    }
    </style>
    """


def check_image_exists(photo_id):
    """
    Check whether a player headshot exists on the Premier League server.

    Results are memoised in session state; without that this issues one blocking HTTP
    HEAD per player per render.
    """
    if not photo_id or photo_id == 'default' or photo_id in MANUAL_MISSING:
        return False

    if IMAGE_CACHE_KEY not in st.session_state:
        st.session_state[IMAGE_CACHE_KEY] = {}
    cache = st.session_state[IMAGE_CACHE_KEY]

    if photo_id in cache:
        return cache[photo_id]

    url = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{photo_id}.png"
    try:
        response = requests.head(url, timeout=2.0)
        # Placeholder images are tiny, so require a realistic payload size too.
        is_valid = (response.status_code == 200
                    and int(response.headers.get('content-length', 0)) > 2000)
    except requests.RequestException:
        is_valid = False

    cache[photo_id] = is_valid
    return is_valid


def resolve_player_image(player):
    """
    Best available image for a player: headshot if it exists, else the club shirt,
    else the generic blank shirt.

    Shirt codes come from bootstrap_static's `teams[].code`, which resolves for every
    club including newly promoted ones. This replaces three separate hardcoded
    TEAM_SHIRT_MAP tables (two of which disagreed with each other) that went stale
    every time the league composition changed.
    """
    fallback = shirt_url(player.get('team_code', 0))

    photo_raw = str(player.get('photo', '')).replace('.jpg', '').replace('.png', '').lstrip('p')
    if photo_raw.isdigit() and check_image_exists(photo_raw):
        return player_photo_url(player.get('photo')) or fallback
    return fallback


def position_badge_html(player):
    """
    Colour-coded GK/DEF/MID/FWD pill.

    Pitch rows imply position by their vertical order, but the bench is a single flat
    row where it is otherwise unreadable — and the bench order decides who gets
    autosubbed in.
    """
    raw = player.get('position') or ELEMENT_TYPE_TO_POSITION.get(player.get('element_type'))
    pos = canon_position(raw) if raw else ''
    if pos not in ('GK', 'DEF', 'MID', 'FWD'):
        return ''
    return f'<span class="pos-badge pos-{pos}">{pos}</span>'


def get_player_card_html(player, is_new=False, is_captain=False, is_vice=False):
    img_url = resolve_player_image(player)

    badges_html = ""
    if is_new:
        badges_html += ('<div style="position:absolute;top:-5px;right:-5px;background:#28a745;'
                        'color:white;border-radius:50%;width:20px;height:20px;font-size:10px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'border:1px solid white;z-index:5;">IN</div>')
    if is_captain:
        badges_html += ('<div style="position:absolute;top:-5px;left:-5px;background:#000;'
                        'color:white;border-radius:50%;width:22px;height:22px;font-size:12px;'
                        'font-weight:bold;display:flex;align-items:center;justify-content:center;'
                        'border:1px solid white;z-index:5;">C</div>')
    elif is_vice:
        badges_html += ('<div style="position:absolute;top:-5px;left:-5px;background:#6c757d;'
                        'color:white;border-radius:50%;width:22px;height:22px;font-size:12px;'
                        'font-weight:bold;display:flex;align-items:center;justify-content:center;'
                        'border:1px solid white;z-index:5;">V</div>')

    next_opp = player.get('next_opponent', '-')
    if next_opp != '-':
        next_opp = f"vs {next_opp}"

    minutes_prob = player.get('minutes_prob', 1.0)
    points_bg = '#e02424' if minutes_prob < 0.6 else '#38003c'

    return f"""<div class="player-card" style="position: relative;">{badges_html}
<div style="display: flex; justify-content: center; margin-bottom: 4px; height: 60px; align-items: flex-end;">
<img src="{img_url}" style="width: auto; height: 60px; object-fit: contain;">
</div>
<div class="player-name" style="white-space: normal; line-height: 1.2; height: 32px; display: flex; align-items: center; justify-content: center;">{player['web_name']}</div>
<div class="player-info">
{position_badge_html(player)}{next_opp} <br/>
£{player['price']:.1f}
</div>
<div class="player-points" style="background-color: {points_bg}">
{player['predicted_points']:.1f} XP
</div>
</div>"""


def render_pitch_view(starters, bench, new_transfers=None, captain_id=None, vice_id=None):
    if new_transfers is None:
        new_transfers = []

    st.markdown(get_pitch_style(), unsafe_allow_html=True)

    def add_row(players):
        html_row = '<div class="pitch-row">'
        for _, p in players.iterrows():
            html_row += get_player_card_html(
                p,
                is_new=p['id'] in new_transfers,
                is_captain=(p['id'] == captain_id),
                is_vice=(p['id'] == vice_id),
            )
        return html_row + '</div>'

    html = '<div class="pitch-container">'
    for element_type in (1, 2, 3, 4):
        html += add_row(starters[starters['element_type'] == element_type])
    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)

    st.subheader(f"Bench (XP: {bench['predicted_points'].sum():.1f})")
    bench_html = '<div class="bench-container">'
    for _, p in bench.iterrows():
        bench_html += get_player_card_html(p, is_new=p['id'] in new_transfers)
    bench_html += '</div>'
    st.markdown(bench_html, unsafe_allow_html=True)
