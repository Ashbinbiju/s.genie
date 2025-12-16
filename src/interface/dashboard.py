import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.api.fpl import FPLClient
from src.features.processor import FeatureProcessor
from src.api.fpl import FPLClient
from src.features.processor import FeatureProcessor
from src.model.predictor import PointsPredictor
from src.optimization.solver import TransferOptimizer
from src.optimization.team_selection import select_starting_xi
from src.optimization.chips import ChipStrategy
from src.analysis.rivals import RivalSpy
from src.interface.pitch_view import render_pitch_view, check_image_exists


st.set_page_config(page_title="FPL AI Engine", layout="wide")

st.title("⚽ FPL AI Engine")

# Sidebar
st.sidebar.header("Configuration")

# Fetch League Members for Dropdown
@st.cache_data(ttl=3600)
def get_league_members(league_id):
    try:
        fpl = FPLClient()
        standings = fpl.get_league_standings(league_id)
        if standings and 'standings' in standings:
            results = standings['standings']['results']
            # Create map: "Player Name - Team Name" -> ID
            return {f"{r['player_name']} ({r['entry_name']})": r['entry'] for r in results}
    except:
        pass
    return {}

LEAGUE_ID = 1311994
members_map = get_league_members(LEAGUE_ID)

if members_map:
    # Default to specific ID if in list, else first
    default_id = 5989967
    default_index = 0
    
    # Find index of default_id
    id_list = list(members_map.values())
    if default_id in id_list:
        default_index = id_list.index(default_id)
        
    selected_name = st.sidebar.selectbox("Select Manager", list(members_map.keys()), index=default_index)
    team_id = members_map[selected_name]
else:
    # Fallback if fetch fails
    team_id = st.sidebar.number_input("Team ID", value=5989967, step=1)

gw = st.sidebar.number_input("Gameweek", value=17, step=1)
budget = st.sidebar.number_input("Budget (£m)", value=100.0, step=0.1)

if st.sidebar.button("Run Analysis"):
    st.session_state['has_run'] = True

if st.session_state.get('has_run', False):
    with st.spinner("Fetching Data & Optimizing..."):
        # 1. Fetch & Switch to caching eventually
        fpl = FPLClient()
        fpl.get_bootstrap_static()
        fpl.get_fixtures()
        
        # Fetch history for chips
        try:
            history = fpl.get_history(team_id)
        except:
            history = {}
        
        # 2. Features
        processor = FeatureProcessor()
        # Force refresh to ensure new columns (next_opponent) are generated
        df = processor.process(force_refresh=True)
        
        # 3. Predict
        predictor = PointsPredictor()
        df = predictor.predict(df)
        
        # 4. Get Current Team
        picks = fpl.get_team_picks(team_id, gw)
        if picks:
            current_ids = [p['element'] for p in picks['picks']]
            current_team_df = df[df['id'].isin(current_ids)]
            current_value = current_team_df['price'].sum()
            
            st.metric("Current Team Value", f"£{current_value:.1f}m")
            
            # 5. Optimize
            # Calculate Free Transfers automatically
            fts = fpl.calculate_free_transfers(team_id, gw)
            st.sidebar.info(f"ℹ️ Detected **{fts}** Free Transfer(s)")
            
            optimizer = TransferOptimizer(budget=max(budget, current_value))
            best_team = optimizer.recommend_transfers(df, current_ids, free_transfers=fts)
            
            if best_team is not None:
                # Layout
                # Layout
                tab1, tab2, tab3, tab4 = st.tabs(["🚀 Optimized Squad", "🔄 Transfer Analysis", "📰 News & Risks", "🏆 Rival Spy"])
                
                # ... (Tabs 1, 2, 3 content remains same, we skip to end of indentation block) ...

                # ... (Existing code for Tab 3 ends roughly around line 273 in original file, we append new tab)

                # OPTIMIZED SQUAD TAB
                with tab1:
                    # Calculate transfers for highlighting
                    new_ids = best_team['id'].tolist()
                    transfers_in_ids = best_team[~best_team['id'].isin(current_ids)]['id'].tolist()
                    
                    starters, bench = select_starting_xi(best_team)
                    
                    # Identify Captain & Vice
                    # CRITICAL: Sort by points to pick best players (ignoring position order)
                    sorted_starters = starters.sort_values('predicted_points', ascending=False)
                    captain = sorted_starters.iloc[0]
                    vice = sorted_starters.iloc[1]
                    
                    # Chip Analysis
                    chip_strat = ChipStrategy(team_id, history)
                    chip_recs = chip_strat.analyze(starters, bench, gw)
                    
                    with st.expander("💡 AI Chip Strategy Advisor", expanded=True):
                        c1, c2, c3 = st.columns(3)
                        cols = [c1, c2, c3]
                        for i, rec in enumerate(chip_recs):
                            with cols[i]:
                                st.write(f"**{rec['icon']} {rec['chip']}**")
                                if rec['recommendation'] == 'Recommended':
                                    st.success(rec['reason'])
                                elif rec['recommendation'] == 'Used':
                                    st.caption(rec['reason'])
                                else:
                                    st.info(rec['reason'])

                    # Captaincy Analysis Header
                    st.divider()
                    cap_col1, cap_col2 = st.columns([1, 3])
                    with cap_col1:
                        st.markdown("### 🧢 Captaincy")
                    with cap_col2:
                         st.info(f"**Recommendation**: **{captain['web_name']}** ({captain['predicted_points']:.1f} XP) over {vice['web_name']} ({vice['predicted_points']:.1f} XP)")

                    # Split into Pitch and Summary
                    col_pitch, col_summary = st.columns([3, 1])
                    
                    with col_pitch:
                        render_pitch_view(starters, bench, new_transfers=transfers_in_ids, captain_id=captain['id'], vice_id=vice['id'])
                        
                    with col_summary:
                        st.subheader("🔁 Changes")
                        transfers_out = current_team_df[~current_team_df['id'].isin(new_ids)]
                        transfers_in = best_team[~best_team['id'].isin(current_ids)]
                        
                        if not transfers_out.empty:
                            # Cost Analysis
                            tx_count = len(transfers_in)
                            hits = max(0, tx_count - fts)
                            cost = hits * 4
                            
                            if hits > 0:
                                st.warning(f"⚠️ **Hit Required**: -{cost} pts")
                            else:
                                st.success("✅ Free Transfer")
                                
                            for i in range(len(transfers_out)):
                                t_out = transfers_out.iloc[i]
                                st.error(f"❌ OUT: {t_out['web_name']}")
                                if i < len(transfers_in):
                                    t_in = transfers_in.iloc[i]
                                    st.success(f"✅ IN: {t_in['web_name']}")
                                st.divider()
                                
                            total_gain = best_team['predicted_points'].sum() - current_team_df['predicted_points'].sum()
                            net_gain = total_gain - cost
                            st.caption(f"📈 Projected Gain: +{total_gain:.1f} XP")
                            if hits > 0:
                                st.caption(f"📉 Net Benefit: +{net_gain:.1f} XP (after hit)")
                        else:
                            st.info("No transfers recommended.")
                        
                # TRANSFER ANALYSIS TAB
                with tab2:
                    new_ids = best_team['id'].tolist()
                    transfers_out = current_team_df[~current_team_df['id'].isin(new_ids)]
                    transfers_in = best_team[~best_team['id'].isin(current_ids)]
                    
                    st.subheader("Suggested Transfers")
                    if not transfers_out.empty:
                        for i in range(len(transfers_out)):
                            t_out = transfers_out.iloc[i]
                            # Simple pairing by position if possible, else index match
                            if i < len(transfers_in):
                                t_in = transfers_in.iloc[i] 
                                gain = t_in['predicted_points'] - t_out['predicted_points']
                                
                                col_out_img, col_out, col_arrow, col_in_img, col_in = st.columns([1, 4, 1, 1, 4])
                                
                                with col_out_img:
                                    # Shirt Map Logic (Inline for dashboard consistency)
                                    SHIRT_MAP = {
                                        1: 3, 2: 7, 3: 91, 4: 94, 5: 36, 6: 8, 7: 31, 8: 11, 9: 54, 10: 40,
                                        11: 13, 12: 14, 13: 43, 14: 1, 15: 4, 16: 17, 17: 20, 18: 6, 19: 21, 20: 39
                                    }
                                    
                                    pid = str(t_out.get('photo', 'default')).replace('.jpg', '').replace('.png', '').replace('p', '')
                                    tid = int(t_out.get('team', 0))
                                    tc = t_out.get('team_code', 0)
                                    
                                    # Use map if available, else fallback
                                    shirt_code = SHIRT_MAP.get(tid, int(tc) if tc else 0)
                                    img_src = f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{shirt_code}-110.webp" if shirt_code else "https://fantasy.premierleague.com/img/shirts/standard/shirt_0.png"
                                    
                                    if pid.isdigit() and check_image_exists(pid):
                                        img_src = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{pid}.png"
                                    st.image(img_src, width=50)
                                with col_out:
                                    st.error(f"OUT: {t_out['web_name']}")
                                    st.caption(f"XP: {t_out['predicted_points']:.1f} | £{t_out['price']}")
                                with col_arrow:
                                    st.markdown("### ➡️")
                                with col_in_img:
                                    pid = str(t_in.get('photo', 'default')).replace('.jpg', '').replace('.png', '').replace('p', '')
                                    tid = int(t_in.get('team', 0))
                                    tc = t_in.get('team_code', 0)
                                    
                                    shirt_code = SHIRT_MAP.get(tid, int(tc) if tc else 0)
                                    img_src = f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{shirt_code}-110.webp" if shirt_code else "https://fantasy.premierleague.com/img/shirts/standard/shirt_0.png"
                                    
                                    if pid.isdigit() and check_image_exists(pid):
                                        img_src = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{pid}.png"
                                    st.image(img_src, width=50)
                                with col_in:
                                    st.success(f"IN: {t_in['web_name']}")
                                    st.caption(f"XP: {t_in['predicted_points']:.1f} | £{t_in['price']}")
                                
                                # --- AI Confidence Indicator ---
                                import time
                                import os
                                try:
                                    # 1. Data Freshness
                                    file_path = "data/raw/bootstrap_static.json"
                                    if os.path.exists(file_path):
                                        mtime = os.path.getmtime(file_path)
                                        hours_old = (time.time() - mtime) / 3600
                                        freshness = max(0.5, 1.0 - (hours_old / 48)) # Decay over 48h, min 0.5
                                    else:
                                        freshness = 0.5
                                    
                                    # 2. Model Agreement (Proxy: Minutes Certainty)
                                    agreement = t_in['minutes_prob']
                                    
                                    confidence = freshness * agreement
                                    
                                    # Badge Color
                                    if confidence > 0.8: conf_color = "green"
                                    elif confidence > 0.6: conf_color = "orange"
                                    else: conf_color = "red"
                                    
                                    st.caption(f"🤖 AI Confidence: :{conf_color}[**{confidence:.2f}**]")
                                except:
                                    pass
                                # -------------------------------
                                
                                def generate_reasoning(player_in, player_out, gain):
                                    # 1. Fixture & FDR
                                    fdr_in = player_in['fixture_difficulty']
                                    fdr_out = player_out['fixture_difficulty']
                                    opp_in = player_in.get('next_opponent', '?')
                                    opp_out = player_out.get('next_opponent', '?')
                                    
                                    # 2. Minutes Buckets (Avoid 100% claims)
                                    mins_in_val = player_in['minutes_prob']
                                    if mins_in_val >= 0.95:
                                        mins_str = "Likely 90-95% starter"
                                        security_level = "High"
                                    elif mins_in_val >= 0.8:
                                        mins_str = "Standard starter"
                                        security_level = "Medium" 
                                    else:
                                        mins_str = "Rotation risk present"
                                        security_level = "Low"

                                    # 3. Position Specific Rationale
                                    # 1=GK, 2=DEF, 3=MID, 4=FWD
                                    pos_rationale = ""
                                    if player_in['element_type'] <= 2:
                                        # Defender/GK logic
                                        if fdr_in < fdr_out:
                                            pos_rationale = f"• **Defensive Upside**: Higher clean-sheet probability (Better FDR {fdr_in:.1f} vs {fdr_out:.1f})"
                                        else:
                                            pos_rationale = "• **Defensive Upside**: Solid clean-sheet potential"
                                    else:
                                        # Attacker logic
                                        pos_rationale = "• **Attacking Threat**: Higher expected goal involvement"

                                    # 4. Price Efficiency
                                    price_diff = player_out['price'] - player_in['price']
                                    if price_diff > 0:
                                        val_note = f"• **Price Efficiency**: Saves £{price_diff:.1f}m for future upgrades"
                                    else:
                                        val_note = f"• **Investment**: Uses £{abs(price_diff):.1f}m to upgrade quality"

                                    # 5. Security/Risk Comparison
                                    sec_note = ""
                                    mins_out_val = player_out['minutes_prob']
                                    if mins_in_val > mins_out_val + 0.1:
                                        sec_note = f"✅ **Security**: High — {player_in['web_name']} has stronger minutes reliability than {player_out['web_name']}."
                                    elif mins_in_val < 0.7:
                                        sec_note = f"⚠️ **Risk**: Note that {player_in['web_name']} carries some rotation risk."
                                    
                                    # Construct Output
                                    return f"""
                                    💡 **AI Rationale**:
                                    **{player_in['web_name']}** projects **+{gain:+.1f} XP** over {player_out['web_name']}, driven by:
                                    • **Minutes Security**: {mins_str}
                                    {pos_rationale}
                                    {val_note}
                                    
                                    ⚠️ **Risk**: Returns dependent on { 'clean sheets' if player_in['element_type'] <= 2 else 'attacking returns' }.
                                    
                                    {sec_note}
                                    """

                                st.markdown(generate_reasoning(t_in, t_out, gain))
                                st.divider()
                    else:
                        st.info("No transfers recommended. Holding current squad is the optimal move.")
                        
                # NEWS & RISKS TAB
                with tab3:
                    st.subheader("⚠️ Injury News & Analysis")
                    risky_players = best_team[best_team['chance_of_playing_next_round'] < 100]
                    if not risky_players.empty:
                        for _, p in risky_players.iterrows():
                            st.warning(f"**{p['web_name']}** ({p['chance_of_playing_next_round']}% chance)")
                            st.write(f"📰 News: {p['news']}")
                            st.write(f"📉 Minutes Probability Factor used: {p['minutes_prob']:.2f}")
                    else:
                        st.success("No significant injury risks in the optimized squad.")
                        
                    st.subheader("Fixture Analysis")
                    
                    # Create a nice display dataframe
                    fixture_df = best_team[['web_name', 'next_opponent', 'fixture_difficulty']].copy()
                    fixture_df = fixture_df.sort_values('fixture_difficulty')
                    
                    # Visual mapping
                    def get_diff_icon(d):
                        if d <= 2.8: return "🟩 Good"
                        if d >= 3.5: return "🟥 Tough"
                        return "🟨 Avg"
                        
                    fixture_df['Rating'] = fixture_df['fixture_difficulty'].apply(get_diff_icon)
                    fixture_df = fixture_df.rename(columns={'web_name': 'Player', 'next_opponent': 'Next Match'})
                    
                    st.dataframe(
                        fixture_df[['Player', 'Next Match', 'Rating', 'fixture_difficulty']], 
                        column_config={
                            "fixture_difficulty": st.column_config.NumberColumn("Diff (1-5)", format="%.1f"),
                        },
                        hide_index=True,
                        width="stretch"
                    )

                # RIVAL SPY TAB
                with tab4:
                    st.subheader("🕵️‍♂️ Rival Scout")
                    league_id = st.text_input("League ID", value="1311994")
                    
                    if st.button("Fetch Standings"):
                        with st.spinner("Spying on league..."):
                            standings = fpl.get_league_standings(league_id)
                            if standings:
                                # Save to session state to persist dropdown
                                st.session_state['standings'] = standings
                            else:
                                st.error("Could not fetch standings.")
                    
                    if 'standings' in st.session_state:
                        standings = st.session_state['standings']
                        # Create options dict: "Name - Team" -> ID
                        results = standings['standings']['results']
                        rival_map = {f"{r['player_name']} - {r['entry_name']}": r['entry'] for r in results if r['entry'] != team_id}
                        
                        target_name = st.selectbox("Select Target", list(rival_map.keys()))
                        target_id = rival_map[target_name]
                        
                        if st.button("Analyze Head-to-Head"):
                            with st.spinner(f"Comparing vs {target_name}..."):
                                # Fetch Rival Picks
                                rival_picks = fpl.get_team_picks(target_id, gw)
                                if rival_picks:
                                    r_ids = [p['element'] for p in rival_picks['picks']]
                                    rival_df = df[df['id'].isin(r_ids)]
                                    
                                    # Run Spy
                                    spy = RivalSpy(current_team_df, rival_df)
                                    analysis = spy.compare()
                                    
                                    # Display
                                    st.divider()
                                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                                    metric_col1.metric("Common Players", f"{analysis['common_count']}")
                                    metric_col2.metric("Differentials", f"{analysis['differential_count']}")
                                    
                                    swing = analysis['net_swing']
                                    metric_col3.metric("Projected Swing", f"{swing:+.1f} XP", delta=f"{swing:.1f}")
                                    st.caption(f"Horizon: GW{gw} only | No captaincy applied")
                                    
                                    # Insights
                                    if swing < -5:
                                        st.info(f"""
                                        **Why you're behind:**
                                        • They own **{analysis['rival_heavy_hitters']}** high-XP players (>5.5 XP).
                                        • You have **{analysis['my_zeros']}** players with ~0.0 XP (Bench/Injured).
                                        • Main gap is in **{analysis['main_gap_pos']}**.
                                        """)
                                    
                                    st.subheader("⚡ Differential Battle")
                                    c1, c2 = st.columns(2)
                                    
                                    def format_player(p):
                                        xp = p['predicted_points']
                                        name = p['web_name']
                                        if xp >= 6.0: return f":red[**{name}**] ({xp:.1f} XP)"
                                        if xp >= 5.0: return f":orange[**{name}**] ({xp:.1f} XP)"
                                        if xp < 0.5: return f"{name} (⚠️ {xp:.1f} XP)"
                                        return f"**{name}** ({xp:.1f} XP)"
                                    
                                    with c1:
                                        st.caption("🛡️ You Have (Unique)")
                                        for _, p in analysis['my_diffs'].iterrows():
                                            ic, nc = st.columns([1, 4])
                                            with ic:
                                                pid = str(p.get('photo', 'default')).replace('.jpg', '').replace('.png', '').replace('p', '')
                                                tid = int(p.get('team', 0))
                                                tc = p.get('team_code', 0)
                                                
                                                # Use map if available, else fallback
                                                shirt_code = SHIRT_MAP.get(tid, int(tc) if tc else 0)
                                                img_src = f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{shirt_code}-110.webp" if shirt_code else "https://fantasy.premierleague.com/img/shirts/standard/shirt_0.png"
                                                
                                                if pid.isdigit() and check_image_exists(pid):
                                                    img_src = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{pid}.png"
                                                st.image(img_src, width=40)
                                            with nc:
                                                st.markdown(format_player(p))
                                            
                                    with c2:
                                        st.caption("⚔️ They Have (Unique)")
                                        for _, p in analysis['rival_diffs'].iterrows():
                                            ic, nc = st.columns([1, 4])
                                            with ic:
                                                pid = str(p.get('photo', 'default')).replace('.jpg', '').replace('.png', '').replace('p', '')
                                                tid = int(p.get('team', 0))
                                                tc = p.get('team_code', 0)
                                                
                                                shirt_code = SHIRT_MAP.get(tid, int(tc) if tc else 0)
                                                img_src = f"https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{shirt_code}-110.webp" if shirt_code else "https://fantasy.premierleague.com/img/shirts/standard/shirt_0.png"
                                                
                                                if pid.isdigit() and check_image_exists(pid):
                                                    img_src = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{pid}.png"
                                                st.image(img_src, width=40)
                                            with nc:
                                                st.markdown(format_player(p))
                                            
                                    if analysis['danger_player'] is not None:
                                        dp = analysis['danger_player']
                                        dp_xp = dp['predicted_points']
                                        
                                        if dp_xp >= 6.0:
                                            st.warning(f"⚠️ **Major Threat**: {dp['web_name']} is their biggest differential ({dp_xp:.1f} XP).")
                                        else:
                                            st.info(f"ℹ️ **Top Scout Target**: {dp['web_name']} is their highest unique ({dp_xp:.1f} XP).")
                                    
                                else:
                                    st.error("Could not fetch rival team.")

            else:
                st.error("Optimization failed to find a valid team.")
        else:
            st.error("Could not fetch team picks.")
