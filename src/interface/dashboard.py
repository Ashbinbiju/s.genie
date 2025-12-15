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

st.set_page_config(page_title="FPL AI Engine", layout="wide")

st.title("⚽ FPL AI Engine")

# Sidebar
st.sidebar.header("Configuration")
team_id = st.sidebar.number_input("Team ID", value=5989967, step=1)
gw = st.sidebar.number_input("Gameweek", value=17, step=1)
budget = st.sidebar.number_input("Budget (£m)", value=100.0, step=0.1)

if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching Data & Optimizing..."):
        # 1. Fetch & Switch to caching eventually
        fpl = FPLClient()
        fpl.get_bootstrap_static()
        fpl.get_fixtures()
        
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
            optimizer = TransferOptimizer(budget=max(budget, current_value))
            best_team = optimizer.recommend_transfers(df, current_ids, free_transfers=1)
            
            if best_team is not None:
                # Layout
                tab1, tab2, tab3 = st.tabs(["🚀 Optimized Squad", "🔄 Transfer Analysis", "📰 News & Risks"])
                
                # OPTIMIZED SQUAD TAB
                with tab1:
                    starters, bench = select_starting_xi(best_team)
                    
                    def format_display_df(d):
                        d = d.copy()
                        pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                        d['Pos'] = d['element_type'].map(pos_map)
                        d['Player'] = d['web_name']
                        d['Price'] = d['price']
                        d['XP'] = d['predicted_points']
                        return d[['Pos', 'Player', 'Price', 'XP']]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Starting XI")
                        st.dataframe(
                            format_display_df(starters), 
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Price": st.column_config.NumberColumn(format="£%.1f"),
                                "XP": st.column_config.NumberColumn(format="%.1f")
                            }
                        )
                        st.write(f"**Starters XP:** {starters['predicted_points'].sum():.1f}")
                        
                    with col2:
                        st.subheader("Bench")
                        st.dataframe(
                            format_display_df(bench), 
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "Price": st.column_config.NumberColumn(format="£%.1f"),
                                "XP": st.column_config.NumberColumn(format="%.1f")
                            }
                        )
                        st.write(f"**Bench XP:** {bench['predicted_points'].sum():.1f}")
                        
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
                                
                                col_out, col_arrow, col_in = st.columns([4, 1, 4])
                                with col_out:
                                    st.error(f"OUT: {t_out['web_name']}")
                                    st.caption(f"XP: {t_out['predicted_points']:.1f} | £{t_out['price']}")
                                with col_arrow:
                                    st.markdown("### ➡️")
                                with col_in:
                                    st.success(f"IN: {t_in['web_name']}")
                                    st.caption(f"XP: {t_in['predicted_points']:.1f} | £{t_in['price']}")
                                
                                st.info(f"💡 **Reason**: {t_in['web_name']} provides a **{gain:+.1f}** point play-off gain over {t_out['web_name']} next GW.")
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

            else:
                st.error("Optimization failed to find a valid team.")
        else:
            st.error("Could not fetch team picks.")
