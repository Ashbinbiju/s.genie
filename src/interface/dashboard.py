import os
import sys
import time

import streamlit as st
import pandas as pd

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.api.fpl import FPLClient
from src.api.async_fpl import refresh_cache
from src.features.processor import FeatureProcessor
from src.model.predictor import PointsPredictor
from src.optimization.solver import TransferOptimizer
from src.optimization.team_selection import select_starting_xi, squad_expected_points, pick_captain
from src.optimization.chips import ChipStrategy
from src.analysis.rivals import RivalSpy
from src.interface.pitch_view import render_pitch_view, resolve_player_image
from src.utils.season import load_bootstrap, get_season_label, get_next_gw, is_preseason

st.set_page_config(page_title="FPL AI Engine", layout="wide")
st.title("⚽ FPL AI Engine v2.3")

# 2026/27 season league. Leagues are per-season — the previous id (1311994) 404s.
DEFAULT_LEAGUE_ID = 1019782
CACHE_TTL = 900  # 15 minutes

# NOTE: there is deliberately no hardcoded default TEAM id. Entry ids are issued fresh
# every season, so any pinned value goes stale each August and silently 404s. The
# default is taken from the league's own membership instead.


# ---------------------------------------------------------------------------
# Cached pipeline stages
#
# Streamlit re-executes this script top-to-bottom on EVERY widget interaction. Without
# caching, typing in the Rival Spy league box re-ran two API fetches, a full feature
# rebuild, a model inference pass and two CBC integer-program solves.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_static():
    fpl = FPLClient()
    static = fpl.get_bootstrap_static()
    fpl.get_fixtures()
    return static or load_bootstrap()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_league_members(league_id):
    """{'Manager (Team)': entry_id}. Empty dict when the league is missing or empty."""
    return FPLClient().get_league_members(league_id)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def describe_entry(team_id):
    """('Manager Name', 'Team Name') for a team id, or None if it does not exist."""
    entry = FPLClient().get_entry(team_id)
    if not entry:
        return None
    name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
    return name, entry.get('name', '')


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_predictions():
    """Features + model predictions. Returns (df, mode, warnings, odds_confidence)."""
    fetch_static()
    try:
        refresh_cache()
    except Exception as e:
        st.warning(f"Could not refresh the player-summary cache: {e}")

    df = FeatureProcessor().process(force_refresh=True)
    if df is None:
        return None, "error", ["Feature processing returned no data."], "NONE"

    predictor = PointsPredictor()
    df = predictor.predict(df)
    return df, predictor.prediction_mode, predictor.prediction_warnings, predictor.odds_confidence


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_squad_context(team_id, gw):
    """History, Free Hit GWs, current picks and free-transfer count for a manager."""
    fpl = FPLClient()
    history = fpl.get_history(team_id) or {}
    freehit_gws = fpl.get_freehit_gws(team_id, history)
    picks = fpl.get_team_picks(team_id, gw, freehit_gws=freehit_gws)
    fts = fpl.calculate_free_transfers(team_id, gw, history)
    return history, sorted(freehit_gws), picks, fts


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_optimal_squad(budget):
    """Best possible 15 from scratch — the Wildcard / Free Hit / pre-season squad."""
    df, _, _, _ = get_predictions()
    if df is None:
        return None
    return TransferOptimizer(budget=budget).solve_team(df)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def build_transfer_plan(current_ids, budget, free_transfers):
    df, _, _, _ = get_predictions()
    if df is None:
        return None
    return TransferOptimizer(budget=budget).recommend_transfers(
        df, list(current_ids), free_transfers=free_transfers)


def player_image(row, width=50):
    st.image(resolve_player_image(row), width=width)


def build_rationale(player_in, player_out, gain):
    """Plain-language justification for a single transfer."""
    fdr_in = player_in['fixture_difficulty']
    fdr_out = player_out['fixture_difficulty']

    mins_in = player_in.get('minutes_prob', 1.0)
    if mins_in >= 0.95:
        mins_str = "Likely 90-95% starter"
    elif mins_in >= 0.8:
        mins_str = "Standard starter"
    else:
        mins_str = "Rotation risk present"

    if player_in['element_type'] <= 2:
        if fdr_in < fdr_out:
            pos_rationale = (f"• **Defensive upside**: better fixture run "
                             f"(FDR {fdr_in:.1f} vs {fdr_out:.1f}), so a higher "
                             f"clean-sheet chance")
        else:
            pos_rationale = "• **Defensive upside**: solid clean-sheet potential"
        risk_driver = "clean sheets"
    else:
        pos_rationale = "• **Attacking threat**: higher expected goal involvement"
        risk_driver = "attacking returns"

    price_diff = player_out['price'] - player_in['price']
    if price_diff > 0:
        val_note = f"• **Price efficiency**: frees £{price_diff:.1f}m for future upgrades"
    else:
        val_note = f"• **Investment**: spends £{abs(price_diff):.1f}m to upgrade quality"

    mins_out = player_out.get('minutes_prob', 1.0)
    if mins_in > mins_out + 0.1:
        sec_note = (f"✅ **Security**: {player_in['web_name']} has stronger minutes "
                    f"reliability than {player_out['web_name']}.")
    elif mins_in < 0.7:
        sec_note = f"⚠️ **Risk**: {player_in['web_name']} carries rotation risk."
    else:
        sec_note = ""

    return f"""
💡 **AI Rationale**

**{player_in['web_name']}** projects **{gain:+.1f} XP** over {player_out['web_name']}, driven by:

• **Minutes security**: {mins_str}
{pos_rationale}
{val_note}

⚠️ **Risk**: returns depend on {risk_driver}.

{sec_note}
"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

static = fetch_static()
season_label = get_season_label(static)
preseason = is_preseason(static)
next_gw = get_next_gw(static)

st.sidebar.caption(f"Season **{season_label}** · next deadline **GW{next_gw}**")

league_id = st.sidebar.number_input("League ID", value=DEFAULT_LEAGUE_ID, step=1)
members_map = get_league_members(int(league_id))

if members_map:
    selected_name = st.sidebar.selectbox(
        "Select Manager", list(members_map.keys()),
        help="Read from the league. Before GW1 is scored, members appear as recent "
             "joiners rather than in the ranked standings.")
    team_id = members_map[selected_name]
    if preseason:
        st.sidebar.caption(f"{len(members_map)} manager(s) in this league "
                           f"(standings start after GW{next_gw}).")
else:
    st.sidebar.caption(
        "No members found for this league — check the ID, or enter a team ID directly.")
    team_id = st.sidebar.number_input(
        "Team ID", min_value=1, step=1,
        help="Find it in the URL when viewing your team on the FPL site: "
             "/entry/<TEAM ID>/event/1. Team IDs are issued fresh each season, so last "
             "season's ID will not work.")

# Validate whatever id we ended up with. A stale id from a previous season returns 404
# on every subsequent call, which previously surfaced only as console noise.
entry_info = describe_entry(int(team_id))
if entry_info:
    manager, entry_team = entry_info
    st.sidebar.caption(f"✅ **{entry_team}** — {manager}")
else:
    st.sidebar.error(
        f"Team ID {int(team_id)} does not exist in {season_label}. FPL issues a new "
        f"team ID every season, so an ID from a previous season will not work."
    )
    st.stop()

gw = st.sidebar.number_input("Gameweek", value=next_gw, min_value=1, max_value=38, step=1)
budget_override = st.sidebar.number_input("Budget (£m)", value=100.0, step=0.1)
bank = st.sidebar.number_input(
    "Bank (£m)", value=0.0, step=0.1, min_value=0.0,
    help="Money not tied up in players. Spending power is squad value + bank; without "
         "this the optimizer systematically under-budgets.")

if st.sidebar.button("Run Analysis"):
    st.session_state['has_run'] = True

if not st.session_state.get('has_run', False):
    st.info("Configure the sidebar and press **Run Analysis**.")
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
with st.spinner("Fetching data & optimizing..."):
    df, prediction_mode, prediction_warnings, odds_confidence = get_predictions()

if df is None:
    st.error("Feature processing failed — no data to work with.")
    st.stop()

# Surface model health rather than letting a degraded run look identical to a good one.
if prediction_mode == "fallback":
    st.error(
        "**Predictions are running on the emergency heuristic, not the ML model.** "
        "They are significantly less accurate. See the warnings below."
    )
for warning in prediction_warnings:
    st.warning(warning)
if odds_confidence == "LOW":
    st.sidebar.caption("⚠️ Odds confidence: LOW (league-average defaults)")
elif odds_confidence == "HIGH":
    st.sidebar.caption("✅ Odds confidence: HIGH (live bookmaker odds)")

history, freehit_gws, picks, fts = get_squad_context(int(team_id), int(gw))

if freehit_gws:
    st.sidebar.info(f"🔁 Free Hit played in GW(s) {freehit_gws} — permanent squad loaded "
                    f"from a different GW.")

# ---------------------------------------------------------------------------
# Pre-season / no squad yet: recommend a squad from scratch instead of dead-ending.
# ---------------------------------------------------------------------------
if not picks:
    if preseason or gw <= 1:
        st.info(f"**{season_label} has not started yet** — no squad exists to improve on. "
                f"Showing the optimal squad to draft for GW{gw}.")
    else:
        st.warning("Could not load your squad from the FPL API. Showing the optimal squad "
                   "from scratch instead.")

    draft_budget = float(budget_override)
    squad = build_optimal_squad(draft_budget)
    if squad is None:
        st.error("Optimization failed to find a valid squad.")
        st.stop()

    starters, bench = select_starting_xi(squad)
    captain, vice = pick_captain(starters)

    st.metric("Squad Cost", f"£{squad['price'].sum():.1f}m",
              help=f"Budget £{draft_budget:.1f}m")
    if captain is not None:
        st.info(f"**Captain**: {captain['web_name']} "
                f"({captain['predicted_points']:.1f} XP) · "
                f"**Vice**: {vice['web_name'] if vice is not None else '-'}")

    render_pitch_view(starters, bench,
                      captain_id=captain['id'] if captain is not None else None,
                      vice_id=vice['id'] if vice is not None else None)
    st.stop()

# ---------------------------------------------------------------------------
# Normal path: we have a squad
# ---------------------------------------------------------------------------
current_ids = [p['element'] for p in picks['picks']]
current_team_df = df[df['id'].isin(current_ids)]

missing = len(current_ids) - len(current_team_df)
if missing:
    st.warning(f"{missing} of your {len(current_ids)} players are not in the current player "
               f"list (transferred out of the league?). They are excluded from the analysis.")

current_value = float(current_team_df['price'].sum())
spending_power = max(float(budget_override), current_value + float(bank))

current_starters, current_bench = select_starting_xi(current_team_df)
current_captain, _ = pick_captain(current_starters)
current_xi_xp = squad_expected_points(
    current_starters, current_captain['id'] if current_captain is not None else None)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Current Team Value", f"£{current_value:.1f}m")
col_b.metric("Spending Power", f"£{spending_power:.1f}m", help="Squad value + bank")
col_c.metric("Current XI (capt. doubled)", f"{current_xi_xp:.1f} XP")

st.sidebar.info(f"ℹ️ Detected **{fts}** Free Transfer(s)")

best_team = build_transfer_plan(tuple(current_ids), spending_power, int(fts))
if best_team is None:
    st.error("Optimization failed to find a valid team.")
    st.stop()

new_ids = best_team['id'].tolist()
transfers_in_ids = best_team[~best_team['id'].isin(current_ids)]['id'].tolist()
starters, bench = select_starting_xi(best_team)
captain, vice = pick_captain(starters)
best_xi_xp = squad_expected_points(starters, captain['id'] if captain is not None else None)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🚀 Optimized Squad", "🔄 Transfer Analysis", "📰 News & Risks", "🏆 Rival Spy"])

# ---------------------------------------------------------------------------
# TAB 1 — Optimized squad
# ---------------------------------------------------------------------------
with tab1:
    # Chip analysis is evaluated against the squad you ACTUALLY OWN, not the
    # post-transfer squad — advice about a bench you don't have is useless.
    active_count = int((current_team_df['predicted_points'] > 0.5).sum())

    wc_squad = build_optimal_squad(spending_power)
    wc_diff = 0.0
    wc_xi_xp = 0.0
    if wc_squad is not None:
        wc_starters, wc_bench = select_starting_xi(wc_squad)
        wc_cap, wc_vice = pick_captain(wc_starters)
        wc_xi_xp = squad_expected_points(wc_starters, wc_cap['id'] if wc_cap is not None else None)
        wc_diff = wc_xi_xp - current_xi_xp

    chip_recs = ChipStrategy(team_id, history).analyze(
        current_starters, current_bench, gw,
        wildcard_diff=wc_diff,
        freehit_diff=wc_diff,  # a Free Hit is a one-week Wildcard
        active_players=active_count,
        current_xi_xp=current_xi_xp,
    )

    with st.expander(f"💡 AI Chip Strategy Advisor (GW {gw})", expanded=True):
        st.caption("Evaluated against your current squad. Gains are XI-level "
                   "(captain doubled), since bench points are not scored.")
        cols = st.columns(len(chip_recs))
        for i, rec in enumerate(chip_recs):
            with cols[i]:
                st.write(f"**{rec['icon']} {rec['chip']}**")
                if rec['recommendation'] == 'Recommended':
                    st.success(rec['reason'])
                elif rec['recommendation'] == 'Consider':
                    st.warning(rec['reason'])
                elif rec['recommendation'] in ('Available', 'Save'):
                    st.info(rec['reason'])
                else:
                    st.error(rec['reason'])

        if wc_squad is not None:
            with st.expander("👀 View AI's Ideal Wildcard/Free Hit Squad"):
                st.caption(f"Projected XI: {wc_xi_xp:.1f} XP (vs current {current_xi_xp:.1f} XP)")
                render_pitch_view(
                    wc_starters, wc_bench,
                    captain_id=wc_cap['id'] if wc_cap is not None else None,
                    vice_id=wc_vice['id'] if wc_vice is not None else None)

    st.divider()
    cap_col1, cap_col2 = st.columns([1, 3])
    with cap_col1:
        st.markdown("### 🧢 Captaincy")
    with cap_col2:
        if captain is not None and vice is not None:
            st.info(f"**Recommendation**: **{captain['web_name']}** "
                    f"({captain['predicted_points']:.1f} XP) over {vice['web_name']} "
                    f"({vice['predicted_points']:.1f} XP)")
            st.caption("Ranked by captaincy score, which discounts rotation risk and a "
                       "hard fixture — not by raw expected points.")

    col_pitch, col_summary = st.columns([3, 1])
    with col_pitch:
        render_pitch_view(starters, bench, new_transfers=transfers_in_ids,
                          captain_id=captain['id'] if captain is not None else None,
                          vice_id=vice['id'] if vice is not None else None)

    with col_summary:
        st.subheader("🔁 Changes")
        transfers_out = current_team_df[~current_team_df['id'].isin(new_ids)]
        transfers_in = best_team[~best_team['id'].isin(current_ids)]

        if transfers_out.empty:
            st.info("No transfers recommended.")
        else:
            hits = max(0, len(transfers_in) - fts)
            cost = hits * 4
            if hits > 0:
                st.warning(f"⚠️ **Hit Required**: -{cost} pts")
            else:
                st.success("✅ Free Transfer")

            for i in range(len(transfers_out)):
                st.error(f"❌ OUT: {transfers_out.iloc[i]['web_name']}")
                if i < len(transfers_in):
                    st.success(f"✅ IN: {transfers_in.iloc[i]['web_name']}")
                st.divider()

            gain = best_xi_xp - current_xi_xp
            st.caption(f"📈 Projected XI gain: {gain:+.1f} XP")
            if hits > 0:
                st.caption(f"📉 Net after hit: {gain - cost:+.1f} XP")

# ---------------------------------------------------------------------------
# TAB 2 — Transfer analysis
# ---------------------------------------------------------------------------
with tab2:
    transfers_out = current_team_df[~current_team_df['id'].isin(new_ids)]
    transfers_in = best_team[~best_team['id'].isin(current_ids)]

    st.subheader("Suggested Transfers")
    if transfers_out.empty:
        st.info("No transfers recommended. Holding the current squad is the optimal move.")
    else:
        # Data freshness decays over 48h from the last bootstrap fetch, floored at 0.5.
        try:
            mtime = os.path.getmtime("data/raw/bootstrap_static.json")
            freshness = max(0.5, 1.0 - ((time.time() - mtime) / 3600) / 48)
        except OSError:
            freshness = 0.5

        for i in range(len(transfers_out)):
            t_out = transfers_out.iloc[i]
            if i >= len(transfers_in):
                break
            t_in = transfers_in.iloc[i]
            gain = t_in['predicted_points'] - t_out['predicted_points']

            c_out_img, c_out, c_arrow, c_in_img, c_in = st.columns([1, 4, 1, 1, 4])
            with c_out_img:
                player_image(t_out)
            with c_out:
                st.error(f"OUT: {t_out['web_name']}")
                st.caption(f"XP: {t_out['predicted_points']:.1f} | £{t_out['price']:.1f}m")
            with c_arrow:
                st.markdown("### ➡️")
            with c_in_img:
                player_image(t_in)
            with c_in:
                st.success(f"IN: {t_in['web_name']}")
                st.caption(f"XP: {t_in['predicted_points']:.1f} | £{t_in['price']:.1f}m")

            confidence = freshness * float(t_in.get('minutes_prob', 1.0))
            conf_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
            st.caption(f"🤖 AI Confidence: :{conf_color}[**{confidence:.2f}**]")

            st.markdown(build_rationale(t_in, t_out, gain))
            st.divider()

# ---------------------------------------------------------------------------
# TAB 3 — News & risks
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("⚠️ Injury News & Analysis")
    risky = best_team[best_team['chance_of_playing_next_round'] < 100]
    if risky.empty:
        st.success("No significant injury risks in the optimized squad.")
    else:
        for _, p in risky.iterrows():
            st.warning(f"**{p['web_name']}** ({p['chance_of_playing_next_round']:.0f}% chance)")
            st.write(f"📰 News: {p['news']}")
            st.write(f"📉 Minutes probability applied: {p['minutes_prob']:.2f}")

    st.subheader("Fixture Analysis")
    fixture_df = best_team[['web_name', 'next_opponent', 'fixture_difficulty']].copy()
    fixture_df = fixture_df.sort_values('fixture_difficulty')
    fixture_df['Rating'] = fixture_df['fixture_difficulty'].apply(
        lambda d: "🟩 Good" if d <= 2.8 else ("🟥 Tough" if d >= 3.5 else "🟨 Avg"))
    fixture_df = fixture_df.rename(columns={'web_name': 'Player', 'next_opponent': 'Next Match'})
    st.caption("Difficulty is the mean FDR of the next 5 fixtures, so it reflects the run "
               "ahead rather than this gameweek alone.")
    st.dataframe(
        fixture_df[['Player', 'Next Match', 'Rating', 'fixture_difficulty']],
        column_config={"fixture_difficulty": st.column_config.NumberColumn("Diff (1-5)",
                                                                           format="%.1f")},
        hide_index=True,
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# TAB 4 — Rival spy
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("🕵️‍♂️ Rival Scout")
    spy_league_id = st.text_input("League ID", value=str(int(league_id)))

    rival_members = get_league_members(int(spy_league_id)) if spy_league_id.isdigit() else {}
    rival_map = {name: entry for name, entry in rival_members.items() if entry != team_id}

    if not rival_map:
        st.info("No other managers found in this league yet.")
    else:
        target_name = st.selectbox("Select Target", list(rival_map.keys()))
        target_id = rival_map[target_name]

        if st.button("Analyze Head-to-Head"):
            with st.spinner(f"Comparing vs {target_name}..."):
                fpl = FPLClient()
                rival_fh = fpl.get_freehit_gws(target_id)
                rival_picks = fpl.get_team_picks(target_id, gw, freehit_gws=rival_fh)

            if not rival_picks:
                st.error("Could not fetch the rival's team.")
            else:
                rival_df = df[df['id'].isin([p['element'] for p in rival_picks['picks']])]
                analysis = RivalSpy(current_team_df, rival_df).compare()

                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Common Players", analysis['common_count'])
                m2.metric("Differentials", analysis['differential_count'])
                swing = analysis['net_swing']
                m3.metric("Projected Swing", f"{swing:+.1f} XP", delta=f"{swing:.1f}")
                st.caption(f"Horizon: GW{gw} only · differentials only · no captaincy applied")

                if swing < -5:
                    st.info(f"""
                    **Why you're behind:**
                    • They own **{analysis['rival_heavy_hitters']}** high-XP differentials.
                    • You have **{analysis['my_zeros']}** differentials projected near 0 XP.
                    • Biggest gap: **{analysis['main_gap_pos']}**.
                    """)

                st.subheader("⚡ Differential Battle")

                def format_player(p):
                    xp, name = p['predicted_points'], p['web_name']
                    if xp >= 6.0:
                        return f":red[**{name}**] ({xp:.1f} XP)"
                    if xp >= 5.0:
                        return f":orange[**{name}**] ({xp:.1f} XP)"
                    if xp < 0.5:
                        return f"{name} (⚠️ {xp:.1f} XP)"
                    return f"**{name}** ({xp:.1f} XP)"

                c1, c2 = st.columns(2)
                for col, caption, frame in (
                    (c1, "🛡️ You Have (Unique)", analysis['my_diffs']),
                    (c2, "⚔️ They Have (Unique)", analysis['rival_diffs']),
                ):
                    with col:
                        st.caption(caption)
                        for _, p in frame.iterrows():
                            ic, nc = st.columns([1, 4])
                            with ic:
                                player_image(p, width=40)
                            with nc:
                                st.markdown(format_player(p))

                if analysis['danger_player'] is not None:
                    dp = analysis['danger_player']
                    if dp['predicted_points'] >= 6.0:
                        st.warning(f"⚠️ **Major Threat**: {dp['web_name']} is their biggest "
                                   f"differential ({dp['predicted_points']:.1f} XP).")
                    else:
                        st.info(f"ℹ️ **Top Scout Target**: {dp['web_name']} is their highest "
                                f"unique ({dp['predicted_points']:.1f} XP).")
