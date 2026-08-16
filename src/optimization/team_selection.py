import pandas as pd

# FPL formation rules for the starting XI.
FORMATION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}   # GK, DEF, MID, FWD
FORMATION_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
XI_SIZE = 11


def _order_bench(df_bench):
    """Reserve keeper first — auto-subs can only use the GK for the GK."""
    if df_bench.empty:
        return df_bench
    df_bench = df_bench.assign(_is_gk=(df_bench['element_type'] == 1).astype(int))
    return df_bench.sort_values(
        ['_is_gk', 'predicted_points'], ascending=[False, False]).drop(columns='_is_gk')


def select_starting_xi(team_df):
    """
    Split a squad into the best legal starting XI and its bench.

    If the frame carries `is_starter` — set by the optimizer, which chooses the squad
    and the XI jointly — that decision is honoured. Re-deriving it here would risk
    displaying a different XI from the one the objective was actually maximised over.

    Otherwise: take the top player at each position to satisfy the minimums, then fill
    the remaining slots greedily by predicted points subject to the per-position maxima.

    Tolerates an incomplete squad (fewer than 15 players, or a position with no
    players at all) rather than raising — squads assembled from API picks can be short
    if a player has left the league since the picks were made.
    """
    if team_df is None or team_df.empty:
        empty = pd.DataFrame(columns=getattr(team_df, 'columns', []))
        return empty, empty

    if 'is_starter' in team_df.columns and team_df['is_starter'].any():
        starters = team_df[team_df['is_starter'].astype(bool)].sort_values(
            'predicted_points', ascending=False)
        bench = _order_bench(team_df[~team_df['is_starter'].astype(bool)])
        return starters, bench

    team_df = team_df.copy().sort_values('predicted_points', ascending=False)

    by_pos = {pos: team_df[team_df['element_type'] == pos] for pos in (1, 2, 3, 4)}

    starter_idxs = []
    bench_idxs = []
    counts = {}

    # 1. Satisfy the minimum at each position with that position's best players.
    for pos in (1, 2, 3, 4):
        take = min(FORMATION_MIN[pos], len(by_pos[pos]))
        starter_idxs.extend(by_pos[pos].iloc[:take].index.tolist())
        counts[pos] = take

    # 2. Fill the remaining slots greedily, respecting the maxima.
    pool = pd.concat(
        [by_pos[pos].iloc[counts[pos]:] for pos in (1, 2, 3, 4)]
    ).sort_values('predicted_points', ascending=False)

    for idx, player in pool.iterrows():
        pos = player['element_type']
        if len(starter_idxs) < XI_SIZE and counts.get(pos, 0) < FORMATION_MAX.get(pos, 0):
            starter_idxs.append(idx)
            counts[pos] = counts.get(pos, 0) + 1
        else:
            bench_idxs.append(idx)

    return team_df.loc[starter_idxs], _order_bench(team_df.loc[bench_idxs])


def squad_expected_points(starters, captain_id=None):
    """
    Expected points actually scored by a squad: the XI only, with the captain doubled.

    Comparing 15-man squad totals (as the dashboard used to) overstates every
    improvement, because bench points are not scored.
    """
    if starters is None or starters.empty:
        return 0.0
    total = float(starters['predicted_points'].sum())
    if captain_id is not None:
        cap = starters[starters['id'] == captain_id]
        if not cap.empty:
            total += float(cap.iloc[0]['predicted_points'])
    return total


def pick_captain(starters):
    """
    (captain, vice) rows, ranked by captaincy_score when available.

    captaincy_score discounts rotation risk and a hard fixture — both matter more when
    the score is doubled — so it is a better captaincy ranking than raw predicted points.

    Goalkeepers are excluded unless the XI contains nothing else. A keeper's scoring
    ceiling is far below an attacker's, so doubling one is never the right call even
    when a flat model ranks them top — which is exactly what happens pre-season, when
    every projection collapses toward the same value.
    """
    if starters is None or starters.empty:
        return None, None

    col = 'captaincy_score' if 'captaincy_score' in starters.columns else 'predicted_points'
    ranked = starters.sort_values(col, ascending=False)

    outfield = ranked[ranked['element_type'] != 1]
    if not outfield.empty:
        ranked = outfield

    # Honour the optimizer's captain: it priced the armband into the objective, so the
    # squad was built around that player doubling.
    if 'is_captain' in starters.columns and starters['is_captain'].astype(bool).any():
        chosen = starters[starters['is_captain'].astype(bool)].iloc[0]
        rest = ranked[ranked['id'] != chosen['id']]
        return chosen, (rest.iloc[0] if not rest.empty else None)

    captain = ranked.iloc[0]
    vice = ranked.iloc[1] if len(ranked) > 1 else None
    return captain, vice
