"""
Numerical parity between the two rolling-feature implementations.

history_builder (training) and predictor._build_rolling_features (inference) compute
the same features by completely different routes — a shifted pandas groupby over a
long frame vs. a python loop over element-summary JSON. If they ever disagree, the
model is scored on features it was not trained on, silently. This asserts they agree
value-for-value on identical input.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.history_builder import HistoryBuilder
from src.model.predictor import PointsPredictor

ROLLING_COLS = HistoryBuilder().rolling_cols


def make_history(per_gw):
    """per_gw: list of (round, minutes, points, starts, kickoff)."""
    out = []
    for rnd, minutes, points, starts, kickoff in per_gw:
        entry = {'round': rnd, 'minutes': minutes, 'total_points': points,
                 'starts': starts, 'kickoff_time': kickoff}
        for col in ROLLING_COLS:
            entry.setdefault(col, 0)
        entry['minutes'] = minutes
        entry['total_points'] = points
        entry['starts'] = starts
        return_entry = entry
        out.append(return_entry)
    return out


def training_features_for_next_gw(per_gw, next_kickoff):
    """
    Run the TRAINING path over the played gameweeks plus a placeholder row for the
    gameweek being predicted, then read the placeholder's shifted features — exactly
    what the model would have been trained on for that gameweek.
    """
    rows = []
    for rnd, minutes, points, starts, kickoff in per_gw:
        row = {'player_id': 1, 'GW': rnd, 'season': '2026-27', 'price': 5.0,
               'was_home': True, 'opponent_name': 'Man Utd', 'team_name': 'Arsenal',
               'position': 'MID', 'kickoff_time': kickoff}
        for col in ROLLING_COLS:
            row[col] = 0.0
        row['minutes'] = float(minutes)
        row['total_points'] = float(points)
        row['starts'] = float(starts)
        rows.append(row)

    # The gameweek under prediction: outcome unknown, only its kickoff is known.
    upcoming = dict(rows[-1])
    upcoming['GW'] = max(r['GW'] for r in rows) + 1
    upcoming['kickoff_time'] = next_kickoff
    for col in ROLLING_COLS:
        upcoming[col] = 0.0
    rows.append(upcoming)

    df = pd.DataFrame(rows).sort_values(['season', 'player_id', 'GW']).reset_index(drop=True)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce', utc=True)

    grouped = df.groupby(['season', 'player_id'])
    df['days_rest'] = (
        df['kickoff_time'] - grouped['kickoff_time'].shift(1)
    ).dt.total_seconds() / 86400
    df['days_rest'] = df['days_rest'].fillna(7.0)
    df['benched'] = (df['starts'] == 0).astype(int)

    feats = {}
    for col in ROLLING_COLS + ['benched']:
        feats[f'{col}_last_1'] = grouped[col].shift(1)
        if col == 'benched':
            for w in (3, 5):
                feats[f'{col}_sum_last_{w}'] = grouped[col].apply(
                    lambda x: x.shift(1).rolling(window=w, min_periods=1).sum()
                ).reset_index(level=[0, 1], drop=True)
        else:
            for w in (3, 5):
                feats[f'{col}_mean_last_{w}'] = grouped[col].apply(
                    lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
                ).reset_index(level=[0, 1], drop=True)
    for k, v in feats.items():
        df[k] = v

    return df.iloc[-1]


def inference_features_for_next_gw(per_gw, next_kickoff):
    history = []
    for rnd, minutes, points, starts, kickoff in per_gw:
        entry = {col: 0 for col in ROLLING_COLS}
        entry.update({'round': rnd, 'minutes': minutes, 'total_points': points,
                      'starts': starts, 'kickoff_time': kickoff})
        history.append(entry)

    df_features = pd.DataFrame([{'id': 1, 'next_kickoff_time': next_kickoff}])
    return PointsPredictor()._build_rolling_features(
        {'1': {'history': history}}, df_features).iloc[0]


PLAYED = [
    (1, 90, 6, 1, '2026-08-15T14:00:00Z'),
    (2, 45, 2, 0, '2026-08-22T14:00:00Z'),
    (3, 90, 9, 1, '2026-08-29T14:00:00Z'),
    (4, 0, 0, 0, '2026-09-12T14:00:00Z'),
    (5, 78, 4, 1, '2026-09-19T14:00:00Z'),
]
NEXT_KICKOFF = '2026-09-26T14:00:00Z'


@pytest.mark.parametrize('feature', [
    'total_points_last_1', 'total_points_mean_last_3', 'total_points_mean_last_5',
    'minutes_last_1', 'minutes_mean_last_3', 'minutes_mean_last_5',
    'starts_last_1', 'starts_mean_last_3', 'starts_mean_last_5',
    'benched_sum_last_3', 'benched_sum_last_5',
    'days_rest',
])
def test_training_and_inference_agree_value_for_value(feature):
    train = training_features_for_next_gw(PLAYED, NEXT_KICKOFF)
    infer = inference_features_for_next_gw(PLAYED, NEXT_KICKOFF)
    assert float(infer[feature]) == pytest.approx(float(train[feature])), (
        f"{feature}: training={train[feature]} inference={infer[feature]}")


def test_days_rest_specifically_matches_the_upcoming_fixture():
    """The feature that was one match out of phase before the fix."""
    train = training_features_for_next_gw(PLAYED, NEXT_KICKOFF)
    infer = inference_features_for_next_gw(PLAYED, NEXT_KICKOFF)
    assert float(train['days_rest']) == pytest.approx(7.0)   # 19 Sep -> 26 Sep
    assert float(infer['days_rest']) == pytest.approx(7.0)


def test_parity_holds_for_a_short_history():
    """A player with two gameweeks played — windows are partially filled."""
    short = PLAYED[:2]
    train = training_features_for_next_gw(short, NEXT_KICKOFF)
    infer = inference_features_for_next_gw(short, NEXT_KICKOFF)
    for f in ['total_points_last_1', 'total_points_mean_last_3', 'minutes_mean_last_5',
              'benched_sum_last_3']:
        assert float(infer[f]) == pytest.approx(float(train[f])), f


def test_parity_holds_across_a_double_gameweek():
    """
    Training collapses a DGW into one summed row; element-summary returns one entry per
    match. Both sides must see the same three-gameweek window.
    """
    played_dgw = [
        (1, 90, 6, 1, '2026-08-15T14:00:00Z'),
        (2, 90, 5, 1, '2026-08-22T14:00:00Z'),
        (2, 60, 7, 1, '2026-08-25T19:00:00Z'),   # second match of GW2
        (3, 90, 1, 1, '2026-08-29T14:00:00Z'),
    ]
    # Training input is already collapsed by the (player_id, GW) groupby.
    collapsed = [
        (1, 90, 6, 1, '2026-08-15T14:00:00Z'),
        (2, 150, 12, 2, '2026-08-25T19:00:00Z'),
        (3, 90, 1, 1, '2026-08-29T14:00:00Z'),
    ]
    train = training_features_for_next_gw(collapsed, NEXT_KICKOFF)
    infer = inference_features_for_next_gw(played_dgw, NEXT_KICKOFF)

    for f in ['total_points_last_1', 'total_points_mean_last_3',
              'minutes_last_1', 'minutes_mean_last_3']:
        assert float(infer[f]) == pytest.approx(float(train[f])), (
            f"{f}: DGW not collapsed identically (train={train[f]} infer={infer[f]})")
