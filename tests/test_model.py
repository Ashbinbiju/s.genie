"""Cache integrity, rolling-feature parity with training, and odds calibration."""
import json

import numpy as np
import pandas as pd
import pytest

from src.model.predictor import load_summary_cache, PointsPredictor, CATEGORICAL_FEATURES
from src.api.async_fpl import cache_filename
from src.api.odds import (
    implied_goals_from_odds, LEAGUE_DEFAULTS, OddsClient,
    TOTAL_GOALS_FALLBACK, GOAL_SHARE_DAMPING,
)


# ---------------------------------------------------------------- cache guard
def write_cache(tmp_path, name, ids):
    payload = {str(i): {'history': []} for i in ids}
    (tmp_path / name).write_text(json.dumps(payload), encoding='utf-8')


def test_cache_filename_includes_season():
    """The season stamp is what makes a cross-season load impossible."""
    assert cache_filename('2026-27', 5) == 'element_summary_2026-27_gw_5.json'


def test_legacy_unstamped_cache_is_rejected(tmp_path, bootstrap):
    """Regression: a previous season's cache joined cleanly onto this season's players
    while describing entirely different people (100% id collision)."""
    write_cache(tmp_path, 'element_summary_gw_37.json', [10, 11, 12])
    summaries, err = load_summary_cache(cache_dir=str(tmp_path), static=bootstrap)
    assert summaries is None
    assert 'legacy' in err.lower()


def test_cache_from_another_season_is_not_found(tmp_path, bootstrap):
    write_cache(tmp_path, cache_filename('2025-26', 37), [10, 11, 12])
    summaries, err = load_summary_cache(cache_dir=str(tmp_path), static=bootstrap)
    assert summaries is None
    assert '2026-27' in err


def test_cache_with_wrong_players_is_rejected(tmp_path, bootstrap):
    """Even a correctly named file must describe the current squad."""
    write_cache(tmp_path, cache_filename('2026-27', 2), [900, 901, 902])
    summaries, err = load_summary_cache(cache_dir=str(tmp_path), static=bootstrap)
    assert summaries is None
    assert 'overlap' in err.lower()


def test_valid_cache_loads(tmp_path, bootstrap):
    write_cache(tmp_path, cache_filename('2026-27', 2), [10, 11, 12])
    summaries, err = load_summary_cache(cache_dir=str(tmp_path), static=bootstrap)
    assert err is None
    assert set(summaries) == {'10', '11', '12'}


def test_missing_cache_dir_is_reported(bootstrap):
    summaries, err = load_summary_cache(cache_dir='does/not/exist', static=bootstrap)
    assert summaries is None and 'does not exist' in err


# ---------------------------------------------------------------- rolling features
def hist(round_, minutes, points, starts=1, kickoff='2026-08-21T14:00:00Z'):
    return {'round': round_, 'minutes': minutes, 'total_points': points, 'starts': starts,
            'kickoff_time': kickoff, 'expected_goals': 0, 'expected_assists': 0,
            'expected_goal_involvements': 0, 'expected_goals_conceded': 0,
            'bps': 0, 'influence': 0, 'creativity': 0, 'threat': 0}


def build(history, next_kickoff=None):
    pp = PointsPredictor()
    df = pd.DataFrame([{'id': 1, 'next_kickoff_time': next_kickoff}])
    return pp._build_rolling_features({'1': {'history': history}}, df).iloc[0]


def test_rolling_means_use_the_last_n_gameweeks():
    row = build([hist(g, 90, g) for g in range(1, 6)])
    assert row['total_points_last_1'] == 5
    assert row['total_points_mean_last_3'] == pytest.approx((3 + 4 + 5) / 3)
    assert row['total_points_mean_last_5'] == pytest.approx(15 / 5)


def test_double_gameweek_is_collapsed_into_one_gameweek():
    """Regression: training groups by (player, GW) and SUMS; element-summary returns one
    entry PER MATCH, so a DGW made the last-3 window span 3 matches, not 3 gameweeks."""
    history = [hist(1, 90, 2), hist(2, 90, 5), hist(2, 60, 7), hist(3, 90, 1)]
    row = build(history)
    # GW2 must read as a single 12-point, 150-minute gameweek.
    assert row['total_points_mean_last_3'] == pytest.approx((2 + 12 + 1) / 3)
    assert row['minutes_last_1'] == 90


def test_benched_counts_use_starts():
    history = [hist(1, 0, 0, starts=0), hist(2, 90, 5, starts=1), hist(3, 0, 0, starts=0)]
    row = build(history)
    assert row['benched_sum_last_3'] == 2


def test_days_rest_measures_the_upcoming_fixture():
    """Regression: this measured the gap between the last two COMPLETED matches, one
    match out of phase with the training definition."""
    history = [hist(1, 90, 2, kickoff='2026-08-01T14:00:00Z'),
               hist(2, 90, 2, kickoff='2026-08-10T14:00:00Z')]
    row = build(history, next_kickoff='2026-08-14T14:00:00Z')
    assert row['days_rest'] == pytest.approx(4.0)


def test_days_rest_falls_back_when_no_fixture_scheduled():
    history = [hist(1, 90, 2, kickoff='2026-08-01T14:00:00Z'),
               hist(2, 90, 2, kickoff='2026-08-08T14:00:00Z')]
    row = build(history, next_kickoff=None)
    assert row['days_rest'] == pytest.approx(7.0)


def test_empty_history_produces_zeros_not_nan():
    row = build([])
    assert row['total_points_mean_last_3'] == 0.0
    assert row['days_rest'] == 7.0


def test_rolling_feature_names_match_training_vocabulary():
    """Inference must emit exactly the columns history_builder produced."""
    row = build([hist(1, 90, 3)])
    for col in ['minutes', 'total_points', 'bps', 'threat', 'starts']:
        assert f'{col}_last_1' in row
        assert f'{col}_mean_last_3' in row
        assert f'{col}_mean_last_5' in row
    assert 'benched_sum_last_3' in row and 'days_rest' in row


# ---------------------------------------------------------------- feature hygiene
def test_leakage_and_unstable_ids_are_excluded_from_features():
    pp = PointsPredictor()
    df = pd.DataFrame(columns=[
        'player_id', 'GW', 'season', 'total_points', 'target', 'target_minutes',
        'minutes', 'bps', 'starts', 'expected_goals', 'team', 'opponent_team',
        'kickoff_time', 'match_date',
        'price', 'team_name', 'opponent_name', 'position', 'total_points_mean_last_3',
    ])
    feats = pp._get_feature_cols(df)
    for banned in ['total_points', 'target', 'minutes', 'bps', 'starts', 'expected_goals',
                   'team', 'opponent_team', 'kickoff_time', 'match_date', 'GW', 'season']:
        assert banned not in feats, f'{banned} must not be a feature'
    for kept in ['price', 'team_name', 'opponent_name', 'position', 'total_points_mean_last_3']:
        assert kept in feats


def test_categorical_features_are_all_name_based():
    """Integer team ids are reassigned every season and must never be categoricals."""
    assert CATEGORICAL_FEATURES == ['position', 'team_name', 'opponent_name', 'was_home']
    assert 'team' not in CATEGORICAL_FEATURES
    assert 'opponent_team' not in CATEGORICAL_FEATURES


# ---------------------------------------------------------------- odds calibration
def test_implied_goals_are_not_inflated():
    """Regression: a stray *1.8 produced 5.36 goals/match against an actual ~3.0."""
    h, a, _, _ = implied_goals_from_odds(0.4, 0.35, over_odds=2.0, under_odds=2.0)
    total = h + a
    assert 2.0 < total < 3.5
    assert total == pytest.approx(1.5 + 0.5 * 2.5)


def test_goal_split_favours_the_stronger_side_but_is_damped():
    h, a, _, _ = implied_goals_from_odds(0.75, 0.10, 2.0, 2.0)
    assert h > a
    raw_share = 0.75 / 0.85
    assert (h / (h + a)) < raw_share, "win probability is more extreme than goal share"
    assert (h / (h + a)) == pytest.approx(0.5 + GOAL_SHARE_DAMPING * (raw_share - 0.5))


def test_even_match_splits_evenly():
    h, a, hcs, acs = implied_goals_from_odds(0.4, 0.4, 2.0, 2.0)
    assert h == pytest.approx(a)
    assert hcs == pytest.approx(acs)


def test_clean_sheet_is_the_poisson_zero_of_the_opponent():
    h, a, hcs, acs = implied_goals_from_odds(0.4, 0.35, 2.0, 2.0)
    assert hcs == pytest.approx(np.exp(-a))
    assert acs == pytest.approx(np.exp(-h))


def test_missing_over_under_uses_the_league_fallback():
    h, a, _, _ = implied_goals_from_odds(0.4, 0.4, 0, 0)
    assert h + a == pytest.approx(TOTAL_GOALS_FALLBACK)


def test_defaults_sit_on_the_same_scale_as_computed_values():
    """Regression: defaults were 1.35/0.30 while computed values were 2.68/0.15, making
    the feature encode 'did the odds lookup succeed' rather than anything about football."""
    h, a, hcs, acs = implied_goals_from_odds(1 / 3, 1 / 3, 0, 0)
    assert h == pytest.approx(LEAGUE_DEFAULTS['team_implied_goals'], abs=0.15)
    assert hcs == pytest.approx(LEAGUE_DEFAULTS['clean_sheet_prob'], abs=0.05)


@pytest.mark.parametrize('position,expected_order', [('FWD', 3), ('MID', 2), ('DEF', 1), ('GK', 0)])
def test_anytime_scorer_probability_ranks_by_position(position, expected_order):
    probs = {p: OddsClient.compute_anytime_scorer_prob(1.5, p)
             for p in ['GK', 'DEF', 'MID', 'FWD']}
    assert probs['FWD'] > probs['MID'] > probs['DEF'] > probs['GK']
    assert 0 <= probs[position] <= 1


def test_gkp_and_gk_score_identically():
    """Both spellings appear in the wild; they must not diverge."""
    assert (OddsClient.compute_anytime_scorer_prob(1.5, 'GK')
            == pytest.approx(OddsClient.compute_anytime_scorer_prob(1.5, 'GKP')))
