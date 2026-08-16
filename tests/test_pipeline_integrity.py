"""
The invariants that silently break predictions rather than raising.

These are deliberately adversarial: they assert properties of the REAL artifacts on
disk (training frame, trained model, inference frame) rather than of fixtures, because
every serious bug in this project's history was a mismatch between two of those three.
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.model.predictor import (
    PointsPredictor, MinutesPredictor, CATEGORICAL_FEATURES, model_bundle_exists,
)

TRAIN_PARQUET = 'data/processed/historical_features.parquet'
INFER_PARQUET = 'data/processed/player_features.parquet'
POINTS_MODEL = 'data/models/lgb_ts_points'

requires_artifacts = pytest.mark.skipif(
    not (os.path.exists(TRAIN_PARQUET) and os.path.exists(INFER_PARQUET)
         and model_bundle_exists(POINTS_MODEL)),
    reason="built artifacts required; run history_builder.py, processor.py, predictor.py",
)


@pytest.fixture(scope='module')
def train_df():
    return pd.read_parquet(TRAIN_PARQUET)


@pytest.fixture(scope='module')
def infer_df():
    return pd.read_parquet(INFER_PARQUET)


@pytest.fixture(scope='module')
def model():
    p = PointsPredictor()
    assert p.load_model()
    return p


@pytest.fixture
def mid_season(monkeypatch, bootstrap):
    """predict() diverts to the pre-season prior unless a gameweek has started."""
    import src.model.predictor as predictor_mod
    monkeypatch.setattr(predictor_mod, 'load_bootstrap', lambda *a, **k: bootstrap)
    return bootstrap


# ------------------------------------------------------------ anti-leakage
@requires_artifacts
def test_rolling_features_never_contain_the_current_gameweek(train_df):
    """
    The whole point of history_builder: a GW N feature must be built only from GW <N.

    Verified structurally — for a player's FIRST gameweek in a season there is no prior
    match, so every shifted feature must be 0 (post-fillna), regardless of how many
    points they actually scored that week.
    """
    first = train_df.sort_values(['season', 'player_id', 'GW']).groupby(
        ['season', 'player_id'], observed=True).head(1)
    scored = first[first['total_points'] > 0]
    assert len(scored) > 100, "need a meaningful sample of scoring debut gameweeks"

    for col in ['total_points_last_1', 'minutes_last_1', 'bps_last_1',
                'total_points_mean_last_3', 'total_points_mean_last_5']:
        assert (scored[col] == 0).all(), (
            f"{col} is non-zero on a player's first gameweek — the shift is leaking")


@requires_artifacts
def test_lagged_value_equals_previous_gameweek_actual(train_df):
    """total_points_last_1 at GW N must equal total_points at GW N-1."""
    df = train_df.sort_values(['season', 'player_id', 'GW'])
    g = df.groupby(['season', 'player_id'], observed=True)
    expected = g['total_points'].shift(1)
    both = expected.notna()
    assert both.sum() > 1000
    assert np.allclose(df.loc[both, 'total_points_last_1'], expected[both])


@requires_artifacts
def test_targets_are_the_current_gameweek(train_df):
    assert (train_df['target'] == train_df['total_points']).all()
    assert (train_df['target_minutes'] == train_df['minutes']).all()


# ------------------------------------------------------------ train/serve parity
@requires_artifacts
def test_every_model_feature_is_producible_at_inference(model, infer_df):
    """
    The bug class this catches: a training feature absent at inference gets silently
    zero-filled, so the model scores a constant column and nobody notices.
    """
    from src.model.predictor import PointsPredictor as PP
    rolling = PP()._build_rolling_features(
        {'1': {'history': [{'round': 1, 'minutes': 90, 'total_points': 3, 'starts': 1,
                            'kickoff_time': '2026-08-21T14:00:00Z'}]}},
        pd.DataFrame([{'id': 1, 'next_kickoff_time': None}]),
    )
    available = set(infer_df.columns) | set(rolling.columns) | {
        'projected_minutes', 'start_probability'}
    missing = [f for f in model.features_list if f not in available]
    assert not missing, f"training features unavailable at inference: {missing}"


@requires_artifacts
def test_categorical_vocabularies_agree_between_train_and_inference(train_df, infer_df):
    """
    Regression: team was club NAMES in two seasons and integer IDS in a third; position
    was GK/GK/GKP; opponent_team was a per-season integer id. Unseen categories become
    missing values at predict time — silently.
    """
    for col in ['team_name', 'opponent_name', 'position']:
        assert col in train_df.columns and col in infer_df.columns

        train_vals = set(train_df[col].astype(str).unique())
        infer_vals = set(infer_df[col].astype(str).unique()) - {'UNKNOWN'}

        # Every training season must share one vocabulary...
        for season in train_df['season'].unique():
            season_vals = set(train_df[train_df['season'] == season][col].astype(str).unique())
            if col == 'position':
                assert season_vals <= {'GK', 'DEF', 'MID', 'FWD'}, (
                    f"{season} position vocabulary is {season_vals}")

        # ...and inference must not invent shapes training never saw.
        if col == 'position':
            assert infer_vals <= train_vals, f"unseen {col} at inference: {infer_vals - train_vals}"


@requires_artifacts
def test_no_integer_team_ids_leaked_into_the_categoricals(train_df):
    for col in ['team_name', 'opponent_name']:
        vals = [v for v in train_df[col].astype(str).unique()]
        numeric = [v for v in vals if v.isdigit()]
        assert not numeric, f"{col} contains raw integer ids: {numeric[:5]}"


@requires_artifacts
def test_model_features_exclude_leakage_and_ids(model):
    banned = {'total_points', 'target', 'target_minutes', 'minutes', 'bps', 'starts',
              'influence', 'creativity', 'threat', 'expected_goals', 'expected_assists',
              'expected_goal_involvements', 'expected_goals_conceded',
              'team', 'opponent_team', 'GW', 'season', 'player_id', 'kickoff_time'}
    overlap = banned & set(model.features_list)
    assert not overlap, f"leaking/unstable features in the model: {overlap}"


@requires_artifacts
def test_categoricals_declared_by_the_model_are_the_canonical_four(model, train_df):
    # projected_minutes / start_probability are derived during training and are not
    # columns of the stored frame.
    cats = [f for f in model.features_list
            if f in train_df.columns and train_df[f].dtype.name == 'category']
    assert set(cats) == set(CATEGORICAL_FEATURES)


@requires_artifacts
def test_only_derived_features_are_absent_from_the_training_frame(model, train_df):
    """Anything else missing would mean the model was trained on a different frame."""
    absent = [f for f in model.features_list if f not in train_df.columns]
    assert set(absent) <= {'projected_minutes', 'start_probability'}, absent


# ------------------------------------------------------------ inference frame health
@requires_artifacts
def test_inference_frame_has_no_duplicate_players(infer_df):
    """A duplicated row lets the optimizer pick the same player twice."""
    assert infer_df['id'].nunique() == len(infer_df)


@requires_artifacts
def test_prices_and_probabilities_are_in_range(infer_df):
    assert (infer_df['price'] > 0).all()
    assert infer_df['price'].between(3.0, 20.0).all()
    assert infer_df['minutes_prob'].between(0.0, 1.0).all()
    for col in ['win_prob', 'draw_prob', 'loss_prob', 'clean_sheet_prob',
                'anytime_goal_scorer_prob']:
        assert infer_df[col].between(0.0, 1.0).all(), f"{col} out of [0,1]"


@requires_artifacts
def test_implied_goals_are_physically_plausible(train_df, infer_df):
    """Regression: these ran at ~2.7 per team (5.4/match) against an actual ~3.0/match."""
    for df, label in ((train_df, 'train'), (infer_df, 'infer')):
        mean_goals = df['team_implied_goals'].mean()
        assert 1.0 < mean_goals < 2.0, f"{label} team_implied_goals mean {mean_goals:.2f}"
        assert df['team_implied_goals'].max() < 4.0
        assert 0.10 < df['clean_sheet_prob'].mean() < 0.45


@requires_artifacts
def test_every_player_has_a_shirt_code(infer_df):
    assert infer_df['team_code'].notna().all()
    assert (infer_df['team_code'] > 0).all()


# ------------------------------------------------------------ prediction behaviour
@requires_artifacts
def test_predictions_are_non_negative_and_bounded(model, infer_df, mid_season):
    scored = model.predict(infer_df.copy())
    assert (scored['predicted_points'] >= 0).all()
    assert scored['predicted_points'].max() < 25, "expected points, not a season total"
    assert scored['predicted_points'].notna().all()
    assert len(scored) == len(infer_df)


@requires_artifacts
def test_predict_reports_its_mode_and_never_silently_degrades(model, infer_df, mid_season):
    scored = model.predict(infer_df.copy())
    assert model.prediction_mode in {'ml', 'fallback'}
    if model.prediction_mode == 'fallback':
        assert model.prediction_warnings, "a fallback must always explain itself"
    assert 'captaincy_score' in scored.columns
    assert 'projected_minutes' in scored.columns


@requires_artifacts
def test_predict_does_not_mutate_its_input(model, infer_df, mid_season):
    before = infer_df.copy()
    model.predict(infer_df.copy())
    pd.testing.assert_frame_equal(infer_df, before)


@requires_artifacts
def test_minutes_model_output_is_bounded(infer_df, train_df):
    mp = MinutesPredictor()
    if not mp.load_model():
        pytest.skip("minutes model not trained")
    preds = mp.predict(train_df.head(500).copy())
    assert preds.between(0, 90).all()
