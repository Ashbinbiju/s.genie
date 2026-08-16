"""
Exercises the real ML prediction path.

Pre-season the app legitimately falls back to a heuristic, so the ML branch is never
reached by a live run. These tests synthesise a valid season-stamped cache for the
CURRENT season's real player ids so the two-stage model actually executes.
"""
import os

import numpy as np
import pandas as pd
import pytest

import src.model.predictor as predictor_mod
from src.model.predictor import PointsPredictor
from src.api.understat import UnderstatClient
from src.optimization.solver import TransferOptimizer

from src.model.predictor import model_bundle_exists

INFER_PARQUET = 'data/processed/player_features.parquet'
POINTS_MODEL = 'data/models/lgb_ts_points'
MINUTES_MODEL = 'data/models/lgb_ts_minutes'

# Resolved through model_bundle_exists so a rename of the on-disk format cannot turn
# these into silent skips again.
requires_model = pytest.mark.skipif(
    not (os.path.exists(INFER_PARQUET) and model_bundle_exists(POINTS_MODEL)
         and model_bundle_exists(MINUTES_MODEL)),
    reason="trained model + inference frame required",
)


def synth_summaries(ids, n_gw=6, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for pid in ids:
        history = []
        for gw in range(1, n_gw + 1):
            minutes = int(rng.choice([0, 20, 65, 90], p=[.15, .15, .2, .5]))
            history.append({
                'round': gw,
                'kickoff_time': f'2026-{8 + gw // 5:02d}-{(gw * 7) % 28 + 1:02d}T14:00:00Z',
                'minutes': minutes,
                'total_points': int(max(0, rng.normal(3, 2))) if minutes else 0,
                'starts': 1 if minutes >= 60 else 0,
                'expected_goals': float(rng.uniform(0, .6)),
                'expected_assists': float(rng.uniform(0, .4)),
                'expected_goal_involvements': float(rng.uniform(0, .9)),
                'expected_goals_conceded': float(rng.uniform(0, 2)),
                'bps': int(rng.integers(0, 40)),
                'influence': float(rng.uniform(0, 60)),
                'creativity': float(rng.uniform(0, 60)),
                'threat': float(rng.uniform(0, 60)),
            })
        out[str(pid)] = {'history': history}
    return out


@pytest.fixture
def infer_df():
    return pd.read_parquet(INFER_PARQUET)


@pytest.fixture(autouse=True)
def mid_season(monkeypatch, bootstrap):
    """
    Force a mid-season view of the world.

    predict() short-circuits to the pre-season prior when no gameweek has started, so
    without this every test here would quietly stop exercising the ML path — which is
    exactly the kind of silent skip these tests exist to prevent.
    """
    monkeypatch.setattr(predictor_mod, 'load_bootstrap', lambda *a, **k: bootstrap)
    return bootstrap


@pytest.fixture
def with_valid_cache(monkeypatch, infer_df):
    summaries = synth_summaries(infer_df['id'].tolist())
    monkeypatch.setattr(predictor_mod, 'load_summary_cache',
                        lambda *a, **k: (summaries, None))
    return summaries


@requires_model
def test_ml_path_actually_runs(with_valid_cache, infer_df):
    p = PointsPredictor()
    out = p.predict(infer_df.copy())
    assert p.prediction_mode == 'ml', f"fell back: {p.prediction_warnings}"
    assert len(out) == len(infer_df)


@requires_model
def test_ml_path_zero_fills_nothing(with_valid_cache, infer_df):
    """
    The loudest silent failure: a training feature missing at inference is zero-filled.
    With a valid cache there must be nothing to fill.
    """
    p = PointsPredictor()
    p.predict(infer_df.copy())
    zero_fill = [w for w in p.prediction_warnings if 'zero-filled' in w]
    assert not zero_fill, zero_fill


@requires_model
def test_ml_predictions_are_sane(with_valid_cache, infer_df):
    p = PointsPredictor()
    out = p.predict(infer_df.copy())
    pts = out['predicted_points']
    assert pts.notna().all()
    assert (pts >= 0).all()
    assert pts.max() < 20, "expected points for one gameweek, not a season"
    assert pts.std() > 0.1, "predictions must vary between players"


@requires_model
def test_projected_minutes_are_bounded_and_vary(with_valid_cache, infer_df):
    p = PointsPredictor()
    out = p.predict(infer_df.copy())
    assert out['projected_minutes'].between(0, 90).all()
    assert out['projected_minutes'].std() > 1.0
    assert out['start_probability'].isin([0.0, 1.0]).all()


@requires_model
def test_captaincy_score_penalises_rotation_risk(with_valid_cache, infer_df):
    """Two players with equal XP must rank by minutes confidence, not tie."""
    p = PointsPredictor()
    out = p.predict(infer_df.copy())
    assert (out['captaincy_score'] <= out['predicted_points'] + 1e-9).all(), (
        "captaincy_score is a discount on expected points, never a premium")
    starters = out[out['projected_minutes'] > 0]
    if len(starters) > 20:
        corr = starters[['captaincy_score', 'projected_minutes']].corr().iloc[0, 1]
        assert corr > 0, "more expected minutes must not lower the captaincy score"


@requires_model
def test_injury_news_reduces_expected_points(with_valid_cache, infer_df):
    df = infer_df.copy()
    df.loc[df.index[:5], 'minutes_prob'] = 0.25
    p = PointsPredictor()
    out = p.predict(df)
    healthy = out.loc[out.index[5:], 'predicted_points'].mean()
    flagged = out.loc[out.index[:5], 'predicted_points'].mean()
    assert flagged < healthy


@requires_model
def test_a_player_missing_from_the_cache_does_not_break_prediction(monkeypatch, infer_df):
    ids = infer_df['id'].tolist()
    partial = synth_summaries(ids[:-10])           # 10 players absent
    monkeypatch.setattr(predictor_mod, 'load_summary_cache', lambda *a, **k: (partial, None))
    p = PointsPredictor()
    out = p.predict(infer_df.copy())
    assert len(out) == len(infer_df)
    assert p.prediction_mode == 'ml'
    assert any('no history' in w for w in p.prediction_warnings)


@requires_model
def test_duplicate_ids_in_cache_trigger_the_fallback_not_a_desync(monkeypatch, infer_df):
    """A row-count change after the merge would desync the positional assignments."""
    summaries = synth_summaries(infer_df['id'].tolist()[:20])
    monkeypatch.setattr(predictor_mod, 'load_summary_cache', lambda *a, **k: (summaries, None))

    real_build = PointsPredictor._build_rolling_features

    def duplicating_build(self, s, df):
        out = real_build(self, s, df)
        return pd.concat([out, out.head(3)], ignore_index=True)   # inject duplicates

    monkeypatch.setattr(PointsPredictor, '_build_rolling_features', duplicating_build)
    p = PointsPredictor()
    out = p.predict(infer_df.copy())
    assert p.prediction_mode == 'fallback'
    assert len(out) == len(infer_df)


@requires_model
def test_empty_cache_falls_back(monkeypatch, infer_df):
    monkeypatch.setattr(predictor_mod, 'load_summary_cache', lambda *a, **k: ({}, None))
    p = PointsPredictor()
    out = p.predict(infer_df.copy())
    assert p.prediction_mode == 'fallback'
    assert len(out) == len(infer_df)


@requires_model
def test_ml_predictions_feed_a_legal_squad(with_valid_cache, infer_df):
    """End to end: predictions must be optimizable into a valid FPL squad."""
    p = PointsPredictor()
    scored = p.predict(infer_df.copy())
    squad = TransferOptimizer(budget=100.0).solve_team(scored)
    assert squad is not None
    assert len(squad) == 15
    assert squad['element_type'].value_counts().to_dict() == {1: 2, 2: 5, 3: 5, 4: 3}
    assert squad['team'].value_counts().max() <= 3
    assert squad['price'].sum() <= 100.0 + 1e-6


# ---------------------------------------------------------------- understat decode
def test_understat_decode_preserves_accented_names(monkeypatch):
    """
    Regression: encoding to utf-8 before 'unicode_escape' mangled every accented name
    ('Ødegaard' -> 'Ãdegaard'), which then failed to match the FPL name.
    """
    # Understat escapes the UTF-8 BYTES of each character, not its codepoint:
    # 'Ø' is C3 98 -> \xc3\x98, 'ã' is C3 A3 -> \xc3\xa3.
    payload = (r'[{"id":"1","player_name":"Martin \xc3\x98degaard","xG":"5.5","xA":"3.2",'
               r'"time":"2700"},{"id":"2","player_name":"Bruno Guimar\xc3\xa3es",'
               r'"xG":"2.1","xA":"1.0","time":"2500"}]')
    html = "var playersData = JSON.parse('" + payload + "');"

    class FakeResp:
        text = html
        def raise_for_status(self): pass

    monkeypatch.setattr('src.api.understat.requests.get', lambda *a, **k: FakeResp())
    df = UnderstatClient(year=2026).get_player_stats()

    assert df is not None
    names = list(df['player_name'])
    assert 'Martin Ødegaard' in names, names
    assert 'Bruno Guimarães' in names, names
