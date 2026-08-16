"""
Pre-season (cold-start) predictions.

Before a ball is kicked there are no rolling features, so the ML model would read an
identical all-zero row for every player. FPL's own `ep_next` is no help either: it is
capped at 4.0 and heavily tied (88 players share exactly 1.0), which made the drafted
squad effectively arbitrary at the one moment the tool matters most.

`history_past` IS populated pre-season and carries each player's previous-season totals.
"""
import pandas as pd
import pytest

import src.model.predictor as predictor_mod
from src.model.predictor import PointsPredictor


def summaries(spec):
    """spec: {player_id: [season_total_points, ...]} oldest season first."""
    return {
        str(pid): {
            'history': [],
            'history_past': [{'season_name': f'20{20 + i}/{21 + i}', 'total_points': pts,
                              'minutes': 2500} for i, pts in enumerate(totals)],
        }
        for pid, totals in spec.items()
    }


def frame(ids, ep_next=None, minutes_prob=None):
    return pd.DataFrame({
        'id': ids,
        'web_name': [f'P{i}' for i in ids],
        'ep_next': ep_next if ep_next is not None else ['2.0'] * len(ids),
        'minutes_prob': minutes_prob if minutes_prob is not None else [1.0] * len(ids),
    })


# ---------------------------------------------------------------- the prior
def test_prior_is_points_per_gameweek():
    prior = PointsPredictor._previous_season_prior(summaries({1: [190]}))
    assert prior[1] == pytest.approx(190 / 38)


def test_prior_weights_recent_seasons_more():
    """Two seasons, 0.7/0.3 toward the most recent."""
    prior = PointsPredictor._previous_season_prior(summaries({1: [76, 190]}))
    expected = (0.7 * (190 / 38) + 0.3 * (76 / 38)) / 1.0
    assert prior[1] == pytest.approx(expected)


def test_prior_ranks_a_consistent_scorer_above_an_injured_one():
    """Points per 38 already discounts missed matches."""
    prior = PointsPredictor._previous_season_prior(summaries({1: [200], 2: [90]}))
    assert prior[1] > prior[2]


def test_prior_skips_players_with_no_history():
    prior = PointsPredictor._previous_season_prior({'5': {'history_past': []}})
    assert 5 not in prior


def test_prior_handles_empty_and_missing_input():
    assert PointsPredictor._previous_season_prior({}) == {}
    assert PointsPredictor._previous_season_prior(None) == {}


def test_prior_is_never_negative():
    prior = PointsPredictor._previous_season_prior(summaries({1: [-20]}))
    assert prior[1] >= 0


# ---------------------------------------------------------------- prediction mode
@pytest.fixture
def preseason(monkeypatch, preseason_bootstrap):
    monkeypatch.setattr(predictor_mod, 'load_bootstrap', lambda *a, **k: preseason_bootstrap)
    return preseason_bootstrap


def test_preseason_is_its_own_mode_not_a_fallback(preseason, monkeypatch):
    """It must not be reported as an error — there is nothing wrong."""
    monkeypatch.setattr(predictor_mod, 'load_summary_cache',
                        lambda *a, **k: (summaries({1: [190], 2: [76]}), None))
    p = PointsPredictor()
    out = p.predict(frame([1, 2]))
    assert p.prediction_mode == 'preseason'
    assert (out['prediction_mode'] == 'preseason').all()


def test_preseason_predictions_differentiate_players(preseason, monkeypatch):
    """The bug: every player tied, so the drafted squad was arbitrary."""
    monkeypatch.setattr(predictor_mod, 'load_summary_cache',
                        lambda *a, **k: (summaries({1: [239], 2: [190], 3: [76], 4: [20]}), None))
    out = PointsPredictor().predict(frame([1, 2, 3, 4]))
    assert out['predicted_points'].nunique() == 4
    assert list(out.sort_values('predicted_points', ascending=False)['id']) == [1, 2, 3, 4]


def test_players_new_to_the_league_fall_back_to_ep_next(preseason, monkeypatch):
    monkeypatch.setattr(predictor_mod, 'load_summary_cache',
                        lambda *a, **k: (summaries({1: [190]}), None))
    out = PointsPredictor().predict(frame([1, 99], ep_next=['0.0', '3.5']))
    assert out.loc[out['id'] == 99, 'predicted_points'].iloc[0] == pytest.approx(3.5)


def test_preseason_applies_the_availability_haircut(preseason, monkeypatch):
    monkeypatch.setattr(predictor_mod, 'load_summary_cache',
                        lambda *a, **k: (summaries({1: [190], 2: [190]}), None))
    out = PointsPredictor().predict(frame([1, 2], minutes_prob=[1.0, 0.25]))
    a = out.loc[out['id'] == 1, 'predicted_points'].iloc[0]
    b = out.loc[out['id'] == 2, 'predicted_points'].iloc[0]
    assert b == pytest.approx(a * 0.25)


def test_preseason_without_a_cache_still_produces_usable_output(preseason, monkeypatch):
    monkeypatch.setattr(predictor_mod, 'load_summary_cache', lambda *a, **k: (None, 'no cache'))
    p = PointsPredictor()
    out = p.predict(frame([1, 2], ep_next=['4.0', '1.0']))
    assert p.prediction_mode == 'preseason'
    assert out['predicted_points'].tolist() == pytest.approx([4.0, 1.0])
    assert any('async_fpl' in w for w in p.prediction_warnings)


def test_preseason_emits_all_columns_downstream_code_needs(preseason, monkeypatch):
    monkeypatch.setattr(predictor_mod, 'load_summary_cache',
                        lambda *a, **k: (summaries({1: [190]}), None))
    out = PointsPredictor().predict(frame([1]))
    for col in ['predicted_points', 'captaincy_score', 'projected_minutes',
                'start_probability', 'odds_confidence', 'prediction_mode']:
        assert col in out.columns


def test_preseason_does_not_mutate_input(preseason, monkeypatch):
    monkeypatch.setattr(predictor_mod, 'load_summary_cache',
                        lambda *a, **k: (summaries({1: [190]}), None))
    df = frame([1])
    before = df.copy()
    PointsPredictor().predict(df)
    pd.testing.assert_frame_equal(df, before)


def test_mid_season_does_not_take_the_preseason_path(monkeypatch, bootstrap):
    """`bootstrap` has a current gameweek, so the ML path must be attempted."""
    monkeypatch.setattr(predictor_mod, 'load_bootstrap', lambda *a, **k: bootstrap)
    monkeypatch.setattr(predictor_mod, 'load_summary_cache', lambda *a, **k: (None, 'boom'))
    p = PointsPredictor()
    p.predict(frame([1]))
    assert p.prediction_mode == 'fallback'
