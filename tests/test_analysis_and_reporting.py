"""Rival comparison, report generation, fixture difficulty and the fallback path."""
import pandas as pd
import pytest

from src.analysis.rivals import RivalSpy
from src.interface.reporter import ReportGenerator
from src.features.processor import FeatureProcessor
from src.model.predictor import PointsPredictor
from conftest import make_squad


# ---------------------------------------------------------------- rival spy
def squad_with(ids, points):
    return pd.DataFrame({
        'id': ids,
        'web_name': [f'P{i}' for i in ids],
        'predicted_points': points,
        'element_type': [(i % 4) + 1 for i in ids],
    })


def test_rival_spy_identifies_shared_and_unique_players():
    mine = squad_with([1, 2, 3, 4], [5.0, 4.0, 3.0, 2.0])
    theirs = squad_with([3, 4, 5, 6], [3.0, 2.0, 6.0, 1.0])
    a = RivalSpy(mine, theirs).compare()

    assert a['common_count'] == 2
    assert a['differential_count'] == 2
    assert set(a['my_diffs']['id']) == {1, 2}
    assert set(a['rival_diffs']['id']) == {5, 6}


def test_rival_spy_swing_is_differentials_only():
    mine = squad_with([1, 2, 3], [5.0, 4.0, 100.0])
    theirs = squad_with([3, 4], [100.0, 2.0])
    a = RivalSpy(mine, theirs).compare()
    # Player 3 is shared and must cancel out entirely.
    assert a['net_swing'] == pytest.approx((5.0 + 4.0) - 2.0)


def test_rival_spy_flags_the_biggest_threat():
    mine = squad_with([1], [1.0])
    theirs = squad_with([2, 3], [2.0, 9.0])
    a = RivalSpy(mine, theirs).compare()
    assert a['danger_player']['id'] == 3


def test_rival_spy_handles_identical_squads():
    same = squad_with([1, 2, 3], [5.0, 4.0, 3.0])
    a = RivalSpy(same, same.copy()).compare()
    assert a['differential_count'] == 0
    assert a['net_swing'] == 0
    assert a['danger_player'] is None


def test_rival_spy_diffs_are_sorted_by_points():
    mine = squad_with([1, 2, 3], [1.0, 9.0, 5.0])
    theirs = squad_with([9], [1.0])
    a = RivalSpy(mine, theirs).compare()
    assert list(a['my_diffs']['predicted_points']) == [9.0, 5.0, 1.0]


# ---------------------------------------------------------------- reporter
def test_report_is_written_as_utf8_with_accented_names(tmp_path):
    """Regression: the default Windows codepage raised UnicodeEncodeError on these."""
    squad = make_squad()
    squad.loc[0, 'web_name'] = 'Højlund'
    squad.loc[1, 'web_name'] = 'Sánchez'
    squad.loc[2, 'web_name'] = 'Güéhi'

    out = tmp_path / 'nested' / 'reports'   # also proves the dir is created
    content = ReportGenerator(output_dir=str(out)).generate(5, squad, captain='Højlund')

    written = (out / 'gw5_report.txt').read_text(encoding='utf-8')
    assert 'Højlund' in written and 'Sánchez' in written
    assert 'Højlund' in content


def test_report_labels_a_full_squad_honestly(tmp_path):
    """It used to print all 15 players under a 'Starting XI:' heading."""
    squad = make_squad()
    content = ReportGenerator(output_dir=str(tmp_path)).generate(5, squad)
    assert 'Starting XI:' not in content
    assert 'Squad (15 players)' in content


def test_report_uses_starters_when_given(tmp_path):
    from src.optimization.team_selection import select_starting_xi
    squad = make_squad()
    starters, _ = select_starting_xi(squad)
    content = ReportGenerator(output_dir=str(tmp_path)).generate(5, squad, starters=starters)
    assert 'Starting XI:' in content


def test_report_vice_differs_from_captain(tmp_path):
    squad = make_squad()
    content = ReportGenerator(output_dir=str(tmp_path)).generate(
        5, squad, captain=squad.iloc[0]['web_name'])
    lines = dict(l.split(': ', 1) for l in content.splitlines() if l.startswith(('Captain:', 'Vice:')))
    assert lines['Captain'] != lines['Vice']


# ---------------------------------------------------------------- fixture difficulty
def test_fixture_difficulty_averages_the_next_n_and_names_the_next_opponent():
    teams = pd.DataFrame([
        {'id': 1, 'name': 'Arsenal', 'short_name': 'ARS', 'code': 3},
        {'id': 2, 'name': 'Man Utd', 'short_name': 'MUN', 'code': 1},
    ])
    fixtures = pd.DataFrame([
        {'team_h': 1, 'team_a': 2, 'finished': False, 'kickoff_time': '2026-08-21T14:00:00Z',
         'team_h_difficulty': 2, 'team_a_difficulty': 4},
        {'team_h': 2, 'team_a': 1, 'finished': False, 'kickoff_time': '2026-08-28T14:00:00Z',
         'team_h_difficulty': 4, 'team_a_difficulty': 2},
        {'team_h': 1, 'team_a': 2, 'finished': True, 'kickoff_time': '2026-08-01T14:00:00Z',
         'team_h_difficulty': 5, 'team_a_difficulty': 5},
    ])
    data = FeatureProcessor().calculate_fixture_difficulty(fixtures, teams)

    assert data[1]['next_opponent'] == 'MUN (H)'
    assert data[1]['opponent_team_id'] == 2
    assert data[1]['is_home'] is True
    assert data[1]['fixture_difficulty'] == pytest.approx((2 + 2) / 2), "finished games excluded"
    assert data[1]['next_kickoff_time'] is not None


def test_fixture_difficulty_handles_a_team_with_no_upcoming_games():
    teams = pd.DataFrame([{'id': 1, 'name': 'Arsenal', 'short_name': 'ARS', 'code': 3},
                          {'id': 2, 'name': 'Man Utd', 'short_name': 'MUN', 'code': 1}])
    fixtures = pd.DataFrame([
        {'team_h': 1, 'team_a': 2, 'finished': True, 'kickoff_time': '2026-08-01T14:00:00Z',
         'team_h_difficulty': 3, 'team_a_difficulty': 3}])
    data = FeatureProcessor().calculate_fixture_difficulty(fixtures, teams)
    # Must not leak the previous team's values or raise.
    assert data[1]['fixture_difficulty'] == 3
    assert data[1]['next_opponent'] == '-'
    assert data[1]['opponent_team_id'] == 0
    assert data[1]['next_kickoff_time'] is None


# ---------------------------------------------------------------- fallback
def _minimal_infer_frame():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'web_name': ['A', 'B', 'C'],
        'ep_next': ['4.5', '2.0', '0.0'],
        'minutes_prob': [1.0, 0.5, 1.0],
        'total_points': [40, 20, 0],
    })


def test_emergency_heuristic_is_loud_and_still_usable():
    p = PointsPredictor()
    out = p._emergency_heuristic(_minimal_infer_frame(), reason='cache missing')

    assert p.prediction_mode == 'fallback'
    assert p.prediction_warnings and 'cache missing' in p.prediction_warnings[0]
    for col in ['predicted_points', 'captaincy_score', 'projected_minutes', 'odds_confidence']:
        assert col in out.columns
    assert (out['predicted_points'] >= 0).all()


def test_emergency_heuristic_uses_ep_next_as_a_cold_start_prior():
    """Pre-season there is no history at all; FPL's own ep_next is the only signal."""
    p = PointsPredictor()
    out = p._emergency_heuristic(_minimal_infer_frame(), reason='preseason')
    # Ordering must follow ep_next, and the availability haircut must apply.
    assert out.loc[0, 'predicted_points'] == pytest.approx(4.5)
    assert out.loc[1, 'predicted_points'] == pytest.approx(2.0 * 0.5)


def test_emergency_heuristic_does_not_mutate_input():
    p = PointsPredictor()
    df = _minimal_infer_frame()
    before = df.copy()
    p._emergency_heuristic(df, reason='x')
    pd.testing.assert_frame_equal(df, before)
