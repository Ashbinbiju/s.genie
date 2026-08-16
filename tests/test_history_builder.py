"""Training-frame construction: opponent resolution, DGW collapse, odds join."""
import numpy as np
import pandas as pd
import pytest

from src.features.history_builder import HistoryBuilder


def vaastav_rows():
    """Two clubs, one fixture, both directions — the shape merged_gw.csv has."""
    return pd.DataFrame([
        {'fixture': 100, 'team_name': 'Arsenal', 'opponent_team': 2},
        {'fixture': 100, 'team_name': 'Arsenal', 'opponent_team': 2},
        {'fixture': 100, 'team_name': 'Man Utd', 'opponent_team': 1},
        {'fixture': 101, 'team_name': 'Arsenal', 'opponent_team': 3},
        {'fixture': 101, 'team_name': 'Chelsea', 'opponent_team': 1},
    ])


def test_opponent_name_map_resolves_ids_to_the_other_club():
    m = HistoryBuilder._opponent_name_map(vaastav_rows())
    assert m[2] == 'Man Utd'   # Arsenal's opponent id 2
    assert m[1] == 'Arsenal'   # Man Utd's / Chelsea's opponent id 1
    assert m[3] == 'Chelsea'


def test_opponent_name_map_ignores_malformed_fixtures():
    df = pd.DataFrame([
        {'fixture': 1, 'team_name': 'Arsenal', 'opponent_team': 2},   # only one club
        {'fixture': 2, 'team_name': 'Arsenal', 'opponent_team': 3},
        {'fixture': 2, 'team_name': 'Spurs', 'opponent_team': 1},
    ])
    m = HistoryBuilder._opponent_name_map(df)
    assert m == {3: 'Spurs', 1: 'Arsenal'}


def test_opponent_name_map_without_a_fixture_column_returns_empty():
    """This is the dangerous case — see the guard test below."""
    assert HistoryBuilder._opponent_name_map(pd.DataFrame({'team_name': ['A']})) == {}


def test_unmapped_opponents_are_never_silently_stringified_as_nan():
    """
    Regression guard: `df['opponent_team'].map({})` yields NaN, and a later
    `.astype(str).astype('category')` turns that into the literal category 'nan' —
    a silently corrupted vocabulary that trains and predicts without any error.
    """
    df = vaastav_rows()
    df['opponent_name'] = df['opponent_team'].map({})       # simulate total mapping failure
    resolved = HistoryBuilder._resolve_opponent_names(df)
    assert resolved.notna().all()
    assert 'nan' not in set(resolved.astype(str))


def test_resolve_opponent_names_uses_the_map_when_available():
    df = vaastav_rows()
    resolved = HistoryBuilder._resolve_opponent_names(df)
    assert list(resolved) == ['Man Utd', 'Man Utd', 'Arsenal', 'Chelsea', 'Arsenal']


# ---------------------------------------------------------------- odds join
def test_odds_join_matches_on_date_not_match_ordinal(tmp_path, monkeypatch):
    """
    Regression: odds were attached by numbering each club's matches 1..38 and assuming
    match N == GW N. One postponement shifted every later fixture's odds.
    """
    hb = HistoryBuilder(raw_dir=str(tmp_path), cache_dir=str(tmp_path),
                        processed_dir=str(tmp_path))

    df_all = pd.DataFrame({
        'season': ['2023-24'] * 3,
        'team_name': ['Arsenal'] * 3,
        'GW': [1, 2, 4],                    # GW3 blank -> ordinal join would misalign
        'kickoff_time': pd.to_datetime(
            ['2023-08-12T14:00Z', '2023-08-19T14:00Z', '2023-09-09T14:00Z'], utc=True),
        'position': ['MID'] * 3,
    })

    odds = pd.DataFrame({
        'date': ['12/08/2023', '19/08/2023', '09/09/2023'],
        'team_name': ['Arsenal'] * 3,
        'win_prob': [0.7, 0.6, 0.5], 'draw_prob': [0.2] * 3, 'loss_prob': [0.1, 0.2, 0.3],
        'team_implied_goals': [2.0, 1.8, 1.6], 'opponent_implied_goals': [1.0] * 3,
        'clean_sheet_prob': [0.4, 0.3, 0.2],
    })

    class FakeOdds:
        def __init__(self, *a, **k): pass
        def download_historical_odds(self): pass
        def load_historical_odds(self, season): return odds.copy()

    monkeypatch.setattr('src.api.odds.OddsClient', FakeOdds)
    out = hb._merge_odds(df_all.copy())

    assert len(out) == len(df_all), "the join must not duplicate rows"
    # GW4 must get the 09/09 odds, not the third-ordinal-match odds.
    assert out.loc[out['GW'] == 4, 'win_prob'].iloc[0] == pytest.approx(0.5)
    assert out.loc[out['GW'] == 1, 'win_prob'].iloc[0] == pytest.approx(0.7)


def test_odds_join_falls_back_to_defaults_for_unmatched_rows(tmp_path, monkeypatch):
    from src.api.odds import LEAGUE_DEFAULTS

    hb = HistoryBuilder(raw_dir=str(tmp_path), cache_dir=str(tmp_path),
                        processed_dir=str(tmp_path))
    df_all = pd.DataFrame({
        'season': ['2023-24'],
        'team_name': ['Arsenal'],
        'GW': [1],
        'kickoff_time': pd.to_datetime(['2023-08-12T14:00Z'], utc=True),
    })
    empty = pd.DataFrame(columns=['date', 'team_name', 'win_prob', 'draw_prob', 'loss_prob',
                                  'team_implied_goals', 'opponent_implied_goals',
                                  'clean_sheet_prob'])

    class FakeOdds:
        def __init__(self, *a, **k): pass
        def download_historical_odds(self): pass
        def load_historical_odds(self, season): return empty

    monkeypatch.setattr('src.api.odds.OddsClient', FakeOdds)
    out = hb._merge_odds(df_all.copy())
    assert out['win_prob'].iloc[0] == pytest.approx(LEAGUE_DEFAULTS['win_prob'])
    assert 'match_date' not in out.columns


def test_odds_join_never_duplicates_on_repeated_odds_rows(tmp_path, monkeypatch):
    """A club appearing twice for one date (data error) must not fan out the frame."""
    hb = HistoryBuilder(raw_dir=str(tmp_path), cache_dir=str(tmp_path),
                        processed_dir=str(tmp_path))
    df_all = pd.DataFrame({
        'season': ['2023-24'] * 2, 'team_name': ['Arsenal'] * 2, 'GW': [1, 1],
        'kickoff_time': pd.to_datetime(['2023-08-12T14:00Z'] * 2, utc=True),
    })
    dupes = pd.DataFrame({
        'date': ['12/08/2023', '12/08/2023'],
        'team_name': ['Arsenal', 'Arsenal'],
        'win_prob': [0.7, 0.6], 'draw_prob': [0.2, 0.2], 'loss_prob': [0.1, 0.2],
        'team_implied_goals': [2.0, 1.0], 'opponent_implied_goals': [1.0, 1.0],
        'clean_sheet_prob': [0.4, 0.3],
    })

    class FakeOdds:
        def __init__(self, *a, **k): pass
        def download_historical_odds(self): pass
        def load_historical_odds(self, season): return dupes

    monkeypatch.setattr('src.api.odds.OddsClient', FakeOdds)
    out = hb._merge_odds(df_all.copy())
    assert len(out) == 2
