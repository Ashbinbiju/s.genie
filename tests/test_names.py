"""
Cross-source player-name matching.

Regression: normalisation stripped characters outside [a-z] instead of folding them, so
FPL's ASCII spelling and Understat's native spelling produced different keys and the
player was dropped from the join with zero xG/xA.
"""
import pandas as pd
import pytest

from src.utils.names import fold_accents, normalize_player_name, normalize_name_series


@pytest.mark.parametrize('understat,fpl', [
    ('Ødegaard', 'Odegaard'),
    ('Martin Ødegaard', 'Martin Odegaard'),
    ('Guimarães', 'Guimaraes'),
    ('Højlund', 'Hojlund'),
    ('Sánchez', 'Sanchez'),
    ('Güéhi', 'Guehi'),
    ('Şahin', 'Sahin'),
    ('Łukasz', 'Lukasz'),
    ('Đorđević', 'Dordevic'),
    ('Sørloth', 'Sorloth'),
    ('Doğan', 'Dogan'),
    ('Nørgaard', 'Norgaard'),
])
def test_native_and_ascii_spellings_produce_the_same_key(understat, fpl):
    assert normalize_player_name(understat) == normalize_player_name(fpl)


def test_stroked_letters_are_folded_not_deleted():
    """The exact failure: Ø must become 'o', not vanish."""
    assert normalize_player_name('Ødegaard') == 'odegaard'
    assert normalize_player_name('Ødegaard') != 'degaard'


@pytest.mark.parametrize('text,expected', [
    ('ß', 'ss'),
    ('Æ', 'ae'),
    ('Œ', 'oe'),
    ('Þ', 'th'),
])
def test_ligatures_expand(text, expected):
    assert normalize_player_name(text) == expected


def test_punctuation_and_spacing_are_removed():
    assert normalize_player_name("O'Reilly") == 'oreilly'
    assert normalize_player_name('Alexander-Arnold') == 'alexanderarnold'
    assert normalize_player_name('  De  Bruyne ') == 'debruyne'


def test_digits_are_removed():
    assert normalize_player_name('Vinicius Jr.2') == 'viniciusjr'


def test_handles_none_and_empty():
    assert normalize_player_name(None) == ''
    assert normalize_player_name('') == ''
    assert fold_accents(None) == ''


def test_distinct_players_do_not_collide():
    assert normalize_player_name('Silva') != normalize_player_name('Silvas')
    assert normalize_player_name('Reguilon') != normalize_player_name('Reguilonn')


def test_series_helper_matches_scalar():
    s = pd.Series(['Ødegaard', 'Guimarães', None])
    out = normalize_name_series(s)
    assert list(out) == ['odegaard', 'guimaraes', '']


def test_missing_names_never_become_the_literal_key_nan():
    """
    str(NaN) is 'nan'. Left unguarded, every nameless row shares that key and the merge
    joins unrelated players to each other, duplicating rows.
    """
    import numpy as np
    for missing in (None, float('nan'), np.nan):
        assert normalize_player_name(missing) == ''
    out = normalize_name_series(pd.Series([None, np.nan, 'Salah']))
    assert 'nan' not in list(out)


def test_realistic_join_matches_every_accented_player():
    """The join as processor performs it, over names that previously all failed."""
    fpl = pd.DataFrame({'web_name': ['Odegaard', 'Hojlund', 'Guimaraes', 'Sorloth']})
    understat = pd.DataFrame({'player_name': ['Martin Ødegaard', 'Rasmus Højlund',
                                              'Bruno Guimarães', 'Alexander Sørloth'],
                              'xG': [5.5, 4.4, 2.1, 6.0]})
    # Understat carries full names; match on the surname token as the pipeline's
    # web_name is a surname.
    fpl['key'] = normalize_name_series(fpl['web_name'])
    understat['key'] = normalize_name_series(understat['player_name'].str.split().str[-1])

    merged = fpl.merge(understat, on='key', how='left')
    assert merged['xG'].notna().all(), merged[['web_name', 'xG']].to_dict('records')
