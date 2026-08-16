"""Season identity, canonical vocabularies and shirt resolution."""
import pytest

from src.utils.season import (
    get_season_label, get_current_gw, get_next_gw, is_preseason,
    team_id_to_name, team_id_to_code, shirt_url, player_photo_url,
    canon_team, canon_position,
)


def test_season_label_from_august_deadline(bootstrap):
    assert get_season_label(bootstrap) == '2026-27'


def test_season_label_handles_missing_data():
    assert get_season_label(None) == 'unknown'
    assert get_season_label({'events': []}) == 'unknown'


def test_current_and_next_gw(bootstrap):
    assert get_current_gw(bootstrap) == 2
    assert get_next_gw(bootstrap) == 3
    assert is_preseason(bootstrap) is False


def test_preseason_current_gw_is_zero_not_last_season(preseason_bootstrap):
    """Regression: this used to fall back to parsing cache filenames and return 37."""
    assert get_current_gw(preseason_bootstrap) == 0
    assert get_next_gw(preseason_bootstrap) == 1
    assert is_preseason(preseason_bootstrap) is True


def test_season_over_returns_last_finished():
    events = [{'id': i, 'deadline_time': '2026-08-21T17:30:00Z', 'finished': True}
              for i in range(1, 39)]
    assert get_current_gw({'events': events}) == 38
    assert get_next_gw({'events': events}) == 1  # nothing next; safe default


@pytest.mark.parametrize('raw,expected', [
    ('GKP', 'GK'), ('GK', 'GK'), ('gkp', 'GK'),
    ('DEF', 'DEF'), ('MID', 'MID'), ('FWD', 'FWD'),
])
def test_position_canonicalisation(raw, expected):
    """FPL says GKP, vaastav says GK. They must collapse or the categorical splits."""
    assert canon_position(raw) == expected


@pytest.mark.parametrize('raw,expected', [
    ('Man United', 'Man Utd'),
    ('Manchester United', 'Man Utd'),
    ('Tottenham', 'Spurs'),
    ('Sheffield United', 'Sheffield Utd'),
    ('Arsenal', 'Arsenal'),      # already canonical
    ('Coventry', 'Coventry City'),
])
def test_team_canonicalisation(raw, expected):
    assert canon_team(raw) == expected


def test_team_maps(bootstrap):
    assert team_id_to_name(bootstrap)[3] == 'Coventry City'
    assert team_id_to_code(bootstrap)[3] == 9


def test_shirt_url_uses_bootstrap_code():
    """Regression: three hardcoded shirt tables went stale on promotion/relegation."""
    assert shirt_url(9).endswith('shirt_9-110.webp')
    assert 'shirt_0' in shirt_url(0)
    assert 'shirt_0' in shirt_url(None)


def test_player_photo_url():
    assert player_photo_url('12345.jpg').endswith('p12345.png')
    assert player_photo_url('p12345') is not None
    assert player_photo_url('') is None
    assert player_photo_url(None) is None
