"""Free-transfer accounting, chip parsing and the GW1 picks guard."""
import pytest

from src.api.fpl import FPLClient, MAX_FREE_TRANSFERS, GW1_UNLIMITED_TRANSFERS


@pytest.fixture
def client(monkeypatch):
    c = FPLClient(data_dir='.')
    # Never touch the network in tests.
    monkeypatch.setattr(c, '_get', lambda *a, **k: pytest.fail("unexpected network call"))
    return c


def stub(client, monkeypatch, transfers, history):
    monkeypatch.setattr(client, 'get_transfers', lambda team_id: transfers)
    monkeypatch.setattr(client, 'get_history', lambda team_id: history)


def test_gw1_transfers_are_unlimited(client, monkeypatch):
    """GW1 has no FT bank — transfers are free and unlimited until its deadline."""
    stub(client, monkeypatch, [], {})
    assert client.calculate_free_transfers(1, 1) == GW1_UNLIMITED_TRANSFERS


def test_gw2_grants_exactly_one_free_transfer(client, monkeypatch):
    """
    Regression: the bank was seeded with a phantom GW1 free transfer, so this reported
    2 going into GW2 when FPL gives 1. Every week was inflated by one, and the solver
    priced the resulting -4 hit as free.
    """
    stub(client, monkeypatch, [], {})
    assert client.calculate_free_transfers(1, 2) == 1


def test_gw1_transfers_do_not_consume_the_bank(client, monkeypatch):
    """Unlimited GW1 transfers cost nothing, so GW2 still opens with its one FT."""
    stub(client, monkeypatch, [{'event': 1}] * 8, {})
    assert client.calculate_free_transfers(1, 2) == 1


def test_ft_accumulates_and_caps_at_five(client, monkeypatch):
    stub(client, monkeypatch, [], {})
    # One credited per gameweek from GW2, so the cap is first reached at GW6.
    assert client.calculate_free_transfers(1, 5) == 4
    assert client.calculate_free_transfers(1, 6) == MAX_FREE_TRANSFERS
    assert client.calculate_free_transfers(1, 20) == MAX_FREE_TRANSFERS


def test_ft_deducted_for_normal_transfers(client, monkeypatch):
    stub(client, monkeypatch, [{'event': 2}], {})
    # GW2 opens with 1, spends it -> 0, +1 = 1 going into GW3.
    assert client.calculate_free_transfers(1, 3) == 1


def test_hits_floor_at_zero_then_regain_one(client, monkeypatch):
    stub(client, monkeypatch, [{'event': 2}] * 11, {})
    # 11 transfers in GW2 wipes the bank; you start GW3 with exactly 1.
    assert client.calculate_free_transfers(1, 3) == 1


def test_wildcard_transfers_do_not_consume_free_transfers(client, monkeypatch):
    """Regression: wildcard transfers used to zero the bank and reset it to 1."""
    transfers = [{'event': 5}] * 11
    history = {'chips': [{'name': 'wildcard', 'event': 5}]}
    stub(client, monkeypatch, transfers, history)
    # Bank keeps accruing straight through the chip week.
    assert client.calculate_free_transfers(1, 6) == 5


def test_freehit_transfers_do_not_consume_free_transfers(client, monkeypatch):
    transfers = [{'event': 3}] * 9
    history = {'chips': [{'name': 'freehit', 'event': 3}]}
    stub(client, monkeypatch, transfers, history)
    # GW2 and GW3 each credit one; the Free Hit week spends none of it.
    assert client.calculate_free_transfers(1, 4) == 3


def test_same_transfers_without_chip_are_punished(client, monkeypatch):
    """Control for the two tests above: no chip means the bank IS consumed."""
    stub(client, monkeypatch, [{'event': 5}] * 11, {})
    assert client.calculate_free_transfers(1, 6) == 1


def test_ft_defaults_to_one_when_api_fails(client, monkeypatch):
    stub(client, monkeypatch, None, {})
    assert client.calculate_free_transfers(1, 10) == 1


def test_chip_gws_collects_both_uses(client, monkeypatch):
    history = {'chips': [{'name': 'freehit', 'event': 7},
                         {'name': 'freehit', 'event': 29},
                         {'name': 'wildcard', 'event': 3}]}
    monkeypatch.setattr(client, 'get_history', lambda team_id: history)
    assert client.get_freehit_gws(1) == {7, 29}
    assert client.get_chip_gws(1)['wildcard'] == [3]


def test_league_members_read_new_entries_before_gw1(monkeypatch):
    """
    Regression: `standings.results` is empty until the first gameweek is scored, so a
    league with members looked completely empty for the whole of pre-season.
    """
    payload = {
        'league': {'id': 1019782, 'name': 'RCFC League'},
        'standings': {'results': []},
        'new_entries': {'results': [
            {'entry': 4772552, 'entry_name': 'Alpha XI',
             'player_first_name': 'Ashbin', 'player_last_name': 'Biju'},
            {'entry': 4767294, 'entry_name': 'ASD66',
             'player_first_name': 'ASWIN', 'player_last_name': 'DEV D R'},
        ]},
    }
    c = FPLClient(data_dir='.')
    monkeypatch.setattr(c, 'get_league_standings', lambda lid: payload)

    members = c.get_league_members(1019782)
    assert members == {'Ashbin Biju (Alpha XI)': 4772552, 'ASWIN DEV D R (ASD66)': 4767294}


def test_league_members_prefer_ranked_standings(monkeypatch):
    """Once scoring starts, standings order wins and members are not duplicated."""
    payload = {
        'standings': {'results': [
            {'entry': 1, 'entry_name': 'Alpha XI', 'player_name': 'Ashbin Biju'},
        ]},
        'new_entries': {'results': [
            {'entry': 1, 'entry_name': 'Alpha XI',
             'player_first_name': 'Ashbin', 'player_last_name': 'Biju'},
            {'entry': 2, 'entry_name': 'Late Joiner',
             'player_first_name': 'New', 'player_last_name': 'Manager'},
        ]},
    }
    c = FPLClient(data_dir='.')
    monkeypatch.setattr(c, 'get_league_standings', lambda lid: payload)

    members = c.get_league_members(9)
    assert list(members.values()) == [1, 2], "no duplicates, standings first"
    assert 'Ashbin Biju (Alpha XI)' in members


def test_league_members_handle_missing_collections(monkeypatch):
    c = FPLClient(data_dir='.')
    monkeypatch.setattr(c, 'get_league_standings', lambda lid: None)
    assert c.get_league_members(1) == {}

    monkeypatch.setattr(c, 'get_league_standings', lambda lid: {'league': {}})
    assert c.get_league_members(1) == {}

    monkeypatch.setattr(c, 'get_league_standings',
                        lambda lid: {'standings': None, 'new_entries': None})
    assert c.get_league_members(1) == {}


def test_league_member_without_a_name_falls_back_to_team_name(monkeypatch):
    payload = {'standings': {'results': []},
               'new_entries': {'results': [{'entry': 5, 'entry_name': 'Anon FC'}]}}
    c = FPLClient(data_dir='.')
    monkeypatch.setattr(c, 'get_league_standings', lambda lid: payload)
    assert c.get_league_members(1) == {'Anon FC': 5}


def test_get_entry_validates_a_team_id(monkeypatch):
    """A stale id from a previous season 404s; get_entry surfaces that as None."""
    c = FPLClient(data_dir='.')
    monkeypatch.setattr(c, '_get', lambda ep, timeout=None:
                        {'name': 'Alpha XI'} if 'entry/4772552/' in ep else None)
    assert c.get_entry(4772552)['name'] == 'Alpha XI'
    assert c.get_entry(5989967) is None


def test_get_team_picks_returns_none_before_gw1(client):
    """Regression: gw=1 produced an empty range and dead-ended the whole app."""
    assert client.get_team_picks(1, 1) is None
    assert client.get_team_picks(1, 0) is None


def test_get_team_picks_skips_freehit_gameweeks(monkeypatch):
    c = FPLClient(data_dir='.')
    seen = []

    def fake_get(endpoint, timeout=None):
        seen.append(endpoint)
        # Only GW3 has a stored squad.
        return {'picks': [{'element': 1}]} if '/event/3/' in endpoint else None

    monkeypatch.setattr(c, '_get', fake_get)
    result = c.get_team_picks(99, 6, freehit_gws={5})

    assert result is not None
    assert not any('/event/5/' in e for e in seen), "must not request the Free Hit gameweek"
    assert any('/event/4/' in e for e in seen)
