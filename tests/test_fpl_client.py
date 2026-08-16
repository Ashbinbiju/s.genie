"""Free-transfer accounting, chip parsing and the GW1 picks guard."""
import pytest

from src.api.fpl import FPLClient, MAX_FREE_TRANSFERS


@pytest.fixture
def client(monkeypatch):
    c = FPLClient(data_dir='.')
    # Never touch the network in tests.
    monkeypatch.setattr(c, '_get', lambda *a, **k: pytest.fail("unexpected network call"))
    return c


def stub(client, monkeypatch, transfers, history):
    monkeypatch.setattr(client, 'get_transfers', lambda team_id: transfers)
    monkeypatch.setattr(client, 'get_history', lambda team_id: history)


def test_ft_starts_at_one(client, monkeypatch):
    stub(client, monkeypatch, [], {})
    assert client.calculate_free_transfers(1, 1) == 1


def test_ft_accumulates_and_caps_at_five(client, monkeypatch):
    stub(client, monkeypatch, [], {})
    assert client.calculate_free_transfers(1, 2) == 2
    assert client.calculate_free_transfers(1, 5) == 5
    assert client.calculate_free_transfers(1, 20) == MAX_FREE_TRANSFERS


def test_ft_deducted_for_normal_transfers(client, monkeypatch):
    stub(client, monkeypatch, [{'event': 2}], {})
    # GW1: 0 used -> 2 banked. GW2: 1 used -> 1, +1 = 2 going into GW3.
    assert client.calculate_free_transfers(1, 3) == 2


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
    assert client.calculate_free_transfers(1, 4) == 4


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
