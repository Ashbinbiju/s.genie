"""Bulk summary fetching: partial-failure handling and cache reuse."""
import asyncio
import json

import pytest

from src.api.async_fpl import AsyncFPLClient, cache_filename, refresh_cache


class FakeClient(AsyncFPLClient):
    """Replaces the network with a scripted set of successes/failures."""

    def __init__(self, cache_dir, fail_ids=()):
        super().__init__(cache_dir=cache_dir)
        self.fail_ids = set(fail_ids)
        self.calls = 0

    async def fetch_summary(self, session, player_id, sem):
        self.calls += 1
        if player_id in self.fail_ids:
            return player_id, None
        return player_id, {'history': [{'round': 1, 'minutes': 90, 'total_points': 3}]}


def run(coro):
    return asyncio.run(coro)


def test_writes_a_season_stamped_cache(tmp_path):
    c = FakeClient(str(tmp_path))
    run(c.get_all_summaries([1, 2, 3], current_gw=4, season='2026-27'))
    assert (tmp_path / cache_filename('2026-27', 4)).exists()


def test_cached_result_is_reused_without_refetching(tmp_path):
    c = FakeClient(str(tmp_path))
    run(c.get_all_summaries([1, 2, 3], 4, '2026-27'))
    first_calls = c.calls

    again = run(c.get_all_summaries([1, 2, 3], 4, '2026-27'))
    assert c.calls == first_calls, "a present cache must not trigger network calls"
    assert set(again) == {'1', '2', '3'}


def test_small_failure_rate_warns_but_succeeds(tmp_path, capsys):
    ids = list(range(1, 101))
    c = FakeClient(str(tmp_path), fail_ids={7})       # 1%
    out = run(c.get_all_summaries(ids, 4, '2026-27'))
    assert len(out) == 99
    assert 'WARNING' in capsys.readouterr().out


def test_large_failure_rate_raises_instead_of_writing_a_partial_cache(tmp_path):
    """
    Regression: failures were dropped silently. Those players then left-merged as NaN
    across every rolling feature and were scored anyway — indistinguishable from success.
    """
    ids = list(range(1, 101))
    c = FakeClient(str(tmp_path), fail_ids=set(range(1, 31)))   # 30%
    with pytest.raises(RuntimeError, match='partial cache'):
        run(c.get_all_summaries(ids, 4, '2026-27'))
    assert not (tmp_path / cache_filename('2026-27', 4)).exists()


def test_refresh_cache_fetches_in_preseason(preseason_bootstrap, tmp_path, monkeypatch):
    """
    Pre-season IS fetched: `history` is empty but `history_past` carries previous-season
    totals, which is the only real signal available for a GW1 draft.
    """
    calls = {}

    def fake_run(coro):
        calls['ran'] = True
        coro.close()
        return {}

    monkeypatch.setattr('src.api.async_fpl.asyncio.run', fake_run)
    path = refresh_cache(preseason_bootstrap, cache_dir=str(tmp_path))

    assert calls.get('ran'), "pre-season must still fetch history_past"
    assert path.endswith(cache_filename('2026-27', 0))


def test_refresh_cache_reuses_an_existing_file_without_fetching(preseason_bootstrap, tmp_path,
                                                                monkeypatch):
    (tmp_path / cache_filename('2026-27', 0)).write_text('{}', encoding='utf-8')
    monkeypatch.setattr('src.api.async_fpl.asyncio.run',
                        lambda c: pytest.fail("must not fetch when cached"))
    assert refresh_cache(preseason_bootstrap, cache_dir=str(tmp_path)) is not None


def test_refresh_cache_without_bootstrap_is_safe(monkeypatch):
    monkeypatch.setattr('src.api.async_fpl.load_bootstrap', lambda *a, **k: None)
    assert refresh_cache(None) is None
