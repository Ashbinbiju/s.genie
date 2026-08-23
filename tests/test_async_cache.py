"""Bulk summary fetching: partial-failure handling and cache reuse."""
import asyncio
import json
import os
import time

import pytest

from src.api.async_fpl import (AsyncFPLClient, cache_filename, refresh_cache,
                               MAX_CACHE_AGE_HOURS)


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


# The preseason_bootstrap fixture carries element ids 10, 11 and 12.
ALL_IDS = ['10', '11', '12']
CACHE = cache_filename('2026-27', 0)


def write_cache(tmp_path, ids, age_hours=0.0):
    """A cache file holding exactly `ids`, optionally backdated."""
    path = tmp_path / CACHE
    path.write_text(
        json.dumps({str(i): {'history': [{'round': 1, 'minutes': 90}]} for i in ids}),
        encoding='utf-8')
    if age_hours:
        old = time.time() - age_hours * 3600
        os.utime(path, (old, old))
    return path


def record_fetches(monkeypatch, requested, succeed=True):
    """Replace the network with a recorder. Returns the list it appends ids to."""
    async def fake_fetch_summaries(self, player_ids):
        requested.extend(player_ids)
        if not succeed:
            return {}
        return {str(p): {'history': [{'round': 1, 'minutes': 90}]} for p in player_ids}

    monkeypatch.setattr(AsyncFPLClient, 'fetch_summaries', fake_fetch_summaries)
    return requested


def test_refresh_cache_reuses_a_complete_fresh_file_without_fetching(preseason_bootstrap,
                                                                     tmp_path, monkeypatch):
    write_cache(tmp_path, ALL_IDS)
    monkeypatch.setattr('src.api.async_fpl.asyncio.run',
                        lambda c: pytest.fail("must not fetch when the cache is good"))
    assert refresh_cache(preseason_bootstrap, cache_dir=str(tmp_path)) is not None


def test_refresh_cache_tops_up_players_added_since_the_snapshot(preseason_bootstrap,
                                                                tmp_path, monkeypatch):
    """
    Regression: FPL registers new players all season, and the only check was "does the
    file exist". Ids added after the snapshot had no cache row, so their rolling
    features left-merged as NaN and the model scored them on nothing.
    """
    write_cache(tmp_path, ['10', '11'])
    requested = record_fetches(monkeypatch, [])

    refresh_cache(preseason_bootstrap, cache_dir=str(tmp_path))

    assert requested == [12], "only the absent player should be fetched"
    merged = json.loads((tmp_path / CACHE).read_text(encoding='utf-8'))
    assert sorted(merged) == ALL_IDS


def test_refresh_cache_refetches_a_stale_snapshot(preseason_bootstrap, tmp_path, monkeypatch):
    """
    A snapshot written before a gameweek's matches were played must not stay frozen
    until the gameweek number changes — staleness affects the rows that ARE present,
    so every player is refetched, not just the gaps.
    """
    write_cache(tmp_path, ALL_IDS, age_hours=MAX_CACHE_AGE_HOURS + 1)
    requested = record_fetches(monkeypatch, [])

    refresh_cache(preseason_bootstrap, cache_dir=str(tmp_path))

    assert sorted(requested) == [10, 11, 12]


def test_refresh_cache_rebuilds_an_unreadable_file(preseason_bootstrap, tmp_path, monkeypatch):
    (tmp_path / CACHE).write_text('{ truncated', encoding='utf-8')
    requested = record_fetches(monkeypatch, [])

    refresh_cache(preseason_bootstrap, cache_dir=str(tmp_path))

    assert sorted(requested) == [10, 11, 12]
    assert sorted(json.loads((tmp_path / CACHE).read_text(encoding='utf-8'))) == ALL_IDS


def test_a_failed_refresh_keeps_the_working_cache(preseason_bootstrap, tmp_path, monkeypatch):
    """A dead API must not cost us the cache we already had."""
    write_cache(tmp_path, ['10', '11'])
    record_fetches(monkeypatch, [], succeed=False)

    refresh_cache(preseason_bootstrap, cache_dir=str(tmp_path))

    assert sorted(json.loads((tmp_path / CACHE).read_text(encoding='utf-8'))) == ['10', '11']


def test_refresh_cache_without_bootstrap_is_safe(monkeypatch):
    monkeypatch.setattr('src.api.async_fpl.load_bootstrap', lambda *a, **k: None)
    assert refresh_cache(None) is None
