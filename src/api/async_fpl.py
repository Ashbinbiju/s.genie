import aiohttp
import asyncio
import json
import os
import sys
import time

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.season import load_bootstrap, get_season_label, get_current_gw


# A snapshot older than this is refetched in full. get_current_gw() returns the most
# recently STARTED gameweek, so the file for GW N is written the moment GW N's deadline
# passes -- potentially before a single match has been played. Without an age bound that
# empty-looking snapshot stays frozen until the gameweek number changes, and every
# rolling feature silently misses the results it was supposed to learn from.
MAX_CACHE_AGE_HOURS = 6


def _read_cache(path):
    """Parsed cache, or None if it is absent or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def cache_filename(season, gw):
    """
    Cache files are stamped with the SEASON as well as the gameweek.

    FPL reassigns element ids every season, so a cache from a previous season will
    join cleanly onto this season's players and silently attribute every player's
    history to a different person. Putting the season in the filename makes a
    cross-season load impossible by construction.
    """
    return f"element_summary_{season}_gw_{gw}.json"


class AsyncFPLClient:
    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    async def fetch_summary(self, session, player_id, sem):
        async with sem:
            url = f"{self.BASE_URL}/element-summary/{player_id}/"
            try:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return player_id, data
            except Exception as e:
                print(f"Error fetching {player_id}: {e}")
                return player_id, None

    async def fetch_summaries(self, player_ids):
        """
        Fetch these ids concurrently. Returns {str(pid): payload} for successes only.

        Shared by the full cache build and the incremental top-up so the two cannot
        drift apart in concurrency limits or timeout behaviour.
        """
        sem = asyncio.Semaphore(20)  # Limit concurrent requests to avoid rate limits
        timeout = aiohttp.ClientTimeout(total=300, sock_connect=15, sock_read=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self.fetch_summary(session, pid, sem) for pid in player_ids]
            results = await asyncio.gather(*tasks)
        return {str(pid): data for pid, data in results if data is not None}

    async def get_all_summaries(self, player_ids, current_gw, season, max_failure_rate=0.05):
        """
        Fetch element-summary for a list of player IDs concurrently, caching the
        combined result per (season, gameweek).

        Raises RuntimeError if more than `max_failure_rate` of players fail. Silently
        dropping failures produces NaN rolling features that the model scores anyway,
        which is indistinguishable from success at the call site.
        """
        cache_file = os.path.join(self.cache_dir, cache_filename(season, current_gw))

        if os.path.exists(cache_file):
            print(f"Loading summaries from cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        print(f"Fetching {len(player_ids)} player summaries concurrently (season {season}, GW{current_gw})...")
        start_time = time.time()

        summaries = await self.fetch_summaries(player_ids)

        failed = len(player_ids) - len(summaries)
        if failed:
            rate = failed / max(len(player_ids), 1)
            msg = f"{failed}/{len(player_ids)} player summaries failed ({rate:.1%})"
            if rate > max_failure_rate:
                raise RuntimeError(
                    f"{msg} — refusing to write a partial cache. "
                    f"Re-run when the FPL API is healthy."
                )
            print(f"  WARNING: {msg} — those players will have missing features.")

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False)

        print(f"Fetched {len(summaries)} summaries in {time.time() - start_time:.2f} seconds.")
        print(f"  Cached to {cache_file}")
        return summaries


def fetch_summaries_sync(player_ids, current_gw, season):
    client = AsyncFPLClient()
    return asyncio.run(client.get_all_summaries(player_ids, current_gw, season))


def refresh_cache(static=None, cache_dir="data/cache"):
    """
    Ensure the element-summary cache for the CURRENT season+gameweek is present,
    COMPLETE and reasonably fresh.

    Safe and cheap to call on every dashboard run: when the cache is already good this
    returns without touching the network.

    Testing only "does the file exist" (as this used to) let two failure modes through
    silently, both of which degrade predictions without degrading anything visible:

      INCOMPLETE — FPL registers new players all season. Ids added after the snapshot
        was taken had no row in the cache at all, so their rolling features merged as
        NaN and the model scored them on nothing.
      STALE — see MAX_CACHE_AGE_HOURS: a snapshot taken before a gameweek's matches
        were played would otherwise stay frozen until the gameweek number changed.

    Pre-season (gw == 0) is fetched too. `history` is empty then, but `history_past`
    carries each player's PREVIOUS-SEASON totals — by far the best signal available
    before a ball is kicked, and the difference between a meaningful draft ranking and
    an arbitrary one.
    """
    static = static or load_bootstrap()
    if not static:
        print("refresh_cache: bootstrap_static.json not found; skipping.")
        return None

    season = get_season_label(static)
    gw = get_current_gw(static)
    if gw < 0:
        return None

    cache_file = os.path.join(cache_dir, cache_filename(season, gw))
    player_ids = [p['id'] for p in static['elements']]
    client = AsyncFPLClient(cache_dir=cache_dir)  # also ensures the directory exists

    cached = _read_cache(cache_file)
    if cached is None:
        # Absent, or present but corrupt. get_all_summaries short-circuits on an
        # existing file, so an unreadable one has to go before it will rebuild.
        if os.path.exists(cache_file):
            print(f"  Cache {cache_file} is unreadable; rebuilding it.")
            os.remove(cache_file)
        asyncio.run(client.get_all_summaries(player_ids, gw, season))
        return cache_file

    age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
    missing = [pid for pid in player_ids if str(pid) not in cached]

    if age_hours > MAX_CACHE_AGE_HOURS:
        # Refetch everyone: staleness affects the rows that are present, not just the
        # absent ones, so topping up the gaps alone would not fix it.
        reason, refetch = f"{age_hours:.1f}h old", player_ids
    elif missing:
        reason, refetch = f"{len(missing)} player(s) absent", missing
    else:
        return cache_file

    print(f"Refreshing element-summary cache ({reason}): fetching {len(refetch)} player(s)...")
    fetched = asyncio.run(client.fetch_summaries(refetch))
    if not fetched:
        # Never let a failed refresh destroy a cache that still works.
        print("  Refresh fetched nothing; keeping the existing cache.")
        return cache_file

    cached.update(fetched)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cached, f, ensure_ascii=False)
    print(f"  Cache now holds {len(cached)} players ({len(fetched)} refreshed).")
    return cache_file


if __name__ == "__main__":
    from src.api.fpl import FPLClient

    fpl = FPLClient()
    static = fpl.get_bootstrap_static() or load_bootstrap()

    season = get_season_label(static)
    gw = get_current_gw(static)
    print(f"Season: {season} | current GW: {gw}")
    if gw < 1:
        print("Pre-season: fetching previous-season totals (history_past) as a prior.")

    player_ids = [p['id'] for p in static['elements']]
    fetch_summaries_sync(player_ids, gw, season)
