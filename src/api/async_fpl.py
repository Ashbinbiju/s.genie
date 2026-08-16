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

        sem = asyncio.Semaphore(20)  # Limit concurrent requests to avoid rate limits

        timeout = aiohttp.ClientTimeout(total=300, sock_connect=15, sock_read=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self.fetch_summary(session, pid, sem) for pid in player_ids]
            results = await asyncio.gather(*tasks)

        summaries = {str(pid): data for pid, data in results if data is not None}

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


def refresh_cache(static=None):
    """
    Ensure the element-summary cache for the CURRENT season+gameweek exists.

    Safe and cheap to call on every dashboard run: if the correct cache file is
    already on disk this returns immediately without touching the network.
    """
    static = static or load_bootstrap()
    if not static:
        print("refresh_cache: bootstrap_static.json not found; skipping.")
        return None

    season = get_season_label(static)
    gw = get_current_gw(static)

    if gw < 1:
        print(f"refresh_cache: pre-season ({season}), no gameweek history exists yet.")
        return None

    cache_file = os.path.join("data", "cache", cache_filename(season, gw))
    if os.path.exists(cache_file):
        return cache_file

    player_ids = [p['id'] for p in static['elements']]
    fetch_summaries_sync(player_ids, gw, season)
    return cache_file


if __name__ == "__main__":
    from src.api.fpl import FPLClient

    fpl = FPLClient()
    static = fpl.get_bootstrap_static() or load_bootstrap()

    season = get_season_label(static)
    gw = get_current_gw(static)
    print(f"Season: {season} | current GW: {gw}")

    if gw < 1:
        print("Pre-season — no gameweek history to fetch yet.")
    else:
        player_ids = [p['id'] for p in static['elements']]
        fetch_summaries_sync(player_ids, gw, season)
