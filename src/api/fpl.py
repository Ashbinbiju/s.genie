import requests
import json
import os

# Every outbound call is bounded. Without a timeout a hung upstream socket blocks the
# Streamlit worker forever with no way to recover.
DEFAULT_TIMEOUT = 20

# Transfers made while these chips are active do not consume free transfers, and your
# banked FT count carries across them untouched.
FREE_TRANSFER_CHIPS = {"wildcard", "freehit"}

MAX_FREE_TRANSFERS = 5
# GW1 has no free-transfer bank at all — transfers are unlimited until the GW1 deadline.
# A full squad is the real ceiling on transfers in a single gameweek, so reporting 15
# prices every GW1 transfer as free, which is exactly right.
GW1_UNLIMITED_TRANSFERS = 15


class FPLClient:
    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _get(self, endpoint, timeout=DEFAULT_TIMEOUT):
        """Helper to make GET requests."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def get_bootstrap_static(self):
        """Fetches general data: players, teams, events (gameweeks)."""
        data = self._get("bootstrap-static/")
        if data:
            self._save_json(data, "bootstrap_static.json")
        return data

    def get_fixtures(self):
        """Fetches all fixtures."""
        data = self._get("fixtures/")
        if data:
            self._save_json(data, "fixtures.json")
        return data

    def get_gameweek_live(self, gw):
        """Fetches live stats for a specific gameweek."""
        data = self._get(f"event/{gw}/live/")
        if data:
            self._save_json(data, f"gw_{gw}_live.json")
        return data

    def get_player_summary(self, player_id):
        """Fetches detailed history and fixtures for a player."""
        return self._get(f"element-summary/{player_id}/")

    def get_transfers(self, team_id):
        """Fetches transfer history."""
        return self._get(f"entry/{team_id}/transfers/")

    def get_history(self, team_id):
        """Fetches history including past performance and chips used."""
        return self._get(f"entry/{team_id}/history/")

    def get_entry(self, team_id):
        """
        Fetches a manager's entry (team) record, or None if it does not exist.

        Entry ids are issued PER SEASON — last season's id returns HTTP 404 — so this
        doubles as a validity check for a team id the user typed or that was saved from
        a previous season.
        """
        return self._get(f"entry/{team_id}/")

    def get_league_standings(self, league_id):
        """
        Fetches standings for a classic league.

        Returns None on failure — note that leagues are per-season, so an id from a
        previous season returns HTTP 404 rather than an empty league.
        """
        return self._get(
            f"leagues-classic/{league_id}/standings/?page_new_entries=1&page_standings=1&phase=1"
        )

    def get_league_members(self, league_id):
        """
        Returns {"Manager Name (Team Name)": entry_id} for every member of a league.

        Reads BOTH collections in the payload. `standings.results` is empty until the
        first gameweek has been scored; until then every member sits in
        `new_entries.results` under a different schema (split first/last name). Reading
        only standings — as this used to — makes a populated league look empty for the
        whole of pre-season.
        """
        payload = self.get_league_standings(league_id)
        if not payload:
            return {}

        members = {}

        # Ranked members first, so the dropdown keeps league order once it exists.
        for r in (payload.get('standings') or {}).get('results', []):
            members[f"{r['player_name']} ({r['entry_name']})"] = r['entry']

        for r in (payload.get('new_entries') or {}).get('results', []):
            name = f"{r.get('player_first_name', '')} {r.get('player_last_name', '')}".strip()
            label = f"{name} ({r['entry_name']})" if name else r['entry_name']
            # setdefault: a member appearing in both collections keeps the ranked entry.
            if r['entry'] not in members.values():
                members.setdefault(label, r['entry'])

        return members

    # ------------------------------------------------------------------
    # Chips
    # ------------------------------------------------------------------
    def get_chip_gws(self, team_id, history=None):
        """
        Returns {chip_name: [gw, ...]} for every chip the manager has played.

        A manager may play each chip twice per season (once either side of the GW20
        restoration boundary), so every value is a list.
        """
        history = history if history is not None else self.get_history(team_id)
        chips = {}
        if history and 'chips' in history:
            for chip in history['chips']:
                chips.setdefault(chip['name'], []).append(chip['event'])
        return chips

    def get_freehit_gws(self, team_id, history=None):
        """Set of GWs in which Free Hit was played. See get_team_picks."""
        return set(self.get_chip_gws(team_id, history).get('freehit', []))

    # ------------------------------------------------------------------
    # Free transfers
    # ------------------------------------------------------------------
    def calculate_free_transfers(self, team_id, current_gw, history=None):
        """
        Calculates available free transfers for the upcoming current_gw.

        Rules (2024/25 onward):
        - GW1 is unlimited; the first free transfer is credited for GW2.
        - Accumulate up to 5 FTs.
        - Deduct transfers made. If < 0, reset to 0 (hits taken), then add 1 for next week.
        - Transfers made under Wildcard or Free Hit are FREE: they neither consume
          nor reset your banked FTs. Counting them (as this used to) drives the
          balance to 0 after any wildcard and produces wrong hit-cost advice.
        """
        # Unlimited until the GW1 deadline, so there is nothing to count or bank yet.
        if current_gw <= 1:
            return GW1_UNLIMITED_TRANSFERS

        transfers = self.get_transfers(team_id)
        if transfers is None:
            return 1  # Default fallback

        chip_gws = self.get_chip_gws(team_id, history)
        exempt_gws = set()
        for chip_name in FREE_TRANSFER_CHIPS:
            exempt_gws.update(chip_gws.get(chip_name, []))

        tx_counts = {}
        for t in transfers:
            ev = t['event']
            tx_counts[ev] = tx_counts.get(ev, 0) + 1

        # GW1 grants no bankable FT and consumes none, because its transfers are
        # unlimited and free. Seeding the bank at 1 here (as this used to) inflated the
        # count by one every week of the season -- it reported 2 FTs going into GW2 when
        # FPL gives 1 -- so the optimizer priced a real -4 hit as free.
        exempt_gws.add(1)
        available_ft = 0

        for g in range(1, current_gw):
            # A chip week costs nothing and leaves the bank untouched.
            if g not in exempt_gws:
                available_ft -= tx_counts.get(g, 0)
                if available_ft < 0:
                    available_ft = 0

            available_ft = min(MAX_FREE_TRANSFERS, available_ft + 1)

        return available_ft

    # ------------------------------------------------------------------
    # Squad picks
    # ------------------------------------------------------------------
    def get_team_picks(self, team_id, gw, freehit_gws=None):
        """
        Fetches a team's most recent permanent squad, searching backwards from gw-1.

        Skips any GW in `freehit_gws`, because for those the API returns the temporary
        Free Hit squad rather than the permanent one — which would otherwise leak into
        the next week's recommendations. A manager can play two FH chips per season,
        so this accepts a set.

        Returns None before the first gameweek has been played (no squad exists yet);
        callers must handle that by building a squad from scratch rather than erroring.
        """
        freehit_gws = freehit_gws or set()
        start_gw = gw - 1

        if start_gw < 1:
            print(f"No completed gameweek before GW{gw} — no squad history exists yet.")
            return None

        for g in range(start_gw, max(0, start_gw - 6), -1):
            if g < 1:
                break
            if g in freehit_gws:
                print(f"Skipping GW{g} (Free Hit was played) — fetching permanent squad from earlier GW.")
                continue
            data = self._get(f"entry/{team_id}/event/{g}/picks/")
            if data:
                print(f"Loaded picks from GW{g}")
                return data

        print(f"Could not find any picks history (checked GW{start_gw} backwards)")
        return None

    def _save_json(self, data, filename):
        """Saves data to local JSON file for inspection/debugging."""
        path = os.path.join(self.data_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved {filename}")


if __name__ == "__main__":
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.utils.season import get_season_label, get_current_gw, get_next_gw

    client = FPLClient()
    print("Fetching static data...")
    static = client.get_bootstrap_static()
    print(f"Fetched {len(static['elements'])} players.")
    print(f"Season: {get_season_label(static)} | current GW: {get_current_gw(static)} | next GW: {get_next_gw(static)}")
    print("Fetching fixtures...")
    fixtures = client.get_fixtures()
    print(f"Fetched {len(fixtures)} fixtures.")
