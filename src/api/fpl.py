import requests
import pandas as pd
import json
import os
from datetime import datetime

class FPLClient:
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
    def _get(self, endpoint):
        """Helper to make GET requests."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers)
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
        data = self._get(f"element-summary/{player_id}/")
        # We don't save every single player summary individually to disk by default 
        # to avoid file clutter, but we could if needed.
        return data

    def get_transfers(self, team_id):
        """Fetches transfer history."""
        return self._get(f"entry/{team_id}/transfers/")

    def calculate_free_transfers(self, team_id, current_gw):
        """
        Calculates available free transfers for the upcoming current_gw.
        Based on 2024/25 Rules:
        - Start with 1 FT.
        - Accumulate up to 5 FTs.
        - Deduct transfers made. If < 0, reset to 0 (hits taken), then add 1 for next week.
        """
        transfers = self.get_transfers(team_id)
        if transfers is None:
            return 1 # Default fallback
            
        # Count transfers per gameweek
        tx_counts = {}
        for t in transfers:
            ev = t['event']
            tx_counts[ev] = tx_counts.get(ev, 0) + 1
            
        # Replay history
        available_ft = 1 # Start of season (GW1)
        
        # Iterate from GW1 up to the GW BEFORE the current one
        # Because we want to know what we have FOR current_gw
        for g in range(1, current_gw):
            used = tx_counts.get(g, 0)
            available_ft -= used
            
            # If we went negative (hits), we start next week with 1 (0 carried + 1 new)
            # If we stayed positive, we carry over + 1 new
            if available_ft < 0:
                available_ft = 0
                
            available_ft += 1
            available_ft = min(5, available_ft) # Cap at 5
            
        return available_ft

    def get_team_picks(self, team_id, gw):
        """Fetches a specific team's picks for a gameweek. Tries gw-1, then gw-2..."""
        # Try to find the latest available picks starting from gw-1
        start_gw = gw - 1
        for g in range(start_gw, max(0, start_gw - 5), -1):
            if g < 1: break
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
    client = FPLClient()
    print("Fetching static data...")
    static = client.get_bootstrap_static()
    print(f"Fetched {len(static['elements'])} players.")
    print("Fetching fixtures...")
    fixtures = client.get_fixtures()
    print(f"Fetched {len(fixtures)} fixtures.")
