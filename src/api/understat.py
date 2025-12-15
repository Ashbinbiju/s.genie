import requests
import re
import json
import pandas as pd
from datetime import datetime

class UnderstatClient:
    BASE_URL = "https://understat.com/league/EPL"
    
    def __init__(self, year=None):
        # If year is None, use current season start year (e.g., 2023 for 23/24)
        if year is None:
            now = datetime.now()
            # If we are in second half of year, it's the start of season. 
            # If first half, it's (year-1)
            self.year = now.year if now.month > 7 else now.year - 1
        else:
            self.year = year

    def get_player_stats(self):
        """Scrapes player data from the main league page."""
        url = f"{self.BASE_URL}/{self.year}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.text
            
            # Find the JSON data inside script tags
            match = re.search(r"var playersData\s*=\s*JSON\.parse\('(.*?)'\);", content)
            if match:
                # The data is hex (or unicode) encoded in the string sometimes, 
                # but usually simplest is json.loads of the captured group decoding unicode escapes
                raw_data = match.group(1)
                # Decode unicode escape sequences if necessary, but requests.text usually handles encoding.
                # The string inside is often like "\x7B\x22id..."
                decoded_data = raw_data.encode('utf-8').decode('unicode_escape')
                data = json.loads(decoded_data)
                
                df = pd.DataFrame(data)
                # Convert numeric columns
                numeric_cols = ['xG', 'xA', 'shots', 'goals', 'assists', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup', 'time']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            else:
                print("Could not find playersData in page content.")
                return None
        except Exception as e:
            print(f"Error fetching Understat data: {e}")
            return None

if __name__ == "__main__":
    client = UnderstatClient()
    print(f"Fetching Understat data for {client.year}...")
    df = client.get_player_stats()
    if df is not None:
        print(f"Fetched {len(df)} players.")
        print(df.head())
        df.to_csv("data/raw/understat_players.csv", index=False)
