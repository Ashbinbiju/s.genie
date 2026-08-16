import os
import requests
import pandas as pd

class VaastavClient:
    BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    
    def __init__(self, data_dir="data/raw/vaastav"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_season(self, season):
        """Downloads the merged_gw.csv for a specific season (e.g., '2023-24')"""
        url = f"{self.BASE_URL}/{season}/gws/merged_gw.csv"
        print(f"Downloading {season} data from {url}...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            output_path = os.path.join(self.data_dir, f"merged_gw_{season}.csv")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved {season} data to {output_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {season} data: {e}")
            return False

if __name__ == "__main__":
    client = VaastavClient()
    client.download_season("2023-24")
    client.download_season("2022-23")
    client.download_season("2021-22")
