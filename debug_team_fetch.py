import requests
import json

ids_to_check = [1, 5989967]
gw = 16
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

for team_id in ids_to_check:
    print(f"--- Checking Team {team_id} for GW{gw} ---")
    url_entry = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
    url_picks = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    
    try:
        r = requests.get(url_entry, headers=headers)
        print(f"Entry Status: {r.status_code}")
        if r.status_code != 200:
             print(f"Entry Error: {r.text[:100]}")
    except Exception as e: 
        print(f"Entry Exception: {e}")
    
    try:
        r = requests.get(url_picks, headers=headers)
        print(f"Picks Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            picks = data.get('picks', [])
            print(f"Found {len(picks)} picks.")
        else:
            print(f"Picks Error: {r.text[:100]}")
    except Exception as e: 
        print(f"Picks Exception: {e}")
