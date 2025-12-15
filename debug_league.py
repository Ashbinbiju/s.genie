import requests
import json

def get_league_details(league_id, page=1):
    url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_new_entries=1&page_standings={page}&phase=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        
        print(f"--- League: {data['league']['name']} ---")
        print("--- Top 5 Standings ---")
        for entry in data['standings']['results'][:5]:
            print(f"Rank: {entry['rank']} | Team: {entry['entry_name']} | Manager: {entry['player_name']} | ID: {entry['entry']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Using the ID provided by user: 1311994
    get_league_details(1311994)
