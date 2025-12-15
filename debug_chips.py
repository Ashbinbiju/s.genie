import requests

def get_chip_history(team_id):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/history/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        
        print("--- Chips Used ---")
        if 'chips' in data:
            for chip in data['chips']:
                print(f"Chip: {chip['name']} | GW: {chip['event']} | Time: {chip['time']}")
        else:
            print("No 'chips' key found in history.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Using the ID seen in previous context: 5989967
    get_chip_history(5989967)
