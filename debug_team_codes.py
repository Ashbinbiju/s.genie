import pandas as pd
import json
import os

def check_data():
    # 1. Check Raw Teams Data
    with open('data/raw/bootstrap_static.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    teams = data['teams']
    print("\n--- Team Codes (Raw) ---")
    print(f"{'ID':<5} {'Name':<20} {'Code':<10}")
    team_codes = {}
    for t in teams:
        print(f"{t['id']:<5} {t['name']:<20} {t['code']:<10}")
        team_codes[t['id']] = t['code']
        
    # 2. Check Processed Features
    print("\n--- Processed Player Features ---")
    try:
        df = pd.read_parquet('data/processed/player_features.parquet')
        
        # Check specific problem players
        # Woltemade (714 or similar), Alderete
        # Let's search by name loosely
        targets = df[df['web_name'].str.lower().str.contains('woltemade|alderete|muñoz|chalobah', na=False)]
        
        print(f"{'Web Name':<15} {'TeamID':<8} {'TeamCode':<10} {'Photo':<15}")
        for _, row in targets.iterrows():
            # Check if team_code matches raw
            raw_code = team_codes.get(row['team'], 'UNKNOWN')
            print(f"{row['web_name']:<15} {row['team']:<8} {row['team_code']:<10} (Expected: {raw_code}) {row['photo']:<15}")

            
    except Exception as e:
        print(f"Error reading parquet: {e}")

if __name__ == "__main__":
    check_data()
