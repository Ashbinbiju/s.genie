from src.api.fpl import FPLClient
import json

client = FPLClient()
team_id = 5989967

print(f"Fetching data for team {team_id}...")

# 1. Entry Summary
entry = client._get(f"entry/{team_id}/")
if entry:
    print("\n--- Entry Summary ---")
    keys = ['last_deadline_bank', 'last_deadline_value', 'last_deadline_total_transfers', 'summary_overall_points', 'summary_overall_rank']
    for k in keys:
        print(f"{k}: {entry.get(k)}")

# 2. History
history = client._get(f"entry/{team_id}/history/")
if history:
    print("\n--- History (Last 3 GWs) ---")
    current = history.get('current', [])
    for gw in current[-3:]:
        print(gw)

# 3. Check for 'transfers' endpoint specifically if it exists?
# Usually just 'transfers' is the history of all transfers
transfers = client._get(f"entry/{team_id}/transfers/")
if transfers:
    print(f"\n--- Transfers (Last 5) ---: {len(transfers)}")
    for t in transfers[-5:]:
        print(t)
