import sys
import os
sys.path.append(os.getcwd())
from src.api.fpl import FPLClient

def debug_ft(team_id, gw):
    client = FPLClient()
    fts = client.calculate_free_transfers(team_id, gw)
    print(f"--- Debug FT for Team {team_id} GW{gw} ---")
    print(f"Calculated Free Transfers: {fts}")
    
    # Also fetch transfers to see history
    tr = client.get_transfers(team_id)
    if tr:
        print(f"Total Transfers Made So Far: {len(tr)}")
        # Show last few
        for t in tr[-5:]:
            print(f"GW{t['event']}: IN {t['element_in']} OUT {t['element_out']}")

if __name__ == "__main__":
    debug_ft(5985256, 17)
