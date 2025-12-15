import sys
import os
sys.path.append(os.getcwd())
try:
    from src.api.fpl import FPLClient
    client = FPLClient()
    fts = client.calculate_free_transfers(5985256, 17)
    print(f"FT_COUNT:{fts}")
except Exception as e:
    print(e)
