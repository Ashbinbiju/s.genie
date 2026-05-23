import aiohttp
import asyncio
import json
import os
import time

class AsyncFPLClient:
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
    async def fetch_summary(self, session, player_id, sem):
        async with sem:
            url = f"{self.BASE_URL}/element-summary/{player_id}/"
            try:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return player_id, data
            except Exception as e:
                print(f"Error fetching {player_id}: {e}")
                return player_id, None

    async def get_all_summaries(self, player_ids, current_gw):
        """
        Fetches element-summary for a list of player IDs asynchronously.
        Caches the combined result per gameweek to avoid redundant fetches.
        """
        cache_file = os.path.join(self.cache_dir, f"element_summary_gw_{current_gw}.json")
        
        # Check cache
        if os.path.exists(cache_file):
            print(f"Loading summaries from cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        print(f"Fetching {len(player_ids)} player summaries concurrently...")
        start_time = time.time()
        
        sem = asyncio.Semaphore(20) # Limit concurrent requests to avoid rate limits
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_summary(session, pid, sem) for pid in player_ids]
            results = await asyncio.gather(*tasks)
            
        summaries = {str(pid): data for pid, data in results if data is not None}
        
        # Save to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False)
            
        print(f"Fetched {len(summaries)} summaries in {time.time() - start_time:.2f} seconds.")
        return summaries

def fetch_summaries_sync(player_ids, current_gw):
    client = AsyncFPLClient()
    return asyncio.run(client.get_all_summaries(player_ids, current_gw))

if __name__ == "__main__":
    # Test script
    from src.api.fpl import FPLClient
    fpl = FPLClient()
    static = fpl.get_bootstrap_static()
    player_ids = [p['id'] for p in static['elements']]
    
    # Determine current GW
    current_gw = 1
    for ev in static['events']:
        if ev['is_current']:
            current_gw = ev['id']
            break
            
    print(f"Current GW: {current_gw}")
    fetch_summaries_sync(player_ids, current_gw)
