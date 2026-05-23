import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is on sys.path for both standalone and package execution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

class HistoryBuilder:
    def __init__(self, raw_dir="data/raw", cache_dir="data/cache", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.rolling_cols = [
            'minutes', 'total_points', 'expected_goals', 'expected_assists', 
            'expected_goal_involvements', 'expected_goals_conceded',
            'bps', 'influence', 'creativity', 'threat', 'starts'
        ]
        
    def _load_vaastav_season(self, season):
        path = os.path.join(self.raw_dir, "vaastav", f"merged_gw_{season}.csv")
        if not os.path.exists(path):
            print(f"Vaastav data for {season} not found at {path}")
            return None
            
        df = pd.read_csv(path)
        
        # Drop existing GW if it exists to avoid duplication when renaming round
        if 'GW' in df.columns:
            df = df.drop(columns=['GW'])
            
        # Ensure standard column names
        rename_map = {
            'round': 'GW',
            'element': 'player_id',
            'value': 'price'
        }
        df = df.rename(columns=rename_map)
        
        # Position might be 'GK', 'DEF', 'MID', 'FWD'
        # Convert price to float
        if 'price' in df.columns:
            df['price'] = df['price'] / 10.0
            
        # Ensure all rolling cols are numeric
        for col in self.rolling_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0
                
        # Drop duplicates if any (e.g. double gameweeks might have 2 rows for same GW, we should aggregate them or keep them separate.
        # For simplicity, if double GW, we group by GW and sum stats, mean price)
        agg_dict = {col: 'sum' for col in self.rolling_cols}
        agg_dict.update({
            'total_points': 'sum',
            'price': 'mean',
            'was_home': 'first', # Approximation
            'opponent_team': 'first',
            'team': 'first',
            'position': 'first',
            'kickoff_time': 'first'
        })
        
        # Group by player and GW to handle Double Gameweeks properly
        df_grouped = df.groupby(['player_id', 'GW']).agg(agg_dict).reset_index()
        df_grouped['season'] = season
        
        return df_grouped

    def _load_current_season(self):
        # We need bootstrap_static for player team and position
        static_path = os.path.join(self.raw_dir, "bootstrap_static.json")
        if not os.path.exists(static_path):
            print("bootstrap_static.json not found")
            return None
            
        with open(static_path, 'r', encoding='utf-8') as f:
            static = json.load(f)
            
        element_types = {e['id']: e['singular_name_short'] for e in static['element_types']}
        player_meta = {}
        for p in static['elements']:
            player_meta[p['id']] = {
                'team': p['team'],
                'position': element_types.get(p['element_type'], 'MID')
            }
            
        # Determine current GW to find cache file
        current_gw = 1
        for ev in static['events']:
            if ev['is_current']:
                current_gw = ev['id']
                break
                
        cache_path = os.path.join(self.cache_dir, f"element_summary_gw_{current_gw}.json")
        if not os.path.exists(cache_path):
            print(f"Current season cache {cache_path} not found")
            return None
            
        with open(cache_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)
            
        rows = []
        for pid_str, data in summaries.items():
            pid = int(pid_str)
            meta = player_meta.get(pid, {'team': 0, 'position': 'UNK'})
            
            for hw in data.get('history', []):
                row = {
                    'player_id': pid,
                    'GW': hw['round'],
                    'season': '2024-25',
                    'price': hw['value'] / 10.0,
                    'was_home': hw['was_home'],
                    'opponent_team': hw['opponent_team'],
                    'team': meta['team'],
                    'position': meta['position'],
                    'kickoff_time': hw['kickoff_time']
                }
                for col in self.rolling_cols:
                    val = hw.get(col, 0)
                    try:
                        row[col] = float(val)
                    except (ValueError, TypeError):
                        row[col] = 0.0
                rows.append(row)
                
        df = pd.DataFrame(rows)
        if df.empty: return df
        
        # Handle Double Gameweeks
        agg_dict = {col: 'sum' for col in self.rolling_cols}
        agg_dict.update({
            'price': 'mean',
            'was_home': 'first',
            'opponent_team': 'first',
            'team': 'first',
            'position': 'first',
            'kickoff_time': 'first'
        })
        df_grouped = df.groupby(['player_id', 'GW']).agg(agg_dict).reset_index()
        df_grouped['season'] = '2024-25'
        
        return df_grouped

    def build_features(self):
        print("Building time-series dataset...")
        
        dfs = []
        df_2223 = self._load_vaastav_season("2022-23")
        if df_2223 is not None: dfs.append(df_2223)
        
        df_2324 = self._load_vaastav_season("2023-24")
        if df_2324 is not None: dfs.append(df_2324)
            
        df_curr = self._load_current_season()
        if df_curr is not None: dfs.append(df_curr)
            
        if not dfs:
            print("No data available.")
            return None
            
        df_all = pd.concat(dfs, ignore_index=True)
        
        # Sort chronologically
        df_all = df_all.sort_values(['season', 'player_id', 'GW']).reset_index(drop=True)
        
        # Feature Engineering: Rolling Averages (STRICT ANTI-LEAKAGE)
        # We want to predict GW N using data from GW N-1, N-2, etc.
        # So we group by season and player_id, then calculate rolling sums/means,
        # and then SHIFT them by 1.
        
        # Convert kickoff_time to datetime
        df_all['kickoff_time'] = pd.to_datetime(df_all['kickoff_time'], errors='coerce', utc=True)
        
        print("Calculating rolling features...")
        grouped = df_all.groupby(['season', 'player_id'])
        
        # Days rest going into THIS gameweek
        df_all['days_rest'] = (df_all['kickoff_time'] - grouped['kickoff_time'].shift(1)).dt.total_seconds() / (24 * 3600)
        # Fill missing days_rest (e.g. GW1) with a default, like 7 days
        df_all['days_rest'] = df_all['days_rest'].fillna(7.0)
        
        # Benching flag
        df_all['benched'] = (df_all['starts'] == 0).astype(int)
        
        # Dictionary to hold new shifted features
        new_features = {}
        
        for col in self.rolling_cols + ['benched']:
            # Shifted 1: The value from the immediate previous gameweek
            new_features[f'{col}_last_1'] = grouped[col].shift(1)
            
            # Rolling means (or sums) for last 3 and 5 GWs
            if col == 'benched':
                # For benched, we want the SUM (count of benchings)
                new_features[f'{col}_sum_last_3'] = grouped[col].apply(lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()).reset_index(level=[0,1], drop=True)
                new_features[f'{col}_sum_last_5'] = grouped[col].apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()).reset_index(level=[0,1], drop=True)
            else:
                new_features[f'{col}_mean_last_3'] = grouped[col].apply(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()).reset_index(level=[0,1], drop=True)
                new_features[f'{col}_mean_last_5'] = grouped[col].apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)
            
        # Add the features to the dataframe
        for k, v in new_features.items():
            df_all[k] = v
            
        # Target variables
        df_all['target'] = df_all['total_points']
        df_all['target_minutes'] = df_all['minutes']
        
        # ---------------------------------------------------------------
        # Merge bookmaker odds features (optional, fallback-safe)
        # ---------------------------------------------------------------
        print("Merging bookmaker odds features...")
        try:
            from src.api.odds import OddsClient, LEAGUE_DEFAULTS
            odds_client = OddsClient(cache_dir=self.cache_dir, raw_dir=self.raw_dir)
            odds_client.download_historical_odds()
            
            odds_frames = []
            for season in df_all['season'].unique():
                df_odds = odds_client.load_historical_odds(season)
                if df_odds is not None:
                    df_odds['season'] = season
                    odds_frames.append(df_odds)
                    
            if odds_frames:
                df_odds_all = pd.concat(odds_frames, ignore_index=True)
                
                # Parse date and create a match key
                df_odds_all['date'] = pd.to_datetime(df_odds_all['date'], dayfirst=True, errors='coerce')
                
                # For the main dataset, extract date from kickoff_time (already dropped? need to keep it briefly)
                # Actually kickoff_time was dropped above. Let's re-extract date from the original
                # We'll match by season + team + GW instead of date
                # Since odds are per-match and we have one row per match per team,
                # we need to match odds to vaastav by season + team_name + round/GW
                
                # Build a lookup: for each season+team, order matches by date and assign GW
                odds_gw_lookup = {}
                for season in df_odds_all['season'].unique():
                    season_odds = df_odds_all[df_odds_all['season'] == season].sort_values('date')
                    for team_name in season_odds['team_name'].unique():
                        team_matches = season_odds[season_odds['team_name'] == team_name].reset_index(drop=True)
                        team_matches['gw_rank'] = range(1, len(team_matches) + 1)
                        for _, row in team_matches.iterrows():
                            key = (season, team_name, int(row['gw_rank']))
                            odds_gw_lookup[key] = {
                                'win_prob': row['win_prob'],
                                'draw_prob': row['draw_prob'],
                                'loss_prob': row['loss_prob'],
                                'team_implied_goals': row['team_implied_goals'],
                                'opponent_implied_goals': row['opponent_implied_goals'],
                                'clean_sheet_prob': row['clean_sheet_prob'],
                            }
                
                # Map odds to each row in df_all
                odds_cols = ['win_prob', 'draw_prob', 'loss_prob', 
                             'team_implied_goals', 'opponent_implied_goals', 'clean_sheet_prob']
                
                # We need team as string for lookup
                team_str = df_all['team'].astype(str) if df_all['team'].dtype.name == 'category' else df_all['team']
                
                for col in odds_cols:
                    df_all[col] = [
                        odds_gw_lookup.get((row_season, row_team, int(row_gw)), LEAGUE_DEFAULTS).get(col, LEAGUE_DEFAULTS[col])
                        for row_season, row_team, row_gw in zip(df_all['season'], team_str, df_all['GW'])
                    ]
                
                matched = sum(1 for s, t, g in zip(df_all['season'], team_str, df_all['GW']) 
                             if (s, t, int(g)) in odds_gw_lookup)
                print(f"  Matched {matched}/{len(df_all)} rows with historical odds ({matched/len(df_all)*100:.1f}%)")
            else:
                print("  No historical odds found. Using league defaults.")
                for col, val in LEAGUE_DEFAULTS.items():
                    if col in ['win_prob', 'draw_prob', 'loss_prob', 
                               'team_implied_goals', 'opponent_implied_goals', 'clean_sheet_prob']:
                        df_all[col] = val
        except Exception as e:
            print(f"  Odds integration failed: {e}. Using league defaults.")
            from src.api.odds import LEAGUE_DEFAULTS  # noqa: F811
            for col in ['win_prob', 'draw_prob', 'loss_prob', 
                        'team_implied_goals', 'opponent_implied_goals', 'clean_sheet_prob']:
                df_all[col] = LEAGUE_DEFAULTS[col]
        
        # Derive anytime_goal_scorer_prob from team_implied_goals + position
        from src.api.odds import OddsClient, POSITION_GOAL_SHARE
        pos_str = df_all['position'].astype(str) if df_all['position'].dtype.name == 'category' else df_all['position']
        df_all['anytime_goal_scorer_prob'] = [
            OddsClient.compute_anytime_scorer_prob(tig, pos)
            for tig, pos in zip(df_all['team_implied_goals'], pos_str)
        ]
        
        # Fill missing values for numerical columns
        num_cols = df_all.select_dtypes(include=[np.number]).columns
        df_all[num_cols] = df_all[num_cols].fillna(0)
        
        # Convert string categoricals to pandas categories
        cat_cols = ['position', 'team', 'opponent_team', 'was_home']
        for c in cat_cols:
            df_all[c] = df_all[c].astype(str).astype('category')
            
        # Drop temporary columns
        df_all = df_all.drop(columns=['benched', 'kickoff_time'])
            
        # Save complete dataset
        out_path = os.path.join(self.processed_dir, "historical_features.parquet")
        df_all.to_parquet(out_path, index=False)
        print(f"Saved {len(df_all)} rows to {out_path}")
        
        return df_all

if __name__ == "__main__":
    builder = HistoryBuilder()
    builder.build_features()

