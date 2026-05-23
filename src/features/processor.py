import pandas as pd
import numpy as np
import os

class FeatureProcessor:
    def __init__(self, data_dir="data"):
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_fpl_data(self):
        """Loads FPL bootstrap static data."""
        try:
            path = os.path.join(self.raw_dir, "bootstrap_static.json")
            data = pd.read_json(path)
            # Elements is the key list of players
            players = pd.DataFrame(data['elements'])
            teams = pd.DataFrame(data['teams'])
            events = pd.DataFrame(data['events'])
            return players, teams, events
        except ValueError:
            # Fallback if saved structure is different (e.g. dict)
            import json
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data['elements']), pd.DataFrame(data['teams']), pd.DataFrame(data['events'])
        except FileNotFoundError:
            print("FPL data not found. Run FPLClient first.")
            return None, None, None

    def load_understat_data(self):
        """Loads Understat data."""
        path = os.path.join(self.raw_dir, "understat_players.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        print("Understat data not found. Run UnderstatClient first.")
        return None

    def process(self, force_refresh=False):
        # Force update for Streamlit cloud sync
        # Output path
        output_path = os.path.join(self.processed_dir, "player_features.parquet")
        
        # Check cache
        if not force_refresh and os.path.exists(output_path):
            df = pd.read_parquet(output_path)
            # Validate significant columns exist
            required = ['next_opponent', 'news', 'fixture_difficulty', 'photo']
            if all(col in df.columns for col in required):
                return df
            print("Cached data missing new columns (e.g. next_opponent). Regenerating...")
        
        print("Processing features...")
        fpl_players, fpl_teams, _ = self.load_fpl_data()
        understat_players = self.load_understat_data()

        if fpl_players is None:
            return None

        # Map Understat to FPL
        if understat_players is not None:
            fpl_players['web_name_norm'] = fpl_players['web_name'].str.lower().str.replace(r'[^a-z]', '', regex=True)
            understat_players['player_name_norm'] = understat_players['player_name'].str.lower().str.replace(r'[^a-z]', '', regex=True)
            merged = pd.merge(fpl_players, understat_players, left_on='web_name_norm', right_on='player_name_norm', how='left', suffixes=('_fpl', '_us'))
        else:
            print("Warning: Understat data missing. Proceeding with FPL data only.")
            merged = fpl_players.copy()
            # Add missing columns with 0
            for col in ['xG', 'xA', 'time']:
                merged[col] = 0

        # Feature Engineering highlights
        # 1. Price
        merged['price'] = merged['now_cost'] / 10.0
        
        # 2. Form (FPL form is a string)
        merged['form'] = pd.to_numeric(merged['form'], errors='coerce')
        
        # 3. xG/xA per 90
        if 'time' in merged.columns and 'xG' in merged.columns:
            merged['xG_per_90'] = merged['xG'] / (merged['time'] / 90)
            merged['xA_per_90'] = merged['xA'] / (merged['time'] / 90)
            merged['xG_per_90'] = merged['xG_per_90'].fillna(0).replace([np.inf, -np.inf], 0)
            merged['xA_per_90'] = merged['xA_per_90'].fillna(0).replace([np.inf, -np.inf], 0)
        else:
             merged['xG_per_90'] = 0
             merged['xA_per_90'] = 0

        # 4. Minutes Probability (Basic proxy from 'chance_of_playing_next_round')
        merged['minutes_prob'] = merged['chance_of_playing_next_round'].fillna(100) / 100.0

        # 5. Value (Points per million)
        merged['ppm'] = merged['total_points'] / merged['price']

        # 6. Fixture Difficulty & Next Opponent
        fixtures = self.load_fixtures()
        if fixtures is not None:
             match_data = self.calculate_fixture_difficulty(fixtures, fpl_teams)
             # Map difficulty to player's team
             merged['fixture_difficulty'] = merged['team'].map(lambda x: match_data.get(x, {}).get('fixture_difficulty', 3))
             merged['next_opponent'] = merged['team'].map(lambda x: match_data.get(x, {}).get('next_opponent', "-"))
             merged['opponent_team'] = merged['team'].map(lambda x: match_data.get(x, {}).get('opponent_team_id', "0")).astype(str)
             merged['was_home'] = merged['team'].map(lambda x: match_data.get(x, {}).get('is_home', True)).astype(str)
        else:
             merged['fixture_difficulty'] = 3
             merged['next_opponent'] = "-"
             merged['opponent_team'] = "0"
             merged['was_home'] = "True"

        # Add team_code for shirt images
        team_code_map = fpl_teams.set_index('id')['code'].to_dict()
        merged['team_code'] = merged['team'].map(team_code_map)

        # Map element_type to position string
        type_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        merged['position'] = merged['element_type'].map(type_map).astype(str)

        # ---------------------------------------------------------------
        # Bookmaker Odds (optional, fallback-safe)
        # ---------------------------------------------------------------
        try:
            import sys
            _proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if _proj not in sys.path: sys.path.insert(0, _proj)
            from src.api.odds import OddsClient, LEAGUE_DEFAULTS, TEAM_NAME_NORMALIZE
            
            odds_client = OddsClient()
            live_odds = odds_client.get_current_odds()
            
            # Map FPL team id → team name
            team_name_map = fpl_teams.set_index('id')['name'].to_dict()
            
            odds_cols = ['win_prob', 'draw_prob', 'loss_prob', 
                         'team_implied_goals', 'opponent_implied_goals', 'clean_sheet_prob']
            
            for col in odds_cols:
                merged[col] = LEAGUE_DEFAULTS[col]  # defaults
                
            if live_odds:
                for idx, row in merged.iterrows():
                    team_name = team_name_map.get(row['team'], '')
                    # Try exact match, then normalized
                    odds_data = live_odds.get(team_name) or live_odds.get(
                        TEAM_NAME_NORMALIZE.get(team_name, team_name))
                    if odds_data:
                        for col in odds_cols:
                            merged.at[idx, col] = odds_data.get(col, LEAGUE_DEFAULTS[col])
                            
            # Derive anytime_goal_scorer_prob
            merged['anytime_goal_scorer_prob'] = merged.apply(
                lambda r: OddsClient.compute_anytime_scorer_prob(
                    r['team_implied_goals'], r['position']), axis=1)
                    
            print(f"Odds features added (live={'Yes' if live_odds else 'No, using defaults'})")
        except Exception as e:
            print(f"Odds integration skipped: {e}")
            merged['win_prob'] = 0.33
            merged['draw_prob'] = 0.33
            merged['loss_prob'] = 0.33
            merged['team_implied_goals'] = 1.35
            merged['opponent_implied_goals'] = 1.35
            merged['clean_sheet_prob'] = 0.30
            merged['anytime_goal_scorer_prob'] = 0.10

        # Select columns for model
        features = [
            'id', 'web_name', 'team', 'team_code', 'element_type', 'position', 'price', 
            'form', 'points_per_game', 'ict_index', 'ep_next',
            'xG', 'xA', 'xG_per_90', 'xA_per_90', 'minutes_prob', 
            'total_points', 'fixture_difficulty',
            'news', 'chance_of_playing_next_round', 'next_opponent',
            'opponent_team', 'was_home',
            'win_prob', 'draw_prob', 'loss_prob',
            'team_implied_goals', 'opponent_implied_goals', 
            'clean_sheet_prob', 'anytime_goal_scorer_prob',
            'photo'
        ]
        
        # Rename element_type_fpl check, usually in FPL it's 'element_type'
        if 'element_type_fpl' not in merged.columns and 'element_type' in merged.columns:
             merged['element_type_fpl'] = merged['element_type']
        
        # Restore ID if lost in merge
        if 'id' not in merged.columns and 'id_fpl' in merged.columns:
            merged['id'] = merged['id_fpl']

        # Iterate over features to ensure they exist
        for f in features:
            if f not in merged.columns:
                merged[f] = 0
                
        final_df = merged[features].copy()
        
        # Cleanings
        cols_to_float = ['points_per_game', 'ict_index', 'ep_next', 'fixture_difficulty']
        for c in cols_to_float:
            final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0)

        # Ensure strings
        final_df['news'] = final_df['news'].fillna("")
        final_df['next_opponent'] = final_df['next_opponent'].fillna("-")

        output_path = os.path.join(self.processed_dir, "player_features.parquet")
        final_df.to_parquet(output_path)
        print(f"Saved processed features to {output_path}")
        return final_df

    def load_fixtures(self):
        path = os.path.join(self.raw_dir, "fixtures.json")
        if os.path.exists(path):
            return pd.read_json(path)
        return None

    def calculate_fixture_difficulty(self, fixtures_df, teams_df, next_n=5):
        # Create team map: id -> short_name
        team_map = teams_df.set_index('id')['short_name'].to_dict()
        
        # Filter for unfinished games
        future = fixtures_df[fixtures_df['finished'] == False].sort_values('kickoff_time')
        
        team_data = {} # {team_id: {'difficulty': float, 'next_match': str}}
        
        # There are 20 teams, IDs 1-20
        for team_id in range(1, 21):
            # Find next n games for this team
            matches = future[
                (future['team_h'] == team_id) | (future['team_a'] == team_id)
            ].head(next_n)
            
            diff_sum = 0
            count = 0
            next_opp_str = "-"
            
            for i, (_, match) in enumerate(matches.iterrows()):
                is_home = (match['team_h'] == team_id)
                opp_id = match['team_a'] if is_home else match['team_h']
                opp_name = team_map.get(opp_id, "?")
                difficulty = match['team_h_difficulty'] if is_home else match['team_a_difficulty']
                
                if i == 0:
                    venue = "(H)" if is_home else "(A)"
                    next_opp_str = f"{opp_name} {venue}"
                    next_opp_id = opp_id
                    next_is_home = is_home

                diff_sum += difficulty
                count += 1
            
            avg_diff = diff_sum / count if count > 0 else 3
            team_data[team_id] = {
                'fixture_difficulty': avg_diff, 
                'next_opponent': next_opp_str,
                'opponent_team_id': next_opp_id if count > 0 else 0,
                'is_home': next_is_home if count > 0 else True
            }
            
        return team_data

if __name__ == "__main__":
    processor = FeatureProcessor()
    processor.process()
