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

    def process(self):
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

        # 6. Fixture Difficulty
        fixtures = self.load_fixtures()
        if fixtures is not None:
             difficulty_map = self.calculate_fixture_difficulty(fixtures)
             # Map difficulty to player's team
             merged['fixture_difficulty'] = merged['team'].map(difficulty_map).fillna(3) # Default average
        else:
             merged['fixture_difficulty'] = 3

        # Select columns for model
        features = [
            'id', 'web_name', 'team', 'element_type', 'price', 
            'form', 'points_per_game', 'ict_index', 'ep_next',
            'xG', 'xA', 'xG_per_90', 'xA_per_90', 'minutes_prob', 
            'total_points', 'fixture_difficulty',
            'news', 'chance_of_playing_next_round'
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

        output_path = os.path.join(self.processed_dir, "player_features.parquet")
        final_df.to_parquet(output_path)
        print(f"Saved processed features to {output_path}")
        return final_df

    def load_fixtures(self):
        path = os.path.join(self.raw_dir, "fixtures.json")
        if os.path.exists(path):
            return pd.read_json(path)
        return None

    def calculate_fixture_difficulty(self, fixtures_df, next_n=5):
        # Filter for unfinished games
        future = fixtures_df[fixtures_df['finished'] == False].sort_values('kickoff_time')
        
        team_difficulty = {}
        # There are 20 teams, IDs 1-20
        for team_id in range(1, 21):
            # Find next n games for this team
            # Matches where team is home (team_h) or away (team_a)
            matches = future[
                (future['team_h'] == team_id) | (future['team_a'] == team_id)
            ].head(next_n)
            
            diff_sum = 0
            count = 0
            for _, match in matches.iterrows():
                if match['team_h'] == team_id:
                    diff_sum += match['team_h_difficulty']
                else:
                    diff_sum += match['team_a_difficulty']
                count += 1
            
            team_difficulty[team_id] = diff_sum / count if count > 0 else 3
            
        return team_difficulty

if __name__ == "__main__":
    processor = FeatureProcessor()
    processor.process()
