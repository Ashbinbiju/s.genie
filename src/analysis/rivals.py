import pandas as pd

class RivalSpy:
    def __init__(self, my_team_df, rival_team_df):
        """
        my_team_df: DataFrame of your squad (Must include 'web_name', 'predicted_points', 'id')
        rival_team_df: DataFrame of rival's squad
        """
        self.my_team = my_team_df
        self.rival_team = rival_team_df
        
    def compare(self):
        """
        Compares the two squads and returns a dictionary with analysis.
        """
        my_ids = set(self.my_team['id'].tolist())
        rival_ids = set(self.rival_team['id'].tolist())
        
        common_ids = my_ids.intersection(rival_ids)
        my_diff_ids = my_ids - rival_ids
        rival_diff_ids = rival_ids - my_ids
        
        # Get DataFrames
        common_players = self.my_team[self.my_team['id'].isin(common_ids)]
        my_diffs = self.my_team[self.my_team['id'].isin(my_diff_ids)]
        rival_diffs = self.rival_team[self.rival_team['id'].isin(rival_diff_ids)]
        
        # Calculate Swing
        my_unique_xp = my_diffs['predicted_points'].sum()
        rival_unique_xp = rival_diffs['predicted_points'].sum()
        net_swing = my_unique_xp - rival_unique_xp
        
        # Danger Player (Max XP in rival diffs)
        danger_player = None
        if not rival_diffs.empty:
            danger_idx = rival_diffs['predicted_points'].idxmax()
            danger_player = rival_diffs.loc[danger_idx]
            
        # Summary & Insights
        rival_heavy_hitters = len(rival_diffs[rival_diffs['predicted_points'] >= 5.5])
        my_zeros = len(my_diffs[my_diffs['predicted_points'] < 0.5])
        
        # Position Analysis
        pos_diff_xg = {
            'GK': my_diffs[my_diffs['element_type']==1]['predicted_points'].sum() - rival_diffs[rival_diffs['element_type']==1]['predicted_points'].sum(),
            'DEF': my_diffs[my_diffs['element_type']==2]['predicted_points'].sum() - rival_diffs[rival_diffs['element_type']==2]['predicted_points'].sum(),
            'MID': my_diffs[my_diffs['element_type']==3]['predicted_points'].sum() - rival_diffs[rival_diffs['element_type']==3]['predicted_points'].sum(),
            'FWD': my_diffs[my_diffs['element_type']==4]['predicted_points'].sum() - rival_diffs[rival_diffs['element_type']==4]['predicted_points'].sum(),
        }
        main_gap_pos = min(pos_diff_xg, key=pos_diff_xg.get) # Position where we are losing most XP
            
        return {
            'common_count': len(common_ids),
            'differential_count': len(my_diff_ids),
            'my_diffs': my_diffs.sort_values('predicted_points', ascending=False),
            'rival_diffs': rival_diffs.sort_values('predicted_points', ascending=False),
            'my_unique_xp': my_unique_xp,
            'rival_unique_xp': rival_unique_xp,
            'net_swing': net_swing,
            'danger_player': danger_player,
            'rival_heavy_hitters': rival_heavy_hitters,
            'my_zeros': my_zeros,
            'main_gap_pos': main_gap_pos
        }
