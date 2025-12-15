class ChipStrategy:
    def __init__(self, team_id, history_data):
        self.team_id = team_id
        # Parse used chips from history
        self.used_chips = {}
        if history_data and 'chips' in history_data:
            for chip in history_data['chips']:
                # chip['name'] is like 'wildcard', 'bboost', '3xc', 'freehit'
                self.used_chips[chip['name']] = chip['event']
                
    def analyze(self, optimized_team, bench, current_gw, full_squad_value=0):
        """
        Analyzes the squad to recommend chips.
        Returns a list of dictionaries: {'chip': 'Name', 'status': 'Recommend/Avoid', 'reason': '...'}
        """
        recommendations = []
        
        # 1. Bench Boost
        # Condition: Bench XP > 15 AND All bench players have games
        bb_status = self._check_bench_boost(bench)
        recommendations.append(bb_status)
        
        # 2. Triple Captain
        # Condition: Top player XP > 10.0 (Adjustable)
        tc_status = self._check_triple_captain(optimized_team)
        recommendations.append(tc_status)
        
        # 3. Wildcard
        # Condition: Optimization gain > 20 points (This requires comparing current vs optimal, 
        # which acts as a proxy here if we assume 'optimized_team' is the target)
        # For now, we just check availability as specific squad diff is complex to pass here
        wc_status = self._check_wildcard()
        recommendations.append(wc_status)
        
        return recommendations

    def _check_bench_boost(self, bench):
        if 'bboost' in self.used_chips:
            return {
                'chip': 'Bench Boost',
                'recommendation': 'Used',
                'icon': '❌',
                'reason': f"Used in GW{self.used_chips['bboost']}"
            }
            
        bench_xp = bench['predicted_points'].sum()
        # Check if any bench player has 0 chance or no fixture (simplified via XP)
        if bench_xp > 18:
            return {
                'chip': 'Bench Boost',
                'recommendation': 'Recommended',
                'icon': '🔥',
                'reason': f"Strong Bench! Projected {bench_xp:.1f} pts."
            }
        elif bench_xp > 12:
            return {
                'chip': 'Bench Boost',
                'recommendation': 'Consider',
                'icon': '🤔',
                'reason': f"Decent Bench ({bench_xp:.1f} pts), but could be higher."
            }
        else:
            return {
                'chip': 'Bench Boost',
                'recommendation': 'Save',
                'icon': '💾',
                'reason': f"Bench too weak ({bench_xp:.1f} pts)."
            }

    def _check_triple_captain(self, team):
        if '3xc' in self.used_chips:
            return {
                'chip': 'Triple Captain',
                'recommendation': 'Used',
                'icon': '❌',
                'reason': f"Used in GW{self.used_chips['3xc']}"
            }
            
        top_player = team.loc[team['predicted_points'].idxmax()]
        xp = top_player['predicted_points']
        
        if xp >= 11.0: # Very high threshold
            return {
                'chip': 'Triple Captain',
                'recommendation': 'Recommended',
                'icon': '🔥',
                'reason': f"Captain {top_player['web_name']} predicted massive {xp:.1f} pts!"
            }
        elif xp >= 8.0:
            return {
                'chip': 'Triple Captain',
                'recommendation': 'Consider',
                'icon': '🤔',
                'reason': f"{top_player['web_name']} has good fixture ({xp:.1f} pts)."
            }
        else:
            return {
                'chip': 'Triple Captain',
                'recommendation': 'Save',
                'icon': '💾',
                'reason': f"No explosive captain option (<8 pts)."
            }

    def _check_wildcard(self):
        # FPL has two wildcards. We check 'wildcard' entry.
        # Usually API distinguishes via separate names or just 'wildcard'
        # Simplified: If used, it returns 'wildcard'. 
        # Real logic often needs to check GW to see if it's WC1 or WC2 window.
        # For now, simplistic check.
        
        if 'wildcard' in self.used_chips:
             return {
                'chip': 'Wildcard',
                'recommendation': 'Used',
                'icon': '❌',
                'reason': f"Used in GW{self.used_chips['wildcard']}"
            }
        
        return {
            'chip': 'Wildcard',
            'recommendation': 'Available',
            'icon': '🃏',
            'reason': "Available if your team needs a total rebuild."
        }
