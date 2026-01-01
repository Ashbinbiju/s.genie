class ChipStrategy:
    def __init__(self, team_id, history_data):
        self.team_id = team_id
        # Parse used chips from history
        self.used_chips = {}
        if history_data and 'chips' in history_data:
            for chip in history_data['chips']:
                # chip['name'] is like 'wildcard', 'bboost', '3xc', 'freehit'
                self.used_chips[chip['name']] = chip['event']
                
    def analyze(self, optimized_team, bench, current_gw, full_squad_value=0, wildcard_diff=0, freehit_diff=0, active_players=15):
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
        # Condition: Optimization gain > 20 points
        wc_status = self._check_wildcard(current_gw, wildcard_diff)
        recommendations.append(wc_status)
        
        # 4. Free Hit 
        # Condition: Massive gain (>25) OR Severe Blank (< 9 players)
        fh_status = self._check_freehit(current_gw, freehit_diff, active_players)
        recommendations.append(fh_status)
        
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

    def _check_wildcard(self, current_gw, diff):
        # FPL allows 2 Wildcards:
        # WC1: Start of season to GW19 deadline
        # WC2: GW20 deadline to end of season
        
        wc_event = self.used_chips.get('wildcard')
        
        is_wc_available = True
        status_reason = "Available"
        
        if wc_event:
            # If we are in the second half (GW20+)
            if current_gw >= 20:
                # If the WC usage was from the first half (< 20), we have a NEW one.
                if wc_event < 20:
                    is_wc_available = True
                    status_reason = "Available (Wildcard 2 active from GW20)"
                else:
                    # Usage was recent (>= 20), so WC2 is gone
                    is_wc_available = False
                    status_reason = f"Used in GW{wc_event}"
            else:
                # First half (< 20)
                is_wc_available = False # Already used WC1
                status_reason = f"Used in GW{wc_event}"
        
        if is_wc_available:
            if diff > 20:
                return {
                    'chip': 'Wildcard',
                    'recommendation': 'Recommended',
                    'icon': '🔥',
                    'reason': f"Huge potential gain! (+{diff:.1f} pts vs current team)"
                }
            elif diff > 12:
                return {
                    'chip': 'Wildcard',
                    'recommendation': 'Consider',
                    'icon': '🤔',
                    'reason': f"Good potential gain (+{diff:.1f} pts)."
                }
            else:
                return {
                    'chip': 'Wildcard',
                    'recommendation': 'Save',
                    'icon': '💾',
                    'reason': f"Current team is strong (Only +{diff:.1f} gain)."
                }
        else:
             return {
                'chip': 'Wildcard',
                'recommendation': 'Used',
                'icon': '❌',
                'reason': status_reason
            }

    def _check_freehit(self, current_gw, diff, active_players):
        fh_event = self.used_chips.get('freehit')
        is_fh_available = True
        status_reason = "Available for blank gameweeks."

        if fh_event:
             # User requested restoration post GW20
             if current_gw >= 20:
                 if fh_event < 20:
                     is_fh_available = True
                     status_reason = "Available (Free Hit 2 active from GW20)"
                 else:
                     is_fh_available = False
                     status_reason = f"Used in GW{fh_event}"
             else:
                 is_fh_available = False
                 status_reason = f"Used in GW{fh_event}"
        
        if is_fh_available:
            if active_players < 9:
                return {
                    'chip': 'Free Hit',
                    'recommendation': 'Recommended',
                    'icon': '🚨',
                    'reason': f"Crisis! Only {active_players} players active."
                }
            elif diff > 25:
                 return {
                    'chip': 'Free Hit',
                    'recommendation': 'Recommended',
                    'icon': '🔥',
                    'reason': f"One-week punt opportunity! (+{diff:.1f} pts)"
                }
            else:
                return {
                    'chip': 'Free Hit',
                    'recommendation': 'Save',
                    'icon': '💾',
                    'reason': f"Save for Blank GWs (Currently {active_players} active)."
                }
        else:
            return {
                'chip': 'Free Hit',
                'recommendation': 'Used',
                'icon': '❌',
                'reason': status_reason
            }
