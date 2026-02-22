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
        bb_status = self._check_bench_boost(bench, current_gw)
        recommendations.append(bb_status)
        
        # 2. Triple Captain
        # Condition: Top player XP > 10.0 (Adjustable)
        tc_status = self._check_triple_captain(optimized_team, current_gw)
        recommendations.append(tc_status)
        
        # 3. Wildcard
        # Condition: Optimization gain > 20 points
        wc_status = self._check_wildcard(current_gw, wildcard_diff)
        recommendations.append(wc_status)
        
        # 4. Free Hit 
        # Condition: Massive gain (>25) OR Severe Blank (< 9 players)
        fh_status = self._check_freehit(current_gw, freehit_diff, active_players)
        recommendations.append(fh_status)
        
        # Add visual indicator for restored chips
        chip_keys = {'Bench Boost': 'bboost', 'Triple Captain': '3xc', 'Wildcard': 'wildcard', 'Free Hit': 'freehit'}
        if current_gw >= 20:
            for rec in recommendations:
                if rec['recommendation'] != 'Used':
                    key = chip_keys.get(rec['chip'])
                    event_used = self.used_chips.get(key)
                    if event_used and event_used < 20:
                        rec['reason'] = f"[RESTORED 2nd CHIP] {rec['reason']}"

        return recommendations

    def _is_chip_available(self, chip_key, current_gw):
        """
        Returns (is_available: bool, status_reason: str) for the given chip.
        Handles restoration logic for chips used before GW20.
        """
        GW_RESTORATION_THRESHOLD = 20
        event_used = self.used_chips.get(chip_key)
        
        if not event_used:
            return True, "Available"
        
        if current_gw >= GW_RESTORATION_THRESHOLD and event_used < GW_RESTORATION_THRESHOLD:
            # Get a "friendly" name for the chip
            chip_display = {
                'bboost': 'Bench Boost 2',
                '3xc': 'Triple Captain 2',
                'wildcard': 'Wildcard 2',
                'freehit': 'Free Hit 2'
            }.get(chip_key, chip_key)
            return True, f"Available ({chip_display} active from GW{GW_RESTORATION_THRESHOLD})"
        
        return False, f"Used in GW{event_used}"

    def _check_bench_boost(self, bench, current_gw):
        is_bb_available, status_reason = self._is_chip_available('bboost', current_gw)

        if not is_bb_available:
            return {
                'chip': 'Bench Boost',
                'recommendation': 'Used',
                'icon': '❌',
                'reason': status_reason
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

    def _check_triple_captain(self, team, current_gw):
        is_tc_available, status_reason = self._is_chip_available('3xc', current_gw)

        if not is_tc_available:
            return {
                'chip': 'Triple Captain',
                'recommendation': 'Used',
                'icon': '❌',
                'reason': status_reason
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
        is_wc_available, status_reason = self._is_chip_available('wildcard', current_gw)
        
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
        is_fh_available, status_reason = self._is_chip_available('freehit', current_gw)
        
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
                    'reason': f"No need. You have a full squad ({active_players} players)."
                }
        else:
            return {
                'chip': 'Free Hit',
                'recommendation': 'Used',
                'icon': '❌',
                'reason': status_reason
            }
