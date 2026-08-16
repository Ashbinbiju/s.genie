"""
Chip strategy advisor.

IMPORTANT — thresholds are RELATIVE, not absolute.

The model predicts *expected* points: a conditional mean that regresses toward ~1-6,
where the best player in the game projects around 6 XP. The original thresholds were
written against *actual* FPL points (Triple Captain at 11.0, Bench Boost at 18.0) and
were therefore mathematically unreachable — measured against a real prediction set,
zero players in the entire game cleared even the "Consider" bar of 8.0, so those two
chips could never fire.

Everything below is expressed as a ratio to the current squad's own average starter, so
the advice stays correct if the model is retrained and its output scale shifts.
"""

GW_RESTORATION_THRESHOLD = 20

# Triple Captain: captain XP as a multiple of the average starter.
TC_RECOMMEND_RATIO = 2.00
TC_CONSIDER_RATIO = 1.60

# Bench Boost: average bench player XP as a fraction of the average starter.
BB_RECOMMEND_RATIO = 0.75
BB_CONSIDER_RATIO = 0.55

# Wildcard / Free Hit: projected XI gain as a fraction of the current XI's total XP.
WC_RECOMMEND_RATIO = 0.25
WC_CONSIDER_RATIO = 0.12
FH_RECOMMEND_RATIO = 0.35

# A squad with fewer than this many players projected to feature is in crisis.
FH_CRISIS_ACTIVE_PLAYERS = 9

CHIP_KEYS = {
    'Bench Boost': 'bboost',
    'Triple Captain': '3xc',
    'Wildcard': 'wildcard',
    'Free Hit': 'freehit',
}

CHIP_DISPLAY_2 = {
    'bboost': 'Bench Boost 2',
    '3xc': 'Triple Captain 2',
    'wildcard': 'Wildcard 2',
    'freehit': 'Free Hit 2',
}


class ChipStrategy:
    def __init__(self, team_id, history_data):
        self.team_id = team_id
        # Parse used chips from history. A chip may be played twice per season (once
        # either side of the GW20 restoration boundary), so keep every event.
        self.used_chips = {}
        if history_data and 'chips' in history_data:
            for chip in history_data['chips']:
                self.used_chips.setdefault(chip['name'], []).append(chip['event'])

    def analyze(self, current_starters, current_bench, current_gw,
                wildcard_diff=0, freehit_diff=0, active_players=15,
                current_xi_xp=None):
        """
        Recommend chips for the upcoming gameweek.

        current_starters / current_bench must describe the squad you ACTUALLY OWN.
        Passing the post-transfer optimized squad (as this used to receive) produces
        advice about a bench you do not have.

        wildcard_diff / freehit_diff are XI-level point gains, not 15-man squad sums —
        you only score your XI, so comparing full squads overstates the benefit.
        """
        recommendations = []

        mean_starter_xp = (
            float(current_starters['predicted_points'].mean())
            if current_starters is not None and len(current_starters) else 0.0
        )
        if current_xi_xp is None:
            current_xi_xp = (
                float(current_starters['predicted_points'].sum())
                if current_starters is not None and len(current_starters) else 0.0
            )

        recommendations.append(self._check_bench_boost(current_bench, current_gw, mean_starter_xp))
        recommendations.append(self._check_triple_captain(current_starters, current_gw, mean_starter_xp))
        recommendations.append(self._check_wildcard(current_gw, wildcard_diff, current_xi_xp))
        recommendations.append(self._check_freehit(current_gw, freehit_diff, active_players, current_xi_xp))

        # Flag chips that are only available because of the GW20 restoration.
        if current_gw >= GW_RESTORATION_THRESHOLD:
            for rec in recommendations:
                if rec['recommendation'] != 'Used':
                    events = self.used_chips.get(CHIP_KEYS.get(rec['chip']), [])
                    if any(e < GW_RESTORATION_THRESHOLD for e in events):
                        rec['reason'] = f"[RESTORED 2nd CHIP] {rec['reason']}"

        return recommendations

    # ------------------------------------------------------------------
    def _is_chip_available(self, chip_key, current_gw):
        """
        (is_available, status_reason) for a chip.

        FPL grants each chip twice: once for GW1-19 and once from GW20. A chip played
        before GW20 is restored at GW20; a chip already played on or after GW20 is gone.
        """
        # Only chips already played can block one now. Guards against a history payload
        # containing an event later than the gameweek being analysed.
        events = sorted(e for e in self.used_chips.get(chip_key, []) if e <= current_gw)
        if not events:
            return True, "Available"

        if current_gw >= GW_RESTORATION_THRESHOLD:
            used_in_second_half = [e for e in events if e >= GW_RESTORATION_THRESHOLD]
            if not used_in_second_half:
                display = CHIP_DISPLAY_2.get(chip_key, chip_key)
                return True, f"Available ({display} active from GW{GW_RESTORATION_THRESHOLD})"
            return False, f"Used in GW{used_in_second_half[0]}"

        return False, f"Used in GW{events[0]}"

    @staticmethod
    def _used(chip, reason):
        return {'chip': chip, 'recommendation': 'Used', 'icon': '❌', 'reason': reason}

    # ------------------------------------------------------------------
    def _check_bench_boost(self, bench, current_gw, mean_starter_xp):
        available, status = self._is_chip_available('bboost', current_gw)
        if not available:
            return self._used('Bench Boost', status)

        if bench is None or len(bench) == 0 or mean_starter_xp <= 0:
            return {'chip': 'Bench Boost', 'recommendation': 'Save', 'icon': '💾',
                    'reason': "Not enough data to evaluate the bench."}

        bench_xp = float(bench['predicted_points'].sum())
        ratio = (bench_xp / len(bench)) / mean_starter_xp

        if ratio >= BB_RECOMMEND_RATIO:
            return {'chip': 'Bench Boost', 'recommendation': 'Recommended', 'icon': '🔥',
                    'reason': f"Strong bench: {bench_xp:.1f} XP, {ratio:.0%} of a typical starter."}
        if ratio >= BB_CONSIDER_RATIO:
            return {'chip': 'Bench Boost', 'recommendation': 'Consider', 'icon': '🤔',
                    'reason': f"Decent bench: {bench_xp:.1f} XP ({ratio:.0%} of a starter)."}
        return {'chip': 'Bench Boost', 'recommendation': 'Save', 'icon': '💾',
                'reason': f"Bench too weak: {bench_xp:.1f} XP ({ratio:.0%} of a starter)."}

    def _check_triple_captain(self, team, current_gw, mean_starter_xp):
        available, status = self._is_chip_available('3xc', current_gw)
        if not available:
            return self._used('Triple Captain', status)

        if team is None or len(team) == 0 or mean_starter_xp <= 0:
            return {'chip': 'Triple Captain', 'recommendation': 'Save', 'icon': '💾',
                    'reason': "Not enough data to evaluate a captain."}

        # Rank by captaincy_score where available: it discounts rotation risk and a hard
        # fixture, both of which matter more when the score is doubled.
        rank_col = 'captaincy_score' if 'captaincy_score' in team.columns else 'predicted_points'
        top_player = team.loc[team[rank_col].idxmax()]
        xp = float(top_player['predicted_points'])
        ratio = xp / mean_starter_xp

        if ratio >= TC_RECOMMEND_RATIO:
            return {'chip': 'Triple Captain', 'recommendation': 'Recommended', 'icon': '🔥',
                    'reason': f"{top_player['web_name']} projects {xp:.1f} XP — "
                              f"{ratio:.1f}x a typical starter."}
        if ratio >= TC_CONSIDER_RATIO:
            return {'chip': 'Triple Captain', 'recommendation': 'Consider', 'icon': '🤔',
                    'reason': f"{top_player['web_name']} at {xp:.1f} XP ({ratio:.1f}x a starter)."}
        return {'chip': 'Triple Captain', 'recommendation': 'Save', 'icon': '💾',
                'reason': f"No standout captain ({top_player['web_name']} "
                          f"only {ratio:.1f}x a typical starter)."}

    def _check_wildcard(self, current_gw, diff, current_xi_xp):
        available, status = self._is_chip_available('wildcard', current_gw)
        if not available:
            return self._used('Wildcard', status)

        ratio = (diff / current_xi_xp) if current_xi_xp > 0 else 0.0

        if ratio >= WC_RECOMMEND_RATIO:
            return {'chip': 'Wildcard', 'recommendation': 'Recommended', 'icon': '🔥',
                    'reason': f"Huge upgrade available: +{diff:.1f} XI points ({ratio:+.0%})."}
        if ratio >= WC_CONSIDER_RATIO:
            return {'chip': 'Wildcard', 'recommendation': 'Consider', 'icon': '🤔',
                    'reason': f"Worthwhile upgrade: +{diff:.1f} XI points ({ratio:+.0%})."}
        return {'chip': 'Wildcard', 'recommendation': 'Save', 'icon': '💾',
                'reason': f"Squad is already close to optimal (only +{diff:.1f} XI points)."}

    def _check_freehit(self, current_gw, diff, active_players, current_xi_xp):
        available, status = self._is_chip_available('freehit', current_gw)
        if not available:
            return self._used('Free Hit', status)

        if active_players < FH_CRISIS_ACTIVE_PLAYERS:
            return {'chip': 'Free Hit', 'recommendation': 'Recommended', 'icon': '🚨',
                    'reason': f"Crisis! Only {active_players} players projected to feature."}

        ratio = (diff / current_xi_xp) if current_xi_xp > 0 else 0.0
        if ratio >= FH_RECOMMEND_RATIO:
            return {'chip': 'Free Hit', 'recommendation': 'Recommended', 'icon': '🔥',
                    'reason': f"One-week punt: +{diff:.1f} XI points ({ratio:+.0%})."}
        return {'chip': 'Free Hit', 'recommendation': 'Save', 'icon': '💾',
                'reason': f"No need — you have a full squad ({active_players} players active)."}
