import pulp
import pandas as pd

# element_type: 1=GK, 2=DEF, 3=MID, 4=FWD
SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
SQUAD_SIZE = 15
XI_SIZE = 11
MAX_PER_CLUB = 3
MAX_TRANSFERS_CONSIDERED = 3

# Legal starting-XI shape.
FORMATION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
FORMATION_MAX = {1: 1, 2: 5, 3: 5, 4: 3}

# What a benched player is worth relative to a starter.
#
# You do not score your bench: its points only materialise through auto-subs when a
# starter blanks, or under Bench Boost. Counting bench points at full value — as this
# used to — makes the optimizer buy fifteen good-value players instead of a strong XI
# with cheap cover, which systematically prices out premiums.
BENCH_WEIGHT = 0.15

# An extra transfer must beat the incumbent plan by more than this to be worth making.
# Point predictions carry roughly this much noise, so churning the squad for a smaller
# margin trades a real, banked free transfer for a difference the model cannot resolve.
# k is evaluated in ascending order, so ties keep the FEWER-transfer plan.
NET_GAIN_MARGIN = 0.5


class TransferOptimizer:
    def __init__(self, budget=100.0):
        self.budget = budget

    @staticmethod
    def _lookups(df):
        """
        Pre-extract columns to plain dicts.

        Constraint building indexes each player's price/type/team several times; doing
        that with df.loc[i, col] runs tens of thousands of scalar lookups per solve.
        """
        return {
            'points': df['predicted_points'].to_dict(),
            'price': df['price'].to_dict(),
            'etype': df['element_type'].to_dict(),
            'team': df['team'].to_dict(),
            'id': df['id'].to_dict(),
        }

    @staticmethod
    def _build(players, lk, budget, name):
        """
        The FPL squad problem.

        Three sets of binaries rather than one:
          x[i] — in the 15-man squad
          y[i] — in the starting XI
          c[i] — captain

        The objective is what you ACTUALLY SCORE: the starting XI, plus the captain's
        points a second time, plus a discounted contribution from the bench. Optimising
        the flat 15-man total instead (the previous formulation) both ignores the
        captain's doubling and treats bench points as if they counted, so it never
        pays up for a premium.
        """
        prob = pulp.LpProblem(name, pulp.LpMaximize)
        x = pulp.LpVariable.dicts(f"sq_{name}", players, 0, 1, pulp.LpBinary)
        y = pulp.LpVariable.dicts(f"xi_{name}", players, 0, 1, pulp.LpBinary)
        c = pulp.LpVariable.dicts(f"cp_{name}", players, 0, 1, pulp.LpBinary)

        prob += pulp.lpSum([
            lk['points'][i] * (y[i] + c[i] + BENCH_WEIGHT * (x[i] - y[i]))
            for i in players
        ])

        # --- squad (15) ---
        prob += pulp.lpSum([lk['price'][i] * x[i] for i in players]) <= budget
        prob += pulp.lpSum([x[i] for i in players]) == SQUAD_SIZE
        for etype, quota in SQUAD_QUOTA.items():
            prob += pulp.lpSum([x[i] for i in players if lk['etype'][i] == etype]) == quota

        by_team = {}
        for i in players:
            by_team.setdefault(lk['team'][i], []).append(i)
        for team_players in by_team.values():
            prob += pulp.lpSum([x[i] for i in team_players]) <= MAX_PER_CLUB

        # --- starting XI (11), a legal formation drawn from the squad ---
        prob += pulp.lpSum([y[i] for i in players]) == XI_SIZE
        for etype in SQUAD_QUOTA:
            in_pos = [y[i] for i in players if lk['etype'][i] == etype]
            prob += pulp.lpSum(in_pos) >= FORMATION_MIN[etype]
            prob += pulp.lpSum(in_pos) <= FORMATION_MAX[etype]
        for i in players:
            prob += y[i] <= x[i]

        # --- captain: exactly one, and he must be starting ---
        prob += pulp.lpSum([c[i] for i in players]) == 1
        for i in players:
            prob += c[i] <= y[i]

        return prob, x, y, c

    @staticmethod
    def _extract(df, players, x, y, c):
        """Selected squad, annotated with who starts and who wears the armband."""
        chosen = [i for i in players if x[i].value() == 1.0]
        squad = df.loc[chosen].copy()
        squad['is_starter'] = [y[i].value() == 1.0 for i in chosen]
        squad['is_captain'] = [c[i].value() == 1.0 for i in chosen]
        return squad

    def solve_team(self, df, current_team_ids=None, must_include=None, verbose=True):
        """
        Selects the best 15 (11 starters + 4 bench) to maximize points actually scored.

        Constraints: budget, GK=2 DEF=5 MID=5 FWD=3, max 3 players per club, and a
        legal starting XI with a captain.

        `must_include` is a collection of FPL player ids to force into the squad — used
        to answer "what would picking X actually cost me?".
        """
        df = df[df['price'] > 0]
        if df.empty:
            if verbose:
                print("No priced players available.")
            return None

        players = df.index.tolist()
        lk = self._lookups(df)
        prob, x, y, c = self._build(players, lk, self.budget, "squad")

        for pid in (must_include or []):
            forced = [i for i in players if lk['id'][i] == pid]
            if not forced:
                if verbose:
                    print(f"Cannot force player id {pid}: not in the pool.")
                return None
            prob += x[forced[0]] == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != 'Optimal':
            if verbose:
                print(f"No optimal solution found (status: {pulp.LpStatus[prob.status]}).")
            return None

        return self._extract(df, players, x, y, c)

    @staticmethod
    def squad_score(squad):
        """Points a squad actually scores: the XI, with the captain counted twice."""
        if squad is None or squad.empty:
            return 0.0
        starters = squad[squad['is_starter']] if 'is_starter' in squad.columns else squad
        total = float(starters['predicted_points'].sum())
        if 'is_captain' in squad.columns and squad['is_captain'].any():
            total += float(squad[squad['is_captain']].iloc[0]['predicted_points'])
        return total

    def explain_exclusion(self, df, player_id, baseline=None):
        """
        Why a given player is (or is not) in the optimal squad.

        Returns a dict with the squad built around them, what it costs versus the
        unconstrained optimum, and who makes way. A player being left out is usually a
        VALUE judgement rather than a low rating, and that distinction is invisible from
        the team sheet alone.
        """
        row = df[df['id'] == player_id]
        if row.empty:
            return None
        player = row.iloc[0]

        if baseline is None:
            baseline = self.solve_team(df, verbose=False)
        if baseline is None:
            return None

        in_squad = player_id in set(baseline['id'])
        result = {
            'player': player,
            'in_squad': in_squad,
            'baseline_score': self.squad_score(baseline),
            'forced_score': None,
            'cost': 0.0,
            'forced_squad': None,
            'displaced': [],
            'is_captain': False,
        }

        if in_squad:
            picked = baseline[baseline['id'] == player_id].iloc[0]
            result['is_captain'] = bool(picked.get('is_captain', False))
            result['forced_score'] = result['baseline_score']
            return result

        forced = self.solve_team(df, must_include=[player_id], verbose=False)
        if forced is None:
            result['cost'] = float('inf')
            return result

        result['forced_squad'] = forced
        result['forced_score'] = self.squad_score(forced)
        result['cost'] = result['baseline_score'] - result['forced_score']
        result['displaced'] = (
            baseline[~baseline['id'].isin(set(forced['id']))]
            .sort_values('predicted_points', ascending=False)
        )
        return result

    def recommend_transfers(self, df_all, current_team_ids, free_transfers=1, cost_per_hit=4):
        """
        Suggests transfers maximizing (points scored - hit costs).

        The hit cost max(0, k - free_transfers) * 4 is non-linear, so rather than
        linearising it we solve a separate program for exactly k transfers
        (k = 0..MAX_TRANSFERS_CONSIDERED) and keep the best NET score. k is capped to
        stop the solver churning the whole squad for a marginal gain.
        """
        df = df_all[df_all['price'] > 0].copy()
        if df.empty:
            print("No priced players available.")
            return None

        players = df.index.tolist()
        lk = self._lookups(df)

        current_set = set(current_team_ids)
        incoming = [i for i in players if lk['id'][i] not in current_set]
        retained_count = len(players) - len(incoming)

        best_solution = None
        best_net_score = float('-inf')

        for k in range(0, MAX_TRANSFERS_CONSIDERED + 1):
            # Infeasible if we cannot field (15 - k) players we already own — e.g. when
            # a squad member has left the league and is absent from the player list.
            if retained_count < (SQUAD_SIZE - k):
                print(f"Skipping k={k}: only {retained_count} owned players available "
                      f"(need {SQUAD_SIZE - k}).")
                continue

            prob, x, y, c = self._build(players, lk, self.budget, f"tx{k}")
            prob += pulp.lpSum([x[i] for i in incoming]) == k

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            status = pulp.LpStatus[prob.status]
            if status != 'Optimal':
                print(f"k={k} infeasible (status: {status})")
                continue

            score = pulp.value(prob.objective)
            hits_taken = max(0, k - free_transfers)
            net_score = score - hits_taken * cost_per_hit

            print(f"Transfers: {k} | XI+captain: {score:.1f} | Hits: {hits_taken} | "
                  f"Net: {net_score:.1f}")

            if net_score > best_net_score + NET_GAIN_MARGIN:
                best_net_score = net_score
                best_solution = self._extract(df, players, x, y, c)

        return best_solution


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/player_features.parquet")
    if 'predicted_points' not in df.columns:
        df['predicted_points'] = df['form'].fillna(0)

    optimizer = TransferOptimizer(budget=100.0)
    best_team = optimizer.solve_team(df)

    if best_team is not None:
        print("Best Team:")
        print(best_team[['web_name', 'element_type', 'team', 'price',
                         'predicted_points', 'is_starter', 'is_captain']])
        print(f"Total Cost: {best_team['price'].sum():.1f}")
