import pulp
import pandas as pd

# element_type: 1=GK, 2=DEF, 3=MID, 4=FWD
SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
SQUAD_SIZE = 15
MAX_PER_CLUB = 3
MAX_TRANSFERS_CONSIDERED = 3

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

    def _add_squad_constraints(self, prob, x, players, lk):
        """Budget, squad size, positional quotas and the max-per-club rule."""
        prob += pulp.lpSum([lk['price'][i] * x[i] for i in players]) <= self.budget
        prob += pulp.lpSum([x[i] for i in players]) == SQUAD_SIZE

        for etype, quota in SQUAD_QUOTA.items():
            prob += pulp.lpSum([x[i] for i in players if lk['etype'][i] == etype]) == quota

        by_team = {}
        for i in players:
            by_team.setdefault(lk['team'][i], []).append(i)
        for team_players in by_team.values():
            prob += pulp.lpSum([x[i] for i in team_players]) <= MAX_PER_CLUB

    def solve_team(self, df, current_team_ids=None):
        """
        Selects the best 15 players (11 starters + 4 bench) to maximize points.

        Constraints: budget, GK=2 DEF=5 MID=5 FWD=3, max 3 players per club.
        """
        df = df[df['price'] > 0]
        if df.empty:
            print("No priced players available.")
            return None

        players = df.index.tolist()
        lk = self._lookups(df)

        prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("player", players, 0, 1, pulp.LpBinary)

        prob += pulp.lpSum([lk['points'][i] * x[i] for i in players])
        self._add_squad_constraints(prob, x, players, lk)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != 'Optimal':
            print(f"No optimal solution found (status: {pulp.LpStatus[prob.status]}).")
            return None

        selected = [i for i in players if x[i].value() == 1.0]
        return df.loc[selected].copy()

    def recommend_transfers(self, df_all, current_team_ids, free_transfers=1, cost_per_hit=4):
        """
        Suggests transfers maximizing (predicted points - hit costs).

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

            prob_k = pulp.LpProblem(f"FPL_Transfers_{k}", pulp.LpMaximize)
            x = pulp.LpVariable.dicts(f"player_k{k}", players, 0, 1, pulp.LpBinary)

            prob_k += pulp.lpSum([lk['points'][i] * x[i] for i in players])
            self._add_squad_constraints(prob_k, x, players, lk)
            prob_k += pulp.lpSum([x[i] for i in incoming]) == k

            prob_k.solve(pulp.PULP_CBC_CMD(msg=0))

            status = pulp.LpStatus[prob_k.status]
            if status != 'Optimal':
                print(f"k={k} infeasible (status: {status})")
                continue

            score = pulp.value(prob_k.objective)
            hits_taken = max(0, k - free_transfers)
            net_score = score - hits_taken * cost_per_hit

            print(f"Transfers: {k} | Pred Points: {score:.1f} | Hits: {hits_taken} | "
                  f"Net: {net_score:.1f}")

            if net_score > best_net_score + NET_GAIN_MARGIN:
                best_net_score = net_score
                selected = [i for i in players if x[i].value() == 1.0]
                best_solution = df.loc[selected].copy()

        return best_solution


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/player_features.parquet")
    if 'predicted_points' not in df.columns:
        df['predicted_points'] = df['form'].fillna(0)

    optimizer = TransferOptimizer(budget=100.0)
    best_team = optimizer.solve_team(df)

    if best_team is not None:
        print("Best Team:")
        print(best_team[['web_name', 'element_type', 'team', 'price', 'predicted_points']])
        print(f"Total Points: {best_team['predicted_points'].sum():.1f}")
        print(f"Total Cost: {best_team['price'].sum():.1f}")
