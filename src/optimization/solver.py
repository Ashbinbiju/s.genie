import pulp
import pandas as pd
import os

class TransferOptimizer:
    def __init__(self, budget=100.0):
        self.budget = budget

    def solve_team(self, df, current_team_ids=None):
        """
        Selects the best 15 players (11 starters + 4 bench) to maximize points.
        Constraints:
        - Budget <= 100.0
        - GK=2, DEF=5, MID=5, FW=3
        - Max 3 players per team
        """
        # Filter out invalid rows (zero prices etc)
        df = df[df['price'] > 0]
        players = df.index.tolist()
        
        # Decision variable: 1 if player i is selected, 0 otherwise
        prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("player", players, 0, 1, pulp.LpBinary)
        
        # Objective: Maximize total predicted points
        prob += pulp.lpSum([df.loc[i, 'predicted_points'] * x[i] for i in players])
        
        # Constraint 1: Budget
        prob += pulp.lpSum([df.loc[i, 'price'] * x[i] for i in players]) <= self.budget
        
        # Constraint 2: Squad Size = 15
        prob += pulp.lpSum([x[i] for i in players]) == 15
        
        # Constraint 3: Positions
        # element_type: 1=GK, 2=DEF, 3=MID, 4=FWD
        prob += pulp.lpSum([x[i] for i in players if df.loc[i, 'element_type'] == 1]) == 2
        prob += pulp.lpSum([x[i] for i in players if df.loc[i, 'element_type'] == 2]) == 5
        prob += pulp.lpSum([x[i] for i in players if df.loc[i, 'element_type'] == 3]) == 5
        prob += pulp.lpSum([x[i] for i in players if df.loc[i, 'element_type'] == 4]) == 3
        
        # Constraint 4: Max 3 per team
        teams = df['team'].unique()
        for t in teams:
            prob += pulp.lpSum([x[i] for i in players if df.loc[i, 'team'] == t]) <= 3

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob.status] == 'Optimal':
            selected_indices = [i for i in players if x[i].value() == 1.0]
            selected_df = df.loc[selected_indices].copy()
            return selected_df
        else:
            print("No optimal solution found.")
            return None

    def recommend_transfers(self, df_all, current_team_ids, free_transfers=1, cost_per_hit=4):
        """
        Suggests transfers to maximize (Predicted Points - Hit Costs).
        """
        df = df_all[df_all['price'] > 0].copy()
        players = df.index.tolist()
        
        # Current team mapping
        # 1 if in current team (matching by FPL ID), 0 otherwise
        in_current = {i: 1 if df.loc[i, 'id'] in current_team_ids else 0 for i in players}
        
        # Decision variable: 1 if player i is selected in NEW team
        prob = pulp.LpProblem("FPL_Transfer_Optimization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("player", players, 0, 1, pulp.LpBinary)
        
        # Variables for transfers
        # We can detect transfer OUT if in_current[i]=1 and x[i]=0
        # We can detect transfer IN if in_current[i]=0 and x[i]=1
        # Number of transfers = Sum(x[i] for i where in_current[i]=0)
        
        # But to punish hits, we need a linearized term.
        # Transfers made = Sum of new players
        # Hit Cost = max(0, (Transfers Made - FT)) * 4
        # This is non-linear.
        
        # Simplification for this battle-tested engine:
        # We will solve for k transfers (0, 1, 2, 3...) separately and pick the best NET score.
        
        best_solution = None
        best_net_score = -9999
        
        # Try making 0 to 3 transfers (hard cap to avoid churning entire team)
        team_size = 15
        
        for k in range(0, 4):
            # re-initialize problem for specific k transfers
            prob_k = pulp.LpProblem(f"FPL_Transfers_{k}", pulp.LpMaximize)
            
            # Objectives & Constraints same as solve_team
            prob_k += pulp.lpSum([df.loc[i, 'predicted_points'] * x[i] for i in players])
            
            prob_k += pulp.lpSum([df.loc[i, 'price'] * x[i] for i in players]) <= self.budget
            prob_k += pulp.lpSum([x[i] for i in players]) == team_size
            
            # Positions
            prob_k += pulp.lpSum([x[i] for i in players if df.loc[i, 'element_type'] == 1]) == 2
            prob_k += pulp.lpSum([x[i] for i in players if df.loc[i, 'element_type'] == 2]) == 5
            prob_k += pulp.lpSum([x[i] for i in players if df.loc[i, 'element_type'] == 3]) == 5
            prob_k += pulp.lpSum([x[i] for i in players if df.loc[i, 'element_type'] == 4]) == 3
            
            # Teams
            teams = df['team'].unique()
            for t in teams:
                prob_k += pulp.lpSum([x[i] for i in players if df.loc[i, 'team'] == t]) <= 3
                
            # Transfer Constraint: Exactly k transfers
            # New players count == k
            # Sum(x[i] for i where not in current) == k
            prob_k += pulp.lpSum([x[i] for i in players if in_current[i] == 0]) == k
            
            # Force keep (15-k) players from current team
            # prob_k += pulp.lpSum([x[i] for i in players if in_current[i] == 1]) == (team_size - k) 
            
            # Check feasibility of keeping (15-k) players
            valid_current_count = sum([1 for i in players if in_current[i] == 1])
            if valid_current_count < (team_size - k):
                print(f"Skipping k={k}: Only {valid_current_count} valid players in current team (need {team_size-k}).")
                continue

            prob_k.solve(pulp.PULP_CBC_CMD(msg=0))
            
            status = pulp.LpStatus[prob_k.status]
            if status == 'Optimal':
                score = pulp.value(prob_k.objective)
                hits_taken = max(0, k - free_transfers)
                penalty = hits_taken * cost_per_hit
                net_score = score - penalty
                
                print(f"Transfers: {k} | Pred Points: {score:.1f} | Hits: {hits_taken} | Net: {net_score:.1f}")
                
                if net_score > best_net_score:
                    best_net_score = net_score
                    selected_indices = [i for i in players if x[i].value() == 1.0]
                    best_solution = df.loc[selected_indices].copy()
            else:
                print(f"k={k} Infeasible status: {status}")
            
        return best_solution

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/player_features.parquet")
    # Quick mock predict if not valid
    if 'predicted_points' not in df.columns:
        df['predicted_points'] = df['form'] # fallback
        
    optimizer = TransferOptimizer(budget=100.0)
    best_team = optimizer.solve_team(df)
    
    if best_team is not None:
        print("Best Team:")
        print(best_team[['web_name', 'element_type', 'team', 'price', 'predicted_points']])
        print(f"Total Points: {best_team['predicted_points'].sum()}")
        print(f"Total Cost: {best_team['price'].sum()}")
