import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.fpl import FPLClient
from src.api.understat import UnderstatClient
from src.features.processor import FeatureProcessor
from src.model.predictor import PointsPredictor
from src.optimization.solver import TransferOptimizer
from src.interface.reporter import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="FPL AI Engine")
    parser.add_argument("--gw", type=int, default=1, help="Gameweek to analyze")
    parser.add_argument("--fetch", action="store_true", help="Fetch new data")
    parser.add_argument("--team_id", type=int, help="FPL Team ID to optimize for")
    args = parser.parse_args()

    # 1. Fetch Data
    if args.fetch:
        print("Fetching FPL data...")
        fpl = FPLClient()
        fpl.get_bootstrap_static()
        fpl.get_fixtures()
        
        print("Fetching Understat data...")
        us = UnderstatClient()
        df_us = us.get_player_stats()
        if df_us is not None:
            df_us.to_csv("data/raw/understat_players.csv", index=False)

    # 2. Process Features
    print("Processing features...")
    processor = FeatureProcessor()
    df_features = processor.process()
    
    if df_features is None:
        print("No features generated. Exiting.")
        return

    # 3. Predict Points
    print("Predicting points...")
    predictor = PointsPredictor()
    # If we had history, we'd train here. For now we run predict heuristics.
    df_scored = predictor.predict(df_features)

    # 4. Optimize
    print("Optimizing team...")
    optimizer = TransferOptimizer(budget=100.0)
    
    transfers_made = []
    
    if args.team_id:
        print(f"Fetching squad for Team ID: {args.team_id}")
        fpl = FPLClient()
        # Fetch picks for previous GW to know current squad
        picks_data = fpl.get_team_picks(args.team_id, args.gw)
        
        if picks_data:
            current_ids = [p['element'] for p in picks_data['picks']]
            print(f"Current squad loaded ({len(current_ids)} players).")
            
            # Calculate current team cost
            current_team_cost = df_scored[df_scored['id'].isin(current_ids)]['price'].sum()
            print(f"Current Team Value: £{current_team_cost:.1f}m")
            
            # Set budget to current cost + a buffer (e.g. 0.5m assumed bank? or just use cost)
            # Safe approach: budget = current_cost. If we want to allow using bank, we need input.
            # We'll use max(100.0, current_team_cost) to be safe for high value teams.
            optimizer.budget = max(100.0, current_team_cost)
            
            best_team = optimizer.recommend_transfers(df_scored, current_ids, free_transfers=1)
            
            # Identify transfers
            if best_team is not None:
                new_ids = best_team['id'].tolist()
                params_out = [pid for pid in current_ids if pid not in new_ids]
                params_in = [pid for pid in new_ids if pid not in current_ids]
                
                # Pair by position if possible
                # Create maps {pid: type}
                type_map = df_scored.set_index('id')['element_type'].to_dict()
                name_map = df_scored.set_index('id')['web_name'].to_dict() # Use map for speed
                
                # Sort both lists by element_type to align roughly
                params_out.sort(key=lambda x: type_map.get(x, 0))
                params_in.sort(key=lambda x: type_map.get(x, 0))
                
                for i in range(max(len(params_out), len(params_in))):
                    out_name = name_map[params_out[i]] if i < len(params_out) else "-"
                    in_name = name_map[params_in[i]] if i < len(params_in) else "-"
                    transfers_made.append((out_name, in_name))
        else:
            print("Could not fetch team picks. optimization from scratch.")
            best_team = optimizer.solve_team(df_scored)

    else:
        # standard free hit
        best_team = optimizer.solve_team(df_scored)
    
    if best_team is None:
        print("Optimization failed.")
        return

    # 5. Report
    print("Generating report...")
    reporter = ReportGenerator()
    captain = best_team.sort_values('predicted_points', ascending=False).iloc[0]['web_name']
    
    reporter.generate(args.gw, best_team, transfers=transfers_made, captain=captain)
    
    print("Done.")

if __name__ == "__main__":
    main()
