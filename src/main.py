import argparse
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.api.fpl import FPLClient
from src.api.understat import UnderstatClient
from src.api.async_fpl import refresh_cache
from src.features.processor import FeatureProcessor
from src.model.predictor import PointsPredictor
from src.optimization.solver import TransferOptimizer
from src.optimization.team_selection import select_starting_xi, pick_captain
from src.interface.reporter import ReportGenerator
from src.utils.season import load_bootstrap, get_next_gw, get_season_label


def main():
    parser = argparse.ArgumentParser(description="FPL AI Engine")
    parser.add_argument("--gw", type=int, default=None,
                        help="Gameweek to analyze (defaults to the next deadline)")
    parser.add_argument("--fetch", action="store_true", help="Fetch new data")
    parser.add_argument("--team_id", type=int, help="FPL Team ID to optimize for")
    parser.add_argument("--budget", type=float, default=100.0, help="Budget in £m")
    parser.add_argument("--bank", type=float, default=0.0, help="Money in the bank, £m")
    args = parser.parse_args()

    fpl = FPLClient()

    # 1. Fetch Data
    if args.fetch:
        print("Fetching FPL data...")
        fpl.get_bootstrap_static()
        fpl.get_fixtures()

        print("Fetching Understat data...")
        us = UnderstatClient()
        df_us = us.get_player_stats()
        if df_us is not None:
            os.makedirs("data/raw", exist_ok=True)
            df_us.to_csv("data/raw/understat_players.csv", index=False)

        # The element-summary cache backs every rolling feature; without it the
        # predictor silently falls back to a heuristic.
        print("Refreshing player summary cache...")
        refresh_cache()

    static = load_bootstrap()
    gw = args.gw if args.gw is not None else get_next_gw(static)
    print(f"Season {get_season_label(static)} — analyzing GW{gw}")

    # 2. Process Features
    print("Processing features...")
    df_features = FeatureProcessor().process()
    if df_features is None:
        print("No features generated. Exiting.")
        return

    # 3. Predict Points
    print("Predicting points...")
    predictor = PointsPredictor()
    df_scored = predictor.predict(df_features)
    if predictor.prediction_mode == "fallback":
        print("WARNING: running on the heuristic fallback, not the ML model.")

    # 4. Optimize
    print("Optimizing team...")
    optimizer = TransferOptimizer(budget=args.budget)
    transfers_made = []
    best_team = None

    if args.team_id:
        print(f"Fetching squad for Team ID: {args.team_id}")
        # Skip Free Hit gameweeks: the API returns the temporary FH squad for those,
        # which would leak into the recommendation as if it were the permanent team.
        history = fpl.get_history(args.team_id)
        freehit_gws = fpl.get_freehit_gws(args.team_id, history)
        picks_data = fpl.get_team_picks(args.team_id, gw, freehit_gws=freehit_gws)

        if picks_data:
            current_ids = [p['element'] for p in picks_data['picks']]
            current_team = df_scored[df_scored['id'].isin(current_ids)]
            current_cost = current_team['price'].sum()
            print(f"Current squad loaded ({len(current_team)}/{len(current_ids)} players).")
            print(f"Current Team Value: £{current_cost:.1f}m")

            optimizer.budget = max(args.budget, current_cost + args.bank)
            free_transfers = fpl.calculate_free_transfers(args.team_id, gw, history)
            print(f"Free transfers available: {free_transfers}")

            best_team = optimizer.recommend_transfers(
                df_scored, current_ids, free_transfers=free_transfers)

            if best_team is not None:
                new_ids = best_team['id'].tolist()
                out_ids = [pid for pid in current_ids if pid not in new_ids]
                in_ids = [pid for pid in new_ids if pid not in current_ids]

                type_map = df_scored.set_index('id')['element_type'].to_dict()
                name_map = df_scored.set_index('id')['web_name'].to_dict()

                out_ids.sort(key=lambda x: type_map.get(x, 0))
                in_ids.sort(key=lambda x: type_map.get(x, 0))

                for i in range(max(len(out_ids), len(in_ids))):
                    # .get, not [] — a player who left the league is absent from the
                    # current player list and would otherwise raise KeyError.
                    out_name = name_map.get(out_ids[i], "?") if i < len(out_ids) else "-"
                    in_name = name_map.get(in_ids[i], "?") if i < len(in_ids) else "-"
                    transfers_made.append((out_name, in_name))
        else:
            print("No squad history available — optimizing from scratch.")
            best_team = optimizer.solve_team(df_scored)
    else:
        best_team = optimizer.solve_team(df_scored)

    if best_team is None:
        print("Optimization failed.")
        return

    # 5. Report
    print("Generating report...")
    starters, _ = select_starting_xi(best_team)
    captain, _ = pick_captain(starters)

    ReportGenerator().generate(
        gw, best_team,
        transfers=transfers_made,
        captain=captain['web_name'] if captain is not None else None,
        starters=starters,
    )

    print("Done.")


if __name__ == "__main__":
    main()
