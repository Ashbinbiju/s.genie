import pandas as pd
import numpy as np
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.season import (
    load_bootstrap, team_id_to_name, team_id_to_code,
    canon_team, ELEMENT_TYPE_TO_POSITION,
)
from src.utils.names import normalize_name_series

# Columns the cached parquet must contain to be considered current. Anything added to
# the feature set below must be added here too, or a stale cache will be served and the
# missing columns will be silently zero-filled downstream.
REQUIRED_CACHE_COLS = [
    'next_opponent', 'news', 'fixture_difficulty', 'photo', 'team_name', 'opponent_name',
    'win_prob', 'team_implied_goals', 'clean_sheet_prob',
    'anytime_goal_scorer_prob', 'next_kickoff_time',
]


class FeatureProcessor:
    def __init__(self, data_dir="data"):
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_fpl_data(self):
        """Loads FPL bootstrap static data."""
        static = load_bootstrap(os.path.join(self.raw_dir, "bootstrap_static.json"))
        if static is None:
            print("FPL data not found. Run FPLClient first.")
            return None, None, None
        return (
            pd.DataFrame(static['elements']),
            pd.DataFrame(static['teams']),
            pd.DataFrame(static['events']),
        )

    def load_understat_data(self):
        """Loads Understat data."""
        path = os.path.join(self.raw_dir, "understat_players.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def process(self, force_refresh=False):
        output_path = os.path.join(self.processed_dir, "player_features.parquet")

        if not force_refresh and os.path.exists(output_path):
            df = pd.read_parquet(output_path)
            missing = [c for c in REQUIRED_CACHE_COLS if c not in df.columns]
            if not missing:
                return df
            print(f"Cached data missing {len(missing)} column(s) {missing[:4]}... Regenerating.")

        print("Processing features...")
        fpl_players, fpl_teams, _ = self.load_fpl_data()
        understat_players = self.load_understat_data()

        if fpl_players is None:
            return None

        if understat_players is not None:
            # Fold accents rather than stripping them. Stripping turns Understat's
            # "Ødegaard" into "degaard" while FPL's "Odegaard" becomes "odegaard", so
            # every player with a stroked or ligatured letter silently fails to match.
            fpl_players['web_name_norm'] = normalize_name_series(fpl_players['web_name'])
            understat_players['player_name_norm'] = normalize_name_series(
                understat_players['player_name'])

            # Collapse Understat duplicates BEFORE merging. Two Understat players can
            # normalise to the same key (accents/short names), which would otherwise
            # duplicate the FPL row — letting the optimizer pick the same player twice
            # and double-counting squad value.
            # Keep the highest-minutes claimant on a collision; tolerate the column
            # being absent, since the Understat payload shape is not guaranteed.
            if 'time' in understat_players.columns:
                understat_players = understat_players.sort_values(
                    'time', ascending=False, key=lambda s: pd.to_numeric(s, errors='coerce'))
            understat_players = understat_players.drop_duplicates(
                subset=['player_name_norm'], keep='first')

            merged = pd.merge(
                fpl_players, understat_players,
                left_on='web_name_norm', right_on='player_name_norm',
                how='left', suffixes=('_fpl', '_us'),
            )
            matched = merged['player_name_norm'].notna().sum()
            print(f"  Understat: matched {matched}/{len(fpl_players)} players "
                  f"({matched / max(len(fpl_players), 1) * 100:.0f}%)")
        else:
            print("  WARNING: Understat data missing (data/raw/understat_players.csv). "
                  "xG/xA features will be ZERO for every player — run "
                  "`python src/api/understat.py` to populate them.")
            merged = fpl_players.copy()
            for col in ['xG', 'xA', 'time']:
                merged[col] = 0

        if 'id' not in merged.columns and 'id_fpl' in merged.columns:
            merged['id'] = merged['id_fpl']

        # Guard against any residual row duplication — the solver indexes by row and
        # would happily select the same player twice.
        before = len(merged)
        merged = merged.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
        if len(merged) != before:
            print(f"  Dropped {before - len(merged)} duplicate player row(s) after merge.")

        # 1. Price
        merged['price'] = merged['now_cost'] / 10.0

        # 2. Form (FPL form is a string)
        merged['form'] = pd.to_numeric(merged['form'], errors='coerce')

        # 3. xG/xA per 90
        if 'time' in merged.columns and 'xG' in merged.columns:
            minutes = pd.to_numeric(merged['time'], errors='coerce').fillna(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                merged['xG_per_90'] = merged['xG'] / (minutes / 90)
                merged['xA_per_90'] = merged['xA'] / (minutes / 90)
            for c in ['xG_per_90', 'xA_per_90']:
                merged[c] = merged[c].replace([np.inf, -np.inf], 0).fillna(0)
        else:
            merged['xG_per_90'] = 0
            merged['xA_per_90'] = 0

        # 4. Minutes Probability (proxy from 'chance_of_playing_next_round').
        # Coerce first: FPL sends null for every unflagged player, which at the start of
        # a season is ALL of them, giving an object-dtype column that does not divide.
        merged['chance_of_playing_next_round'] = pd.to_numeric(
            merged.get('chance_of_playing_next_round'), errors='coerce')
        merged['minutes_prob'] = merged['chance_of_playing_next_round'].fillna(100) / 100.0

        # 5. Canonical categoricals — MUST match history_builder's vocabulary.
        team_names = team_id_to_name_from_df(fpl_teams)
        merged['team_name'] = merged['team'].map(lambda t: canon_team(team_names.get(t, str(t))))
        merged['position'] = merged['element_type'].map(ELEMENT_TYPE_TO_POSITION).astype(str)

        # team_code drives shirt images; it comes straight from the API and is correct
        # for every club, including newly promoted ones.
        merged['team_code'] = merged['team'].map(fpl_teams.set_index('id')['code'].to_dict())

        # 6. Fixture Difficulty & Next Opponent
        fixtures = self.load_fixtures()
        if fixtures is not None:
            match_data = self.calculate_fixture_difficulty(fixtures, fpl_teams)
            merged['fixture_difficulty'] = merged['team'].map(
                lambda x: match_data.get(x, {}).get('fixture_difficulty', 3))
            merged['next_opponent'] = merged['team'].map(
                lambda x: match_data.get(x, {}).get('next_opponent', "-"))
            # Opponent as a NAME, not the per-season integer id — ids mean different
            # clubs in different seasons and would not match the training vocabulary.
            merged['opponent_name'] = merged['team'].map(
                lambda x: canon_team(team_names.get(
                    match_data.get(x, {}).get('opponent_team_id', 0), "UNKNOWN")))
            merged['was_home'] = merged['team'].map(
                lambda x: match_data.get(x, {}).get('is_home', True)).astype(str)
            # Needed so the predictor can compute days_rest against the UPCOMING match
            # rather than the gap between the last two completed ones.
            merged['next_kickoff_time'] = merged['team'].map(
                lambda x: match_data.get(x, {}).get('next_kickoff_time', None))
        else:
            merged['fixture_difficulty'] = 3
            merged['next_opponent'] = "-"
            merged['opponent_name'] = "UNKNOWN"
            merged['was_home'] = "True"
            merged['next_kickoff_time'] = None

        # ---------------------------------------------------------------
        # Bookmaker Odds (optional, fallback-safe)
        # ---------------------------------------------------------------
        odds_cols = ['win_prob', 'draw_prob', 'loss_prob',
                     'team_implied_goals', 'opponent_implied_goals', 'clean_sheet_prob']
        try:
            from src.api.odds import OddsClient, LEAGUE_DEFAULTS

            odds_client = OddsClient()
            live_odds = odds_client.get_current_odds()

            for col in odds_cols:
                merged[col] = LEAGUE_DEFAULTS[col]

            if live_odds:
                # Live odds are keyed by the bookmaker's team naming; canonicalise both
                # sides before looking up.
                live_canon = {canon_team(k): v for k, v in live_odds.items()}
                hits = 0
                for idx, row in merged.iterrows():
                    odds_data = live_canon.get(row['team_name'])
                    if odds_data:
                        hits += 1
                        for col in odds_cols:
                            merged.at[idx, col] = odds_data.get(col, LEAGUE_DEFAULTS[col])
                print(f"  Odds: live data applied to {hits}/{len(merged)} rows")
            else:
                print("  Odds: no live data, using league defaults "
                      "(set ODDS_API_KEY for live odds)")

            merged['anytime_goal_scorer_prob'] = merged.apply(
                lambda r: OddsClient.compute_anytime_scorer_prob(
                    r['team_implied_goals'], r['position']), axis=1)
        except Exception as e:
            print(f"Odds integration skipped: {e}")
            from src.api.odds import LEAGUE_DEFAULTS
            for col in odds_cols:
                merged[col] = LEAGUE_DEFAULTS[col]
            merged['anytime_goal_scorer_prob'] = LEAGUE_DEFAULTS['anytime_goal_scorer_prob']

        features = [
            'id', 'web_name', 'team', 'team_name', 'team_code', 'element_type', 'position',
            'price', 'form', 'points_per_game', 'ict_index', 'ep_next',
            'xG', 'xA', 'xG_per_90', 'xA_per_90', 'minutes_prob',
            'total_points', 'fixture_difficulty',
            'news', 'chance_of_playing_next_round', 'next_opponent', 'next_kickoff_time',
            'opponent_name', 'was_home',
            'win_prob', 'draw_prob', 'loss_prob',
            'team_implied_goals', 'opponent_implied_goals',
            'clean_sheet_prob', 'anytime_goal_scorer_prob',
            'photo',
        ]

        for f in features:
            if f not in merged.columns:
                merged[f] = 0

        final_df = merged[features].copy()

        cols_to_float = ['points_per_game', 'ict_index', 'ep_next', 'fixture_difficulty']
        for c in cols_to_float:
            final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0)

        final_df['news'] = final_df['news'].fillna("")
        final_df['next_opponent'] = final_df['next_opponent'].fillna("-")

        final_df.to_parquet(output_path)
        print(f"Saved processed features to {output_path} ({len(final_df)} players)")
        return final_df

    def load_fixtures(self):
        path = os.path.join(self.raw_dir, "fixtures.json")
        if os.path.exists(path):
            return pd.read_json(path)
        return None

    def calculate_fixture_difficulty(self, fixtures_df, teams_df, next_n=5):
        team_map = teams_df.set_index('id')['short_name'].to_dict()
        team_ids = teams_df['id'].tolist()

        future = fixtures_df[fixtures_df['finished'] == False].sort_values('kickoff_time')

        team_data = {}
        for team_id in team_ids:
            matches = future[
                (future['team_h'] == team_id) | (future['team_a'] == team_id)
            ].head(next_n)

            diff_sum = 0
            count = 0
            next_opp_str = "-"
            next_opp_id = 0
            next_is_home = True
            next_kickoff = None

            for i, (_, match) in enumerate(matches.iterrows()):
                is_home = (match['team_h'] == team_id)
                opp_id = match['team_a'] if is_home else match['team_h']
                opp_name = team_map.get(opp_id, "?")
                difficulty = match['team_h_difficulty'] if is_home else match['team_a_difficulty']

                if i == 0:
                    venue = "(H)" if is_home else "(A)"
                    next_opp_str = f"{opp_name} {venue}"
                    next_opp_id = opp_id
                    next_is_home = is_home
                    next_kickoff = match.get('kickoff_time')

                diff_sum += difficulty
                count += 1

            team_data[team_id] = {
                'fixture_difficulty': diff_sum / count if count > 0 else 3,
                'next_opponent': next_opp_str,
                'opponent_team_id': next_opp_id,
                'is_home': next_is_home,
                'next_kickoff_time': str(next_kickoff) if next_kickoff is not None else None,
            }

        return team_data


def team_id_to_name_from_df(teams_df):
    """{id: name} from a teams DataFrame (bootstrap `teams`)."""
    return teams_df.set_index('id')['name'].to_dict()


if __name__ == "__main__":
    processor = FeatureProcessor()
    processor.process(force_refresh=True)
