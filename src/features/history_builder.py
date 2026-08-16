import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is on sys.path for both standalone and package execution
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.season import (
    load_bootstrap, get_season_label, get_current_gw,
    team_id_to_name, canon_team, canon_position,
)
from src.api.async_fpl import cache_filename


class HistoryBuilder:
    def __init__(self, raw_dir="data/raw", cache_dir="data/cache", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        self.rolling_cols = [
            'minutes', 'total_points', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'bps', 'influence', 'creativity', 'threat', 'starts'
        ]

    @staticmethod
    def _opponent_name_map(df):
        """
        Build {opponent_team_id: club_name} for one season from the fixture pairings.

        Each fixture contains rows for both clubs; a row's `opponent_team` id therefore
        identifies the *other* club in that fixture, whose name is on the paired rows.
        """
        mapping = {}
        if 'fixture' not in df.columns:
            return mapping
        for _, group in df.groupby('fixture'):
            names = group['team_name'].dropna().unique()
            if len(names) != 2:
                continue
            for name in names:
                other = names[0] if names[1] == name else names[1]
                for opp_id in group.loc[group['team_name'] == name, 'opponent_team'].unique():
                    mapping.setdefault(opp_id, other)
        return mapping

    @classmethod
    def _resolve_opponent_names(cls, df):
        """
        Map each row's `opponent_team` id to a club name, loudly.

        Unmapped ids must never reach the categorical cast: `.map({})` yields NaN, and
        `.astype(str).astype('category')` then turns NaN into the literal category
        'nan'. That trains and predicts without raising while quietly destroying the
        opponent vocabulary — the single most damaging failure mode in this file.
        """
        mapping = cls._opponent_name_map(df)
        resolved = df['opponent_team'].map(mapping)

        unmapped = resolved.isna()
        if unmapped.any():
            missing_ids = sorted(df.loc[unmapped, 'opponent_team'].dropna().unique())[:8]
            print(f"  WARNING: {int(unmapped.sum())} row(s) have an unresolvable "
                  f"opponent id {missing_ids}; falling back to 'UNKNOWN'. "
                  f"Check that the source CSV has a 'fixture' column.")
            resolved = resolved.fillna("UNKNOWN")

        return resolved

    # ------------------------------------------------------------------
    # Historical seasons (vaastav)
    # ------------------------------------------------------------------
    def _load_vaastav_season(self, season):
        path = os.path.join(self.raw_dir, "vaastav", f"merged_gw_{season}.csv")
        if not os.path.exists(path):
            print(f"Vaastav data for {season} not found at {path}")
            return None

        df = pd.read_csv(path)

        # Drop existing GW if it exists to avoid duplication when renaming round
        if 'GW' in df.columns:
            df = df.drop(columns=['GW'])

        rename_map = {'round': 'GW', 'element': 'player_id', 'value': 'price'}
        df = df.rename(columns=rename_map)

        if 'price' in df.columns:
            df['price'] = df['price'] / 10.0

        # Canonical vocabularies — MUST match what the inference path produces.
        # `team` in vaastav is a club NAME; team ids are not stable across seasons so
        # the name is the only usable cross-season key.
        df['team_name'] = df['team'].map(canon_team)
        df['position'] = df['position'].map(canon_position)

        # `opponent_team` is a per-season integer id and has exactly the same
        # instability: id 1 is a different club in a different season, and different
        # again at inference. Resolve it to a club name.
        df['opponent_name'] = self._resolve_opponent_names(df)

        for col in self.rolling_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # Group by player and GW to collapse Double Gameweeks into a single row
        agg_dict = {col: 'sum' for col in self.rolling_cols}
        agg_dict.update({
            'price': 'mean',
            'was_home': 'first',
            'opponent_name': 'first',
            'team_name': 'first',
            'position': 'first',
            'kickoff_time': 'first',
        })

        df_grouped = df.groupby(['player_id', 'GW']).agg(agg_dict).reset_index()
        df_grouped['season'] = season

        return df_grouped

    # ------------------------------------------------------------------
    # Current season (live FPL cache)
    # ------------------------------------------------------------------
    def _load_current_season(self):
        static = load_bootstrap(os.path.join(self.raw_dir, "bootstrap_static.json"))
        if not static:
            print("bootstrap_static.json not found")
            return None

        season = get_season_label(static)
        current_gw = get_current_gw(static)

        if current_gw < 1:
            print(f"Current season {season} has not started yet — no history to load.")
            return None

        element_types = {e['id']: e['singular_name_short'] for e in static['element_types']}
        team_names = team_id_to_name(static)

        player_meta = {}
        for p in static['elements']:
            player_meta[p['id']] = {
                # Store the team NAME, not the id: ids are reassigned every season and
                # would give the model a categorical vocabulary that never transfers.
                'team_name': canon_team(team_names.get(p['team'], str(p['team']))),
                # FPL says 'GKP', vaastav says 'GK'. Collapse to 'GK'.
                'position': canon_position(element_types.get(p['element_type'], 'MID')),
            }

        cache_path = os.path.join(self.cache_dir, cache_filename(season, current_gw))
        if not os.path.exists(cache_path):
            print(f"Current season cache {cache_path} not found. Run: python src/api/async_fpl.py")
            return None

        with open(cache_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)

        rows = []
        for pid_str, data in summaries.items():
            pid = int(pid_str)
            meta = player_meta.get(pid, {'team_name': 'UNKNOWN', 'position': 'MID'})

            for hw in data.get('history', []):
                row = {
                    'player_id': pid,
                    'GW': hw['round'],
                    'season': season,
                    'price': hw['value'] / 10.0,
                    'was_home': hw['was_home'],
                    'opponent_name': canon_team(team_names.get(hw['opponent_team'],
                                                               str(hw['opponent_team']))),
                    'team_name': meta['team_name'],
                    'position': meta['position'],
                    'kickoff_time': hw['kickoff_time'],
                }
                for col in self.rolling_cols:
                    val = hw.get(col, 0)
                    try:
                        row[col] = float(val)
                    except (ValueError, TypeError):
                        row[col] = 0.0
                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        agg_dict = {col: 'sum' for col in self.rolling_cols}
        agg_dict.update({
            'price': 'mean',
            'was_home': 'first',
            'opponent_name': 'first',
            'team_name': 'first',
            'position': 'first',
            'kickoff_time': 'first',
        })
        df_grouped = df.groupby(['player_id', 'GW']).agg(agg_dict).reset_index()
        df_grouped['season'] = season

        return df_grouped

    # ------------------------------------------------------------------
    # Odds merge
    # ------------------------------------------------------------------
    def _merge_odds(self, df_all):
        """
        Attach bookmaker features by matching on (season, team_name, match date).

        Matching on date rather than on match ordinal is essential: the previous
        implementation numbered each club's matches 1..38 and assumed match N == GW N,
        so a single postponement shifted every later fixture's odds by one. 16 of 20
        clubs had a non-38 GW count in 2023-24 alone.
        """
        from src.api.odds import OddsClient, LEAGUE_DEFAULTS, POSITION_GOAL_SHARE

        odds_cols = ['win_prob', 'draw_prob', 'loss_prob',
                     'team_implied_goals', 'opponent_implied_goals', 'clean_sheet_prob']

        try:
            odds_client = OddsClient(cache_dir=self.cache_dir, raw_dir=self.raw_dir)
            odds_client.download_historical_odds()

            odds_frames = []
            for season in df_all['season'].unique():
                df_odds = odds_client.load_historical_odds(season)
                if df_odds is not None and not df_odds.empty:
                    df_odds['season'] = season
                    odds_frames.append(df_odds)

            if not odds_frames:
                print("  No historical odds found. Using league defaults.")
                for col in odds_cols:
                    df_all[col] = LEAGUE_DEFAULTS[col]
                return df_all

            df_odds_all = pd.concat(odds_frames, ignore_index=True)
            df_odds_all['match_date'] = pd.to_datetime(
                df_odds_all['date'], dayfirst=True, errors='coerce').dt.date
            df_odds_all = df_odds_all.dropna(subset=['match_date'])
            df_odds_all = df_odds_all.drop_duplicates(subset=['season', 'team_name', 'match_date'])

            df_all['match_date'] = df_all['kickoff_time'].dt.date

            merged = df_all.merge(
                df_odds_all[['season', 'team_name', 'match_date'] + odds_cols],
                on=['season', 'team_name', 'match_date'],
                how='left',
                suffixes=('', '_odds'),
            )

            matched = int(merged['win_prob'].notna().sum())
            print(f"  Matched {matched}/{len(merged)} rows with historical odds "
                  f"({matched / max(len(merged), 1) * 100:.1f}%)")

            for col in odds_cols:
                merged[col] = merged[col].fillna(LEAGUE_DEFAULTS[col])

            return merged.drop(columns=['match_date'])

        except Exception as e:
            print(f"  Odds integration failed: {e}. Using league defaults.")
            for col in odds_cols:
                df_all[col] = LEAGUE_DEFAULTS[col]
            return df_all

    # ------------------------------------------------------------------
    def build_features(self):
        print("Building time-series dataset...")

        dfs = []
        for season in ("2022-23", "2023-24"):
            df_season = self._load_vaastav_season(season)
            if df_season is not None:
                dfs.append(df_season)

        df_curr = self._load_current_season()
        if df_curr is not None and not df_curr.empty:
            dfs.append(df_curr)

        if not dfs:
            print("No data available.")
            return None

        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.sort_values(['season', 'player_id', 'GW']).reset_index(drop=True)

        df_all['kickoff_time'] = pd.to_datetime(df_all['kickoff_time'], errors='coerce', utc=True)

        # Feature Engineering: Rolling Averages (STRICT ANTI-LEAKAGE)
        # Predict GW N using only data from GW N-1, N-2, ... — so every rolling window
        # is shifted by one before aggregating.
        print("Calculating rolling features...")
        grouped = df_all.groupby(['season', 'player_id'])

        # Rest going INTO this gameweek (kickoff minus previous kickoff). Known before
        # the match, so no leakage. The inference path must reproduce this definition
        # using the UPCOMING fixture's kickoff, not the last completed one.
        df_all['days_rest'] = (
            df_all['kickoff_time'] - grouped['kickoff_time'].shift(1)
        ).dt.total_seconds() / (24 * 3600)
        df_all['days_rest'] = df_all['days_rest'].fillna(7.0)

        df_all['benched'] = (df_all['starts'] == 0).astype(int)

        new_features = {}
        for col in self.rolling_cols + ['benched']:
            new_features[f'{col}_last_1'] = grouped[col].shift(1)

            if col == 'benched':
                new_features[f'{col}_sum_last_3'] = grouped[col].apply(
                    lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
                ).reset_index(level=[0, 1], drop=True)
                new_features[f'{col}_sum_last_5'] = grouped[col].apply(
                    lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()
                ).reset_index(level=[0, 1], drop=True)
            else:
                new_features[f'{col}_mean_last_3'] = grouped[col].apply(
                    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
                ).reset_index(level=[0, 1], drop=True)
                new_features[f'{col}_mean_last_5'] = grouped[col].apply(
                    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
                ).reset_index(level=[0, 1], drop=True)

        for k, v in new_features.items():
            df_all[k] = v

        # Target variables
        df_all['target'] = df_all['total_points']
        df_all['target_minutes'] = df_all['minutes']

        print("Merging bookmaker odds features...")
        df_all = self._merge_odds(df_all)

        # Derive anytime_goal_scorer_prob from team_implied_goals + position
        from src.api.odds import OddsClient
        pos_str = df_all['position'].astype(str)
        df_all['anytime_goal_scorer_prob'] = [
            OddsClient.compute_anytime_scorer_prob(tig, pos)
            for tig, pos in zip(df_all['team_implied_goals'], pos_str)
        ]

        num_cols = df_all.select_dtypes(include=[np.number]).columns
        df_all[num_cols] = df_all[num_cols].fillna(0)

        # Categorical model features. `team_name` (not the unstable integer team id)
        # is the canonical club key.
        #
        # Fill BEFORE the string cast: astype(str) turns NaN into the literal 'nan',
        # which becomes a real category and silently pollutes the vocabulary.
        cat_cols = ['position', 'team_name', 'opponent_name', 'was_home']
        for c in cat_cols:
            n_missing = int(df_all[c].isna().sum())
            if n_missing:
                print(f"  WARNING: {n_missing} missing value(s) in categorical '{c}' "
                      f"-> 'UNKNOWN'")
            df_all[c] = df_all[c].fillna("UNKNOWN").astype(str).astype('category')
            assert 'nan' not in set(df_all[c].cat.categories), (
                f"'{c}' contains a literal 'nan' category — a NaN slipped through")

        df_all = df_all.drop(columns=['benched', 'kickoff_time'])

        out_path = os.path.join(self.processed_dir, "historical_features.parquet")
        df_all.to_parquet(out_path, index=False)
        print(f"Saved {len(df_all)} rows to {out_path}")

        # Sanity report — catches vocabulary drift before it reaches the model.
        print("\n--- Vocabulary check (train/infer must agree) ---")
        for c in ['position', 'team_name', 'opponent_name']:
            for s in sorted(df_all['season'].unique()):
                vals = sorted(df_all[df_all['season'] == s][c].astype(str).unique())
                print(f"  {c:<10} {s}: {len(vals)} values, e.g. {vals[:4]}")

        return df_all


if __name__ == "__main__":
    builder = HistoryBuilder()
    builder.build_features()
