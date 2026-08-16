import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import re
import sys
import json
import joblib
from datetime import datetime, timezone

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.season import load_bootstrap, get_season_label, get_current_gw

# Categorical model features. Club NAMES — never the integer team ids, which FPL
# reassigns every season — are the canonical keys for both team and opponent.
CATEGORICAL_FEATURES = ['position', 'team_name', 'opponent_name', 'was_home']


def _get_current_gw():
    """The most recently started GW, from bootstrap_static. 0 before the season begins."""
    return get_current_gw(load_bootstrap())


def _safe_print(text):
    """Print with ASCII fallback for Windows console encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))


# ---------------------------------------------------------------------------
# Model persistence
#
# Models are stored in LightGBM's NATIVE TEXT format plus a JSON sidecar, not as a
# pickle. Pickles carry the Python and library version that wrote them, and these
# artifacts are trained locally but loaded on Streamlit Cloud under a different Python
# (runtime.txt pins 3.11). The text format is version-independent, diffable, and
# round-trips `pandas_categorical` — the category ordering the categorical features
# depend on — byte for byte.
#
# Legacy .pkl bundles are still readable so an existing checkout keeps working.
# ---------------------------------------------------------------------------
def save_booster(booster, features, metadata, base_path):
    """Write {base}.txt (native model) and {base}.meta.json (features + metadata)."""
    os.makedirs(os.path.dirname(base_path) or '.', exist_ok=True)
    booster.save_model(f"{base_path}.txt")
    payload = dict(metadata or {})
    payload['features'] = list(features)
    with open(f"{base_path}.meta.json", 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def load_booster(base_path):
    """
    Load a model bundle. Returns (booster, meta_dict) or (None, None).

    Prefers the portable text bundle; falls back to a legacy joblib pickle.
    """
    txt, meta_path = f"{base_path}.txt", f"{base_path}.meta.json"
    if os.path.exists(txt) and os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return lgb.Booster(model_file=txt), meta

    legacy = f"{base_path}.pkl"
    if os.path.exists(legacy):
        data = joblib.load(legacy)
        meta = {k: v for k, v in data.items() if k != 'model'}
        return data['model'], meta

    return None, None


def model_bundle_exists(base_path):
    return (os.path.exists(f"{base_path}.txt") and os.path.exists(f"{base_path}.meta.json")) \
        or os.path.exists(f"{base_path}.pkl")


def load_summary_cache(cache_dir="data/cache", static=None):
    """
    Load the element-summary cache for the CURRENT season, with an integrity check.

    Returns (summaries, error_reason). `summaries` is None when no usable cache exists.

    This function exists because of a silent-corruption bug: FPL reassigns element ids
    every season, so a previous season's cache joins perfectly onto this season's
    players while describing entirely different people (100% id collision was measured
    between two consecutive seasons). Two defences:
      1. cache files are stamped with the season, so a stale file is never *found*;
      2. any legacy/unstamped file is rejected outright rather than loaded.
    """
    static = static if static is not None else load_bootstrap()
    if not static:
        return None, "bootstrap_static.json not found — cannot determine the current season"

    season = get_season_label(static)
    live_ids = {p['id'] for p in static.get('elements', [])}

    if not os.path.exists(cache_dir):
        return None, f"Cache directory {cache_dir}/ does not exist"

    pattern = re.compile(rf"^element_summary_{re.escape(season)}_gw_(\d+)\.json$")
    matches = [(int(m.group(1)), f) for f in os.listdir(cache_dir)
               for m in [pattern.match(f)] if m]

    if not matches:
        legacy = [f for f in os.listdir(cache_dir)
                  if f.startswith('element_summary_gw_')]
        if legacy:
            return None, (
                f"Only unstamped legacy cache files found ({', '.join(sorted(legacy)[:3])}). "
                f"These may belong to a previous season, in which case player ids point at "
                f"different people. Refusing to load. Run: python src/api/async_fpl.py"
            )
        return None, (
            f"No element-summary cache for season {season} in {cache_dir}/. "
            f"Run: python src/api/async_fpl.py"
        )

    gw, filename = max(matches)
    with open(os.path.join(cache_dir, filename), 'r', encoding='utf-8') as f:
        summaries = json.load(f)

    # Belt and braces: even a correctly named file must describe this season's players.
    cache_ids = {int(k) for k in summaries.keys()}
    if live_ids:
        overlap = len(cache_ids & live_ids) / len(live_ids)
        if overlap < 0.80:
            return None, (
                f"Cache {filename} overlaps the live player list by only {overlap:.0%} — "
                f"it does not describe the current squad. Refusing to load."
            )

    print(f"  Loaded element-summary cache: {filename} ({len(summaries)} players, GW{gw})")
    return summaries, None


class MinutesPredictor:
    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_base = os.path.join(self.model_dir, "lgb_ts_minutes")
        self.model_path = f"{self.model_base}.txt"
        self.model = None
        self.features_list = [
            'minutes_last_1', 'minutes_mean_last_3', 'minutes_mean_last_5',
            'starts_last_1', 'starts_mean_last_3', 'starts_mean_last_5',
            'benched_sum_last_3', 'benched_sum_last_5', 'days_rest',
            'position', 'price', 'team_name'
        ]

    PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "verbose": -1,
    }

    def train(self, df_train, gw=None, persist=True, verbose=True):
        """Train the minutes model. persist=False keeps it in memory (used for OOF folds)."""
        if verbose:
            print("\n--- Training Minutes Model ---")
        df_train = df_train.dropna(subset=['target_minutes'])

        X = df_train[self.features_list]
        y = df_train['target_minutes']

        cat_features = [f for f in self.features_list if df_train[f].dtype.name == 'category']
        train_data = lgb.Dataset(X, label=y, categorical_feature=cat_features)

        self.model = lgb.train(self.PARAMS, train_data, num_boost_round=100)

        if persist:
            meta = {'trained_at': datetime.now().isoformat(),
                    'season': get_season_label(load_bootstrap()), 'gw': gw}
            save_booster(self.model, self.features_list, meta, self.model_base)
            print(f"Minutes Model saved to {self.model_base}.txt")
            if gw:
                versioned = os.path.join(self.model_dir, f"minutes_model_gw{gw}")
                save_booster(self.model, self.features_list, meta, versioned)
                print(f"  Versioned copy: {versioned}.txt")

        return self.model

    def load_model(self):
        if self.model is not None:
            return True
        booster, meta = load_booster(self.model_base)
        if booster is None:
            return False
        self.model = booster
        self.features_list = meta.get('features', self.features_list)
        return True

    def predict(self, df_features):
        if not self.load_model():
            print("WARNING: Minutes Model not found. Returning naive minutes_last_1.")
            return df_features.get('minutes_last_1', pd.Series([0] * len(df_features),
                                                               index=df_features.index))

        df = df_features.copy()

        # Fill missing columns BEFORE casting categoricals — casting first would raise
        # KeyError on an absent column instead of defaulting it.
        for c in [c for c in self.features_list if c not in df.columns]:
            df[c] = 0.0

        for f in self.features_list:
            if f in CATEGORICAL_FEATURES:
                df[f] = df[f].astype(str).astype('category')

        preds = self.model.predict(df[self.features_list])
        return pd.Series(preds, index=df.index).clip(0, 90)


class PointsPredictor:
    PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_base = os.path.join(self.model_dir, "lgb_ts_points")
        self.model_path = f"{self.model_base}.txt"
        self.model = None
        self.features_list = None
        self.prediction_mode = "ml"
        self.prediction_warnings = []
        self.odds_confidence = "UNKNOWN"
        self._train_feature_means = None
        self._cv_rmse = None

    def _get_feature_cols(self, df):
        """Feature columns, dropping identifiers, targets and same-gameweek leakage."""
        drop_cols = [
            'player_id', 'GW', 'season', 'total_points', 'target', 'target_minutes',
            'minutes', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'bps', 'influence', 'creativity', 'threat', 'starts',
            # Integer team ids are NOT stable cross-season keys — the *_name columns
            # are used instead.
            'team', 'opponent_team', 'match_date', 'kickoff_time',
        ]
        return [c for c in df.columns if c not in drop_cols]

    # ------------------------------------------------------------------
    # Out-of-fold minutes (prevents stacking leakage)
    # ------------------------------------------------------------------
    def _oof_projected_minutes(self, df_train, n_blocks=5):
        """
        Generate projected_minutes for the training set using models that never saw
        the row being predicted.

        Feeding in-sample minutes predictions into the points model makes it over-trust
        that feature: at serving time minutes predictions are materially worse, so the
        points model is miscalibrated exactly where it matters.

        Folds are contiguous chronological blocks of (season, GW). This is
        leave-one-block-out rather than a strictly forward-looking split — a block is
        predicted by a model trained on both earlier and later blocks — which still
        removes the memorisation that causes the bias, while keeping every row usable.
        """
        print("\n--- Generating out-of-fold minutes predictions ---")
        keys = df_train['season'].astype(str) + "_" + df_train['GW'].astype(int).astype(str).str.zfill(2)
        ordered = sorted(keys.unique())
        blocks = np.array_split(np.array(ordered), n_blocks)

        oof = pd.Series(np.nan, index=df_train.index, dtype=float)

        for i, block in enumerate(blocks, 1):
            mask_val = keys.isin(set(block.tolist()))
            mask_train = ~mask_val
            if mask_train.sum() == 0 or mask_val.sum() == 0:
                continue
            fold_model = MinutesPredictor(model_dir=self.model_dir)
            fold_model.train(df_train[mask_train], persist=False, verbose=False)
            oof.loc[mask_val] = fold_model.predict(df_train[mask_val]).values
            print(f"  fold {i}/{len(blocks)}: trained on {int(mask_train.sum()):>6} rows, "
                  f"predicted {int(mask_val.sum()):>6}")

        # Any row we could not cover falls back to its own recent minutes.
        oof = oof.fillna(df_train.get('minutes_mean_last_3', pd.Series(0, index=df_train.index)))
        return oof.clip(0, 90)

    def train(self, df_train=None):
        """Trains a LightGBM model to predict points using Time-Series CV."""
        if df_train is None:
            path = "data/processed/historical_features.parquet"
            if not os.path.exists(path):
                print(f"Data not found at {path}. Run HistoryBuilder first.")
                return False
            df_train = pd.read_parquet(path)

        print(f"Loaded {len(df_train)} rows for training.")
        gw = _get_current_gw()

        # 1. Out-of-fold minutes for TRAINING the points model...
        df_train['projected_minutes'] = self._oof_projected_minutes(df_train)
        df_train['start_probability'] = (df_train['projected_minutes'] > 45).astype(float)

        # 2. ...then fit the minutes model on everything and persist it for serving.
        min_predictor = MinutesPredictor(model_dir=self.model_dir)
        min_predictor.train(df_train, gw=gw, persist=True)

        # 3. Setup Points Model Features
        base_features = self._get_feature_cols(df_train)
        base_features = [f for f in base_features
                         if f not in ['projected_minutes', 'start_probability']]

        raw_minute_cols = [c for c in base_features
                           if 'minutes_' in c or 'starts_' in c or 'benched_' in c or c == 'days_rest']
        features_A = [f for f in base_features if f not in raw_minute_cols] + \
                     ['projected_minutes', 'start_probability']
        features_B = base_features + ['projected_minutes', 'start_probability']

        cv_splits = [
            ({'GW': (1, 20)}, {'GW': (21, 25)}),
            ({'GW': (1, 25)}, {'GW': (26, 30)}),
        ]

        print("\n--- Running A/B Test on Features ---")
        best_rmse_A = self._run_cv(df_train, features_A, cv_splits, self.PARAMS)
        best_rmse_B = self._run_cv(df_train, features_B, cv_splits, self.PARAMS)

        print(f"\nRMSE A (Projected Only): {best_rmse_A:.4f}")
        print(f"RMSE B (Projected + Raw): {best_rmse_B:.4f}")

        if best_rmse_A <= best_rmse_B + 0.05:
            print("Winner: Set A (Projected Minutes Only)")
            self.features_list = features_A
            self._cv_rmse = best_rmse_A
        else:
            print("Winner: Set B (Projected + Raw)")
            self.features_list = features_B
            self._cv_rmse = best_rmse_B

        print(f"Using {len(self.features_list)} features for final Points Model training.")

        print("\nTraining final points model on all data...")
        df_train = df_train.dropna(subset=['target'])

        X_all = df_train[self.features_list]
        y_all = df_train['target']
        cat_features = [f for f in self.features_list if df_train[f].dtype.name == 'category']
        train_data_all = lgb.Dataset(X_all, label=y_all, categorical_feature=cat_features)

        self.model = lgb.train(self.PARAMS, train_data_all, num_boost_round=150)

        num_features = [f for f in self.features_list if df_train[f].dtype.name != 'category']
        self._train_feature_means = df_train[num_features].mean().to_dict()

        meta = {
            'train_feature_means': self._train_feature_means,
            'trained_at': datetime.now().isoformat(),
            'season': get_season_label(load_bootstrap()),
            'gw': gw,
            'cv_rmse': self._cv_rmse,
            'n_train_rows': int(len(df_train)),
            'train_seasons': sorted(df_train['season'].astype(str).unique().tolist()),
        }
        save_booster(self.model, self.features_list, meta, self.model_base)
        print(f"Points Model saved to {self.model_base}.txt")

        if gw:
            versioned = os.path.join(self.model_dir, f"points_model_gw{gw}")
            save_booster(self.model, self.features_list, meta, versioned)
            print(f"  Versioned copy: {versioned}.txt")

        return True

    def _run_cv(self, df_train, features, cv_splits, params):
        cat_features = [f for f in features if df_train[f].dtype.name == 'category']
        cv_season = "2023-24"
        mask_season = df_train['season'] == cv_season
        rmse_list = []

        for train_bounds, val_bounds in cv_splits:
            t_start, t_end = train_bounds['GW']
            v_start, v_end = val_bounds['GW']

            mask_train = mask_season & (df_train['GW'] >= t_start) & (df_train['GW'] <= t_end)
            mask_val = mask_season & (df_train['GW'] >= v_start) & (df_train['GW'] <= v_end)

            df_t, df_v = df_train[mask_train], df_train[mask_val]
            if df_t.empty or df_v.empty:
                continue

            train_data = lgb.Dataset(df_t[features], label=df_t['target'],
                                     categorical_feature=cat_features)
            val_data = lgb.Dataset(df_v[features], label=df_v['target'],
                                   categorical_feature=cat_features, reference=train_data)

            model_cv = lgb.train(params, train_data, num_boost_round=500,
                                 valid_sets=[val_data],
                                 callbacks=[lgb.early_stopping(50, verbose=False)])
            rmse_list.append(model_cv.best_score['valid_0']['rmse'])

        return sum(rmse_list) / len(rmse_list) if rmse_list else 999.0

    def load_model(self):
        booster, meta = load_booster(self.model_base)
        if booster is None:
            return False
        self.model = booster
        self.features_list = meta['features']
        self._train_feature_means = meta.get('train_feature_means', {})
        self._cv_rmse = meta.get('cv_rmse')
        return True

    # ------------------------------------------------------------------
    # Rolling features at inference time
    # ------------------------------------------------------------------
    def _build_rolling_features(self, summaries, df_features):
        """
        Rebuild the rolling features from element-summary history.

        Definitions here MUST mirror history_builder exactly, since the model was
        trained on that version of each feature.
        """
        rolling_cols = [
            'minutes', 'total_points', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'bps', 'influence', 'creativity', 'threat', 'starts'
        ]

        # Upcoming kickoff per player, so days_rest can be measured against the match
        # being predicted (as in training) rather than between the last two played.
        next_kickoff = {}
        if 'next_kickoff_time' in df_features.columns:
            for pid, ko in zip(df_features['id'], df_features['next_kickoff_time']):
                next_kickoff[int(pid)] = ko

        def _parse(ts):
            if not ts or pd.isna(ts):
                return None
            try:
                return pd.to_datetime(ts, utc=True).to_pydatetime()
            except Exception:
                return None

        def collapse_double_gameweeks(history):
            """
            Merge a gameweek's multiple fixtures into one entry.

            history_builder groups training rows by (player_id, GW) and SUMS the stats,
            so a double gameweek is one row there. element-summary returns one entry per
            MATCH, so without this a DGW would make the last-3 window span 3 matches
            instead of 3 gameweeks — a different feature from the one trained on.
            """
            by_round = {}
            order = []
            for hw in history:
                rnd = hw.get('round')
                if rnd is None:
                    rnd = f"_{len(order)}"  # keep unroundable entries distinct
                if rnd not in by_round:
                    by_round[rnd] = dict(hw)
                    order.append(rnd)
                else:
                    merged_hw = by_round[rnd]
                    for col in rolling_cols:
                        merged_hw[col] = (float(merged_hw.get(col, 0) or 0)
                                          + float(hw.get(col, 0) or 0))
                    # Latest kickoff in the gameweek is the one rest is measured from.
                    if str(hw.get('kickoff_time') or '') > str(merged_hw.get('kickoff_time') or ''):
                        merged_hw['kickoff_time'] = hw.get('kickoff_time')
            return [by_round[r] for r in order]

        rolling_data = []
        for pid_str, data in summaries.items():
            pid = int(pid_str)
            history = collapse_double_gameweeks(data.get('history', []))
            row = {'id': pid}

            if not history:
                for col in rolling_cols:
                    row[f'{col}_last_1'] = 0.0
                    row[f'{col}_mean_last_3'] = 0.0
                    row[f'{col}_mean_last_5'] = 0.0
                row['benched_sum_last_3'] = 0.0
                row['benched_sum_last_5'] = 0.0
                row['days_rest'] = 7.0
                rolling_data.append(row)
                continue

            vals = {col: [float(hw.get(col, 0) or 0) for hw in history] for col in rolling_cols}

            for col in rolling_cols:
                lst = vals[col]
                row[f'{col}_last_1'] = lst[-1] if lst else 0.0
                row[f'{col}_mean_last_3'] = sum(lst[-3:]) / len(lst[-3:]) if lst else 0.0
                row[f'{col}_mean_last_5'] = sum(lst[-5:]) / len(lst[-5:]) if lst else 0.0

            starts_list = vals.get('starts', [])
            benched_flags = [1 if s == 0 else 0 for s in starts_list]
            row['benched_sum_last_3'] = float(sum(benched_flags[-3:])) if benched_flags else 0.0
            row['benched_sum_last_5'] = float(sum(benched_flags[-5:])) if benched_flags else 0.0

            # days_rest = UPCOMING kickoff - last played kickoff.
            last_ko = _parse(history[-1].get('kickoff_time'))
            upcoming = _parse(next_kickoff.get(pid))
            if last_ko and upcoming:
                row['days_rest'] = max((upcoming - last_ko).total_seconds() / 86400, 0)
            elif len(history) >= 2:
                # No fixture scheduled (blank GW): fall back to the last observed gap.
                prev_ko = _parse(history[-2].get('kickoff_time'))
                row['days_rest'] = (
                    max((last_ko - prev_ko).total_seconds() / 86400, 0)
                    if last_ko and prev_ko else 7.0
                )
            else:
                row['days_rest'] = 7.0

            rolling_data.append(row)

        return pd.DataFrame(rolling_data)

    def predict(self, df_features):
        """Two-stage prediction: Minutes Model → Points Model.

        Sets self.prediction_mode / .prediction_warnings / .odds_confidence for callers.
        """
        self.prediction_mode = "ml"
        self.prediction_warnings = []
        self.odds_confidence = "UNKNOWN"

        if not self.load_model():
            return self._emergency_heuristic(
                df_features,
                reason=f"ML Points Model not found at {self.model_base}.txt — "
                       f"the trained model must be committed to the repo, since a "
                       f"deployed instance cannot retrain itself")

        summaries, err = load_summary_cache()
        if summaries is None:
            return self._emergency_heuristic(df_features, reason=err)
        if not summaries:
            return self._emergency_heuristic(
                df_features, reason="Element-summary cache is empty (0 players)")

        df_rolling = self._build_rolling_features(summaries, df_features)
        if df_rolling.empty or 'id' not in df_rolling.columns:
            return self._emergency_heuristic(
                df_features, reason="Could not build any rolling features from the cache")

        df_merged = df_features.merge(df_rolling, on='id', how='left')
        # A left merge must not change row count; duplicate ids in df_rolling would
        # expand it and silently desync the positional assignments below.
        if len(df_merged) != len(df_features):
            return self._emergency_heuristic(
                df_features,
                reason=f"Rolling-feature merge changed row count "
                       f"({len(df_features)} -> {len(df_merged)}); cache has duplicate ids")

        covered = df_merged['days_rest'].notna().sum()
        if covered < len(df_merged):
            self.prediction_warnings.append(
                f"{len(df_merged) - covered}/{len(df_merged)} players have no history in the "
                f"cache; their rolling features are missing."
            )

        # --- Stage 1: Minutes Model ---
        min_predictor = MinutesPredictor(model_dir=self.model_dir)
        df_merged['projected_minutes'] = min_predictor.predict(df_merged)
        df_merged['start_probability'] = (df_merged['projected_minutes'] > 45).astype(float)

        print(f"  Minutes Model: avg projected = {df_merged['projected_minutes'].mean():.1f} min")

        # --- Stage 2: Points Model ---
        missing_cols = [c for c in self.features_list if c not in df_merged.columns]
        if missing_cols:
            print(f"  Adding {len(missing_cols)} missing column(s) as 0 to match training data: "
                  f"{missing_cols[:6]}")
            self.prediction_warnings.append(
                f"{len(missing_cols)} training feature(s) absent at inference and zero-filled: "
                f"{missing_cols[:6]}"
            )
            for col in missing_cols:
                df_merged[col] = 0.0

        for f in self.features_list:
            if f in CATEGORICAL_FEATURES:
                df_merged[f] = df_merged[f].astype(str).astype('category')

        preds = self.model.predict(df_merged[self.features_list])

        df_features = df_features.copy()
        df_features['predicted_points'] = np.clip(preds, 0, None)
        df_features['projected_minutes'] = df_merged['projected_minutes'].values
        df_features['start_probability'] = df_merged['start_probability'].values

        # Availability haircut from FPL's injury news. This is NOT redundant with
        # projected_minutes: the minutes model only sees match history, so it cannot
        # know about an injury announced since the last fixture. minutes_prob is 1.0
        # for every unflagged player, so this only touches flagged ones.
        if 'minutes_prob' in df_features.columns:
            df_features['predicted_points'] *= df_features['minutes_prob'].fillna(1.0)

        # --- Detect odds confidence ---
        has_real_odds = any(
            col in df_features.columns and df_features[col].nunique() > 1
            for col in ['win_prob', 'draw_prob', 'loss_prob']
        )
        self.odds_confidence = "HIGH" if has_real_odds else "LOW"
        df_features['odds_confidence'] = self.odds_confidence

        if self.odds_confidence == "LOW":
            self.prediction_warnings.append(
                "Odds confidence: LOW -- using league-average defaults. "
                "Set ODDS_API_KEY env var for live odds. "
                "Captaincy ranking accuracy is reduced."
            )

        # --- Captaincy Score ---
        # predicted_points x minutes_confidence x odds_confidence. Prefer this over raw
        # predicted_points when ranking captains: it penalises rotation risk and hard
        # fixtures, which matter far more for a doubled score.
        min_conf = (df_features['projected_minutes'] / 90.0).clip(0, 1)
        odds_conf = df_features.get('win_prob', pd.Series([0.33] * len(df_features),
                                                          index=df_features.index))
        df_features['captaincy_score'] = (
            df_features['predicted_points']
            * (0.6 + 0.4 * min_conf)
            * (0.7 + 0.3 * odds_conf)
        )

        df_features['prediction_mode'] = "ml"
        return df_features

    def generate_audit_report(self, df_train, df_features):
        """Audit report to catch leakage and check model behaviour. Saves a CSV."""
        gw = _get_current_gw()

        print("\n" + "=" * 60)
        print(f"  PREDICTION AUDIT REPORT - GW {gw}")
        print("=" * 60)

        if not self.load_model():
            print("Model not loaded. Cannot generate audit.")
            return

        importance = self.model.feature_importance(importance_type='gain')
        feat_imp = pd.Series(importance, index=self.features_list).sort_values(ascending=False)
        print("\n--- Top 10 Features (Gain) ---")
        print(feat_imp.head(10).to_string())

        if self._cv_rmse is not None:
            print(f"\n--- Time-Series CV RMSE (out-of-sample): {self._cv_rmse:.4f} ---")
        print("NOTE: the final model is fitted on ALL rows, so any per-slice RMSE below is\n"
              "      IN-SAMPLE and optimistic. Use the CV figure above for model quality.")

        mask_val = (df_train['season'] == '2023-24') & (df_train['GW'] >= 30) & (df_train['target'].notna())
        df_val = df_train[mask_val].copy()

        if not df_val.empty:
            min_predictor = MinutesPredictor(model_dir=self.model_dir)
            df_val['projected_minutes'] = min_predictor.predict(df_val)
            df_val['start_probability'] = (df_val['projected_minutes'] > 45).astype(float)

            for c in [c for c in self.features_list if c not in df_val.columns]:
                df_val[c] = 0.0
            for f in self.features_list:
                if f in CATEGORICAL_FEATURES:
                    df_val[f] = df_val[f].astype(str).astype('category')

            df_val['pred'] = self.model.predict(df_val[self.features_list])
            df_val['sq_err'] = (df_val['target'] - df_val['pred']) ** 2

            print("\n--- In-sample RMSE (2023-24 GW30+) ---")
            print(f"Overall RMSE: {np.sqrt(df_val['sq_err'].mean()):.4f}")

            df_val['min_sq_err'] = (df_val['target_minutes'] - df_val['projected_minutes']) ** 2
            print(f"Minutes Model RMSE: {np.sqrt(df_val['min_sq_err'].mean()):.2f}")

            print("\nRMSE by Position:")
            print(df_val.groupby('position', observed=True)['sq_err'].mean().apply(np.sqrt).to_string())

            print("\nRMSE by Price Band:")
            df_val['price_band'] = pd.cut(df_val['price'], bins=[0, 5.0, 7.5, 10.0, 15.0],
                                          labels=['Budget (<5.0)', 'Mid (5-7.5)',
                                                  'Premium (7.5-10)', 'Ultra (>10)'])
            print(df_val.groupby('price_band', observed=False)['sq_err'].mean().apply(np.sqrt).to_string())

        print("\n--- Top 20 Predicted Players (Next GW) ---")
        df_pred = self.predict(df_features.copy())
        top_20 = df_pred.sort_values('predicted_points', ascending=False).head(20)
        display_cols = [c for c in ['web_name', 'team_name', 'position', 'price', 'next_opponent',
                                    'projected_minutes', 'predicted_points', 'captaincy_score']
                        if c in top_20.columns]
        _safe_print(top_20[display_cols].to_string(index=False))

        print("\n--- Top 5 Captain Picks (by captaincy_score) ---")
        cap_top = df_pred.sort_values('captaincy_score', ascending=False).head(5)
        cap_cols = [c for c in ['web_name', 'position', 'price', 'next_opponent',
                                'predicted_points', 'projected_minutes', 'captaincy_score']
                    if c in cap_top.columns]
        _safe_print(cap_top[cap_cols].to_string(index=False))

        if 'total_points_mean_last_3' in df_pred.columns:
            print("\n--- Top 10 Prediction Surprises ---")
            df_pred['surprise_delta'] = df_pred['predicted_points'] - df_pred['total_points_mean_last_3']
            surprises = df_pred.sort_values('surprise_delta', ascending=False).head(10)
            _safe_print(surprises[['web_name', 'predicted_points',
                                   'total_points_mean_last_3', 'surprise_delta']].to_string(index=False))

        if self._train_feature_means:
            print("\n--- Feature Drift Report (current vs training means) ---")
            num_features = [f for f in self.features_list
                            if f in df_pred.columns and df_pred[f].dtype.kind in 'ifc']
            drift_rows = []
            for f in num_features:
                train_mean = self._train_feature_means.get(f, 0)
                curr_mean = df_pred[f].mean()
                pct_drift = ((curr_mean - train_mean) / abs(train_mean) * 100) if train_mean else 0.0
                drift_rows.append({'feature': f, 'train_mean': round(train_mean, 4),
                                   'current_mean': round(curr_mean, 4), 'drift_%': round(pct_drift, 1)})
            df_drift = pd.DataFrame(drift_rows)
            significant = df_drift[df_drift['drift_%'].abs() > 20].sort_values(
                'drift_%', key=abs, ascending=False)
            if not significant.empty:
                print(f"  {len(significant)} features with >20% drift detected:")
                print(significant.to_string(index=False))
            else:
                print("  No significant feature drift detected (all within 20%).")

        reports_dir = os.path.join("data", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        save_cols = [c for c in ['id', 'web_name', 'team_name', 'position', 'price', 'next_opponent',
                                 'projected_minutes', 'start_probability', 'predicted_points',
                                 'captaincy_score'] if c in df_pred.columns]
        report_path = os.path.join(reports_dir, f"predictions_gw{gw}.csv")
        df_pred[save_cols].sort_values('predicted_points', ascending=False).to_csv(report_path, index=False)
        print(f"\n--- Prediction CSV saved to {report_path} ---")

        print(f"\nPrediction Mode: {self.prediction_mode}")
        print(f"Odds Confidence: {self.odds_confidence}")
        for w in self.prediction_warnings:
            print(f"  [!] {w}")

    def _emergency_heuristic(self, df_features, reason="Unknown"):
        """Emergency fallback if the model or its inputs are unusable. LOUD warnings."""
        self.prediction_mode = "fallback"
        warning = f"[FALLBACK] {reason}. Using heuristic instead of ML model."
        self.prediction_warnings.append(warning)

        print("\n" + "!" * 60)
        print("  !!!  EMERGENCY HEURISTIC FALLBACK ACTIVE  !!!")
        print("!" * 60)
        print(f"  Reason: {reason}")
        print("  Predictions will be SIGNIFICANTLY less accurate.")
        print("  Fix: run `python src/api/async_fpl.py` then `python src/model/predictor.py`.")
        print("!" * 60 + "\n")

        df_features = df_features.copy()
        if 'total_points_mean_last_3' in df_features.columns:
            pts = df_features['total_points_mean_last_3'].fillna(0)
            xg = df_features.get(
                'expected_goal_involvements_mean_last_3',
                pd.Series([0] * len(df_features), index=df_features.index)).fillna(0)
            preds = pts * 0.4 + xg * 2.0
        elif 'ep_next' in df_features.columns:
            # FPL's own expected points for the next GW — a far better cold-start prior
            # than season totals, and the only signal available before GW1.
            preds = pd.to_numeric(df_features['ep_next'], errors='coerce').fillna(0)
        else:
            preds = df_features.get(
                'total_points', pd.Series([0] * len(df_features), index=df_features.index)) * 0.1

        df_features['predicted_points'] = pd.Series(preds, index=df_features.index).clip(lower=0)
        if 'minutes_prob' in df_features.columns:
            df_features['predicted_points'] *= df_features['minutes_prob'].fillna(1.0)
        df_features['projected_minutes'] = 0.0
        df_features['start_probability'] = 0.0
        df_features['captaincy_score'] = df_features['predicted_points']
        df_features['odds_confidence'] = "NONE"
        df_features['prediction_mode'] = "fallback"
        df_features['prediction_warning'] = warning
        self.odds_confidence = "NONE"
        return df_features


if __name__ == "__main__":
    predictor = PointsPredictor()
    predictor.train()

    if os.path.exists("data/processed/player_features.parquet") and \
            os.path.exists("data/processed/historical_features.parquet"):
        df_features = pd.read_parquet("data/processed/player_features.parquet")
        df_train = pd.read_parquet("data/processed/historical_features.parquet")
        predictor.generate_audit_report(df_train, df_features)
    else:
        print("Run history_builder.py and processor.py first to generate data.")
