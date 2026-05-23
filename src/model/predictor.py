import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import json
import joblib
from datetime import datetime

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _get_current_gw():
    """Detect the current GW from bootstrap_static or cache filenames."""
    static_path = os.path.join("data", "raw", "bootstrap_static.json")
    if os.path.exists(static_path):
        with open(static_path, 'r', encoding='utf-8') as f:
            static = json.load(f)
        for ev in static.get('events', []):
            if ev.get('is_current'):
                return ev['id']
    # Fallback: parse from cache filenames
    cache_dir = os.path.join("data", "cache")
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('element_summary_gw_')]
        if cache_files:
            return max(int(f.split('_')[-1].split('.')[0]) for f in cache_files)
    return 0


def _safe_print(text):
    """Print with ASCII fallback for Windows console encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))


class MinutesPredictor:
    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "lgb_ts_minutes.pkl")
        self.model = None
        self.features_list = [
            'minutes_last_1', 'minutes_mean_last_3', 'minutes_mean_last_5',
            'starts_last_1', 'starts_mean_last_3', 'starts_mean_last_5',
            'benched_sum_last_3', 'benched_sum_last_5', 'days_rest',
            'position', 'price', 'team'
        ]

    def train(self, df_train, gw=None):
        print("\n--- Training Minutes Model ---")
        df_train = df_train.dropna(subset=['target_minutes'])
        
        X = df_train[self.features_list]
        y = df_train['target_minutes']
        
        cat_features = [f for f in self.features_list if df_train[f].dtype.name == 'category']
        train_data = lgb.Dataset(X, label=y, categorical_feature=cat_features)
        
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "verbose": -1
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=100)
        joblib.dump({'model': self.model, 'features': self.features_list}, self.model_path)
        print(f"Minutes Model saved to {self.model_path}")
        
        # Save versioned copy
        if gw:
            versioned = os.path.join(self.model_dir, f"minutes_model_gw{gw}.pkl")
            joblib.dump({'model': self.model, 'features': self.features_list}, versioned)
            print(f"  Versioned copy: {versioned}")
        
    def load_model(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.features_list = data['features']
            return True
        return False
        
    def predict(self, df_features):
        if not self.load_model():
            print("WARNING: Minutes Model not found. Returning naive minutes_last_1.")
            return df_features.get('minutes_last_1', pd.Series([0]*len(df_features)))
            
        for f in self.features_list:
             if f in ['position', 'team']:
                 df_features[f] = df_features[f].astype(str).astype('category')
                 
        missing = [c for c in self.features_list if c not in df_features.columns]
        for c in missing:
            df_features[c] = 0.0
            
        X = df_features[self.features_list]
        preds = self.model.predict(X)
        return pd.Series(preds, index=df_features.index).clip(0, 90)


class PointsPredictor:
    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "lgb_ts_points.pkl")
        self.model = None
        self.features_list = None
        self.prediction_mode = "ml"  # "ml" or "fallback"
        self.prediction_warnings = []
        self.odds_confidence = "UNKNOWN"  # "HIGH", "LOW", or "UNKNOWN"
        # Store training feature means for drift detection
        self._train_feature_means = None

    def _get_feature_cols(self, df):
        """Extracts feature columns by dropping identifiers and leakages."""
        drop_cols = [
            'player_id', 'GW', 'season', 'total_points', 'target',
            'minutes', 'expected_goals', 'expected_assists', 
            'expected_goal_involvements', 'expected_goals_conceded',
            'bps', 'influence', 'creativity', 'threat', 'starts'
        ]
        return [c for c in df.columns if c not in drop_cols]

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
        
        # 1. Train the Minutes Model
        min_predictor = MinutesPredictor(model_dir=self.model_dir)
        min_predictor.train(df_train, gw=gw)
        
        # 2. Add projected_minutes to df_train
        df_train['projected_minutes'] = min_predictor.predict(df_train)
        df_train['start_probability'] = (df_train['projected_minutes'] > 45).astype(float)
        
        # 3. Setup Points Model Features
        base_features = self._get_feature_cols(df_train)
        base_features = [f for f in base_features if f not in ['target_minutes', 'projected_minutes', 'start_probability']]
        
        # A/B Test
        raw_minute_cols = [c for c in base_features if 'minutes_' in c or 'starts_' in c or 'benched_' in c or c == 'days_rest']
        features_A = [f for f in base_features if f not in raw_minute_cols] + ['projected_minutes', 'start_probability']
        features_B = base_features + ['projected_minutes', 'start_probability']
        
        cv_splits = [
            ({'GW': (1, 20)}, {'GW': (21, 25)}),
            ({'GW': (1, 25)}, {'GW': (26, 30)})
        ]
        
        params = {
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
            "verbose": -1
        }
        
        print("\n--- Running A/B Test on Features ---")
        best_rmse_A = self._run_cv(df_train, features_A, cv_splits, params)
        best_rmse_B = self._run_cv(df_train, features_B, cv_splits, params)
        
        print(f"\nRMSE A (Projected Only): {best_rmse_A:.4f}")
        print(f"RMSE B (Projected + Raw): {best_rmse_B:.4f}")
        
        if best_rmse_A <= best_rmse_B + 0.05:
            print("Winner: Set A (Projected Minutes Only)")
            self.features_list = features_A
        else:
            print("Winner: Set B (Projected + Raw)")
            self.features_list = features_B
            
        print(f"Using {len(self.features_list)} features for final Points Model training.")
        
        # Final Training
        print("\nTraining final points model on all data...")
        df_train = df_train.dropna(subset=['target'])
        
        X_all = df_train[self.features_list]
        y_all = df_train['target']
        cat_features = [f for f in self.features_list if df_train[f].dtype.name == 'category']
        train_data_all = lgb.Dataset(X_all, label=y_all, categorical_feature=cat_features)
        
        self.model = lgb.train(params, train_data_all, num_boost_round=150)
        
        # Save training feature means for drift detection
        num_features = [f for f in self.features_list if df_train[f].dtype.name != 'category']
        self._train_feature_means = df_train[num_features].mean().to_dict()
        
        joblib.dump({
            'model': self.model, 
            'features': self.features_list,
            'train_feature_means': self._train_feature_means,
            'trained_at': datetime.now().isoformat(),
            'gw': gw,
        }, self.model_path)
        print(f"Points Model saved to {self.model_path}")
        
        # Save versioned copy
        if gw:
            versioned = os.path.join(self.model_dir, f"points_model_gw{gw}.pkl")
            joblib.dump({
                'model': self.model, 
                'features': self.features_list,
                'train_feature_means': self._train_feature_means,
                'trained_at': datetime.now().isoformat(),
                'gw': gw,
            }, versioned)
            print(f"  Versioned copy: {versioned}")
            
        return True

    def _run_cv(self, df_train, features, cv_splits, params):
        cat_features = [f for f in features if df_train[f].dtype.name == 'category']
        cv_season = "2023-24"
        mask_season = df_train['season'] == cv_season
        rmse_list = []
        
        for i, (train_bounds, val_bounds) in enumerate(cv_splits):
            t_start, t_end = train_bounds['GW']
            v_start, v_end = val_bounds['GW']
            
            mask_train = mask_season & (df_train['GW'] >= t_start) & (df_train['GW'] <= t_end)
            mask_val = mask_season & (df_train['GW'] >= v_start) & (df_train['GW'] <= v_end)
            
            df_t, df_v = df_train[mask_train], df_train[mask_val]
            if df_t.empty or df_v.empty: continue
                
            train_data = lgb.Dataset(df_t[features], label=df_t['target'], categorical_feature=cat_features)
            val_data = lgb.Dataset(df_v[features], label=df_v['target'], categorical_feature=cat_features, reference=train_data)
            
            model_cv = lgb.train(params, train_data, num_boost_round=500, valid_sets=[val_data], callbacks=[lgb.early_stopping(50, verbose=False)])
            rmse_list.append(model_cv.best_score['valid_0']['rmse'])
            
        return sum(rmse_list)/len(rmse_list) if rmse_list else 999.0

    def load_model(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.features_list = data['features']
            self._train_feature_means = data.get('train_feature_means', {})
            return True
        return False

    def predict(self, df_features):
        """Two-stage prediction: Minutes Model → Points Model.
        
        Sets self.prediction_mode and self.prediction_warnings for callers to inspect.
        """
        self.prediction_mode = "ml"
        self.prediction_warnings = []
        self.odds_confidence = "UNKNOWN"
        
        if not self.load_model():
            return self._emergency_heuristic(df_features, 
                reason="ML Points Model file not found at " + self.model_path)
            
        cache_dir = "data/cache"
        if not os.path.exists(cache_dir):
            return self._emergency_heuristic(df_features,
                reason="Cache directory data/cache/ does not exist")
                
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('element_summary_gw_')]
        if not cache_files:
            return self._emergency_heuristic(df_features,
                reason="No element_summary cache files found in data/cache/")
            
        # Get latest cache file
        latest_cache = sorted(cache_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        with open(os.path.join(cache_dir, latest_cache), 'r', encoding='utf-8') as f:
            summaries = json.load(f)
            
        # Rolling columns to calculate
        rolling_cols = [
            'minutes', 'total_points', 'expected_goals', 'expected_assists', 
            'expected_goal_involvements', 'expected_goals_conceded',
            'bps', 'influence', 'creativity', 'threat', 'starts'
        ]
        
        rolling_data = []
        for pid_str, data in summaries.items():
            history = data.get('history', [])
            row = {'id': int(pid_str)}
            
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
            
            kickoff_times = [hw.get('kickoff_time', None) for hw in history]
            if len(kickoff_times) >= 2 and kickoff_times[-1] and kickoff_times[-2]:
                try:
                    t1 = datetime.fromisoformat(kickoff_times[-1].replace('Z', '+00:00'))
                    t2 = datetime.fromisoformat(kickoff_times[-2].replace('Z', '+00:00'))
                    row['days_rest'] = max((t1 - t2).total_seconds() / 86400, 0)
                except Exception:
                    row['days_rest'] = 7.0
            else:
                row['days_rest'] = 7.0
                
            rolling_data.append(row)
            
        df_rolling = pd.DataFrame(rolling_data)
        df_merged = df_features.merge(df_rolling, on='id', how='left')
        
        # --- Stage 1: Minutes Model ---
        min_predictor = MinutesPredictor(model_dir=self.model_dir)
        df_merged['projected_minutes'] = min_predictor.predict(df_merged)
        df_merged['start_probability'] = (df_merged['projected_minutes'] > 45).astype(float)
        
        print(f"  Minutes Model: avg projected = {df_merged['projected_minutes'].mean():.1f} min")
        
        # --- Stage 2: Points Model ---
        missing_cols = [c for c in self.features_list if c not in df_merged.columns]
        if missing_cols:
             print(f"Adding {len(missing_cols)} missing columns as 0 to match training data.")
             for col in missing_cols:
                 df_merged[col] = 0.0
                 
        for f in self.features_list:
             if f in ['position', 'team', 'opponent_team', 'was_home']:
                 df_merged[f] = df_merged[f].astype(str).astype('category')
                 
        X = df_merged[self.features_list]
        preds = self.model.predict(X)
        
        df_features['predicted_points'] = preds
        df_features['projected_minutes'] = df_merged['projected_minutes'].values
        df_features['start_probability'] = df_merged['start_probability'].values
        df_features['predicted_points'] = df_features['predicted_points'].clip(lower=0)
        
        if 'minutes_prob' in df_features.columns:
            df_features['predicted_points'] *= df_features['minutes_prob'].fillna(1.0)
        
        # --- Detect odds confidence ---
        # If all odds columns are identical (league defaults), confidence is LOW
        odds_cols_check = ['win_prob', 'draw_prob', 'loss_prob']
        has_real_odds = False
        for col in odds_cols_check:
            if col in df_features.columns:
                if df_features[col].nunique() > 1:
                    has_real_odds = True
                    break
        
        self.odds_confidence = "HIGH" if has_real_odds else "LOW"
        df_features['odds_confidence'] = self.odds_confidence
        
        if self.odds_confidence == "LOW":
            self.prediction_warnings.append(
                "Odds confidence: LOW -- using league-average defaults. "
                "Set ODDS_API_KEY env var for live odds. "
                "Captaincy ranking accuracy is reduced."
            )
        
        # --- Captaincy Score ---
        # predicted_points × minutes_confidence × odds_confidence
        min_conf = (df_features['projected_minutes'] / 90.0).clip(0, 1)
        odds_conf = df_features.get('win_prob', pd.Series([0.33]*len(df_features)))
        df_features['captaincy_score'] = (
            df_features['predicted_points'] * 
            (0.6 + 0.4 * min_conf) *  # Weighted minutes confidence 
            (0.7 + 0.3 * odds_conf)   # Weighted odds confidence
        )
            
        return df_features

    def generate_audit_report(self, df_train, df_features):
        """Generates an audit report to catch leakage and check model behavior.
        Saves a CSV report to data/reports/.
        """
        gw = _get_current_gw()
        
        print("\n" + "="*60)
        print(f"  PREDICTION AUDIT REPORT — GW {gw}")
        print("="*60)
        
        if not self.load_model():
            print("Model not loaded. Cannot generate audit.")
            return

        # 1. Top Features
        importance = self.model.feature_importance(importance_type='gain')
        feat_imp = pd.Series(importance, index=self.features_list).sort_values(ascending=False)
        print("\n--- Top 10 Features (Gain) ---")
        print(feat_imp.head(10).to_string())

        # 2. Holdout RMSE Analysis
        mask_val = (df_train['season'] == '2023-24') & (df_train['GW'] >= 30) & (df_train['target'].notna())
        df_val = df_train[mask_val].copy()
        
        if not df_val.empty:
            min_predictor = MinutesPredictor(model_dir=self.model_dir)
            df_val['projected_minutes'] = min_predictor.predict(df_val)
            df_val['start_probability'] = (df_val['projected_minutes'] > 45).astype(float)
            
            missing_val = [c for c in self.features_list if c not in df_val.columns]
            for c in missing_val:
                df_val[c] = 0.0
                
            for f in self.features_list:
                if f in ['position', 'team', 'opponent_team', 'was_home']:
                    df_val[f] = df_val[f].astype(str).astype('category')
                    
            preds = self.model.predict(df_val[self.features_list])
            df_val['pred'] = preds
            df_val['sq_err'] = (df_val['target'] - df_val['pred']) ** 2
            
            print("\n--- Holdout RMSE (2023-24 GW30+) ---")
            overall_rmse = np.sqrt(df_val['sq_err'].mean())
            print(f"Overall RMSE: {overall_rmse:.4f}")
            
            df_val['min_sq_err'] = (df_val['target_minutes'] - df_val['projected_minutes']) ** 2
            min_rmse = np.sqrt(df_val['min_sq_err'].mean())
            print(f"Minutes Model RMSE: {min_rmse:.2f}")
            
            print("\nRMSE by Position:")
            rmse_pos = df_val.groupby('position')['sq_err'].mean().apply(np.sqrt)
            print(rmse_pos.to_string())
            
            print("\nRMSE by Price Band:")
            df_val['price_band'] = pd.cut(df_val['price'], bins=[0, 5.0, 7.5, 10.0, 15.0], labels=['Budget (<5.0)', 'Mid (5-7.5)', 'Premium (7.5-10)', 'Ultra (>10)'])
            rmse_price = df_val.groupby('price_band', observed=False)['sq_err'].mean().apply(np.sqrt)
            print(rmse_price.to_string())

        # 3. Current Predictions
        print("\n--- Top 20 Predicted Players (Next GW) ---")
        df_pred = self.predict(df_features.copy())
        top_20 = df_pred.sort_values('predicted_points', ascending=False).head(20)
        display_cols = ['web_name', 'team', 'position', 'price', 'next_opponent', 
                       'projected_minutes', 'predicted_points', 'captaincy_score']
        display_cols = [c for c in display_cols if c in top_20.columns]
        _safe_print(top_20[display_cols].to_string(index=False))

        # 4. Captaincy Ranking
        print("\n--- Top 5 Captain Picks ---")
        cap_top = df_pred.sort_values('captaincy_score', ascending=False).head(5)
        cap_cols = ['web_name', 'position', 'price', 'next_opponent', 
                   'predicted_points', 'projected_minutes', 'captaincy_score']
        cap_cols = [c for c in cap_cols if c in cap_top.columns]
        _safe_print(cap_top[cap_cols].to_string(index=False))

        # 5. Prediction Surprises
        if 'total_points_mean_last_3' in df_pred.columns:
            print("\n--- Top 10 Prediction Surprises ---")
            df_pred['surprise_delta'] = df_pred['predicted_points'] - df_pred['total_points_mean_last_3']
            surprises = df_pred[df_pred.get('minutes_prob', pd.Series([1]*len(df_pred))) > 0]
            surprises = surprises.sort_values('surprise_delta', ascending=False).head(10)
            _safe_print(surprises[['web_name', 'predicted_points', 'total_points_mean_last_3', 'surprise_delta']].to_string(index=False))

        # 6. Feature Drift Report
        if self._train_feature_means:
            print("\n--- Feature Drift Report (current vs training means) ---")
            num_features = [f for f in self.features_list if f in df_pred.columns and df_pred[f].dtype.kind in 'ifc']
            drift_rows = []
            for f in num_features:
                train_mean = self._train_feature_means.get(f, 0)
                curr_mean = df_pred[f].mean()
                if train_mean != 0:
                    pct_drift = ((curr_mean - train_mean) / abs(train_mean)) * 100
                else:
                    pct_drift = 0.0
                drift_rows.append({'feature': f, 'train_mean': round(train_mean, 4), 
                                  'current_mean': round(curr_mean, 4), 'drift_%': round(pct_drift, 1)})
            df_drift = pd.DataFrame(drift_rows)
            # Show only features with >20% drift
            significant_drift = df_drift[df_drift['drift_%'].abs() > 20].sort_values('drift_%', key=abs, ascending=False)
            if not significant_drift.empty:
                print(f"  {len(significant_drift)} features with >20% drift detected:")
                print(significant_drift.to_string(index=False))
            else:
                print("  No significant feature drift detected (all within 20%).")

        # 7. Save prediction audit CSV
        reports_dir = os.path.join("data", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        save_cols = ['id', 'web_name', 'team', 'position', 'price', 'next_opponent',
                    'projected_minutes', 'start_probability', 'predicted_points', 'captaincy_score']
        save_cols = [c for c in save_cols if c in df_pred.columns]
        
        report_path = os.path.join(reports_dir, f"predictions_gw{gw}.csv")
        df_pred[save_cols].sort_values('predicted_points', ascending=False).to_csv(report_path, index=False)
        print(f"\n--- Prediction CSV saved to {report_path} ---")

        # 8. Prediction mode status
        print(f"\nPrediction Mode: {self.prediction_mode}")
        print(f"Odds Confidence: {self.odds_confidence}")
        if self.prediction_warnings:
            for w in self.prediction_warnings:
                print(f"  [!] {w}")

    def _emergency_heuristic(self, df_features, reason="Unknown"):
        """Emergency fallback if model file is lost. LOUD warnings."""
        self.prediction_mode = "fallback"
        warning = f"[FALLBACK] {reason}. Using heuristic instead of ML model."
        self.prediction_warnings.append(warning)
        
        # Print loud warnings
        print("\n" + "!" * 60)
        print("  !!!  EMERGENCY HEURISTIC FALLBACK ACTIVE  !!!")
        print("!" * 60)
        print(f"  Reason: {reason}")
        print(f"  Predictions will be SIGNIFICANTLY less accurate.")
        print(f"  Fix: Run `python src/model/predictor.py` to retrain.")
        print("!" * 60 + "\n")
        
        if 'total_points_mean_last_3' in df_features.columns:
            pts = df_features['total_points_mean_last_3'].fillna(0)
            xg = df_features.get('expected_goal_involvements_mean_last_3', pd.Series([0]*len(df_features))).fillna(0)
            preds = pts * 0.4 + xg * 2.0
        else:
            preds = df_features.get('total_points', pd.Series([0]*len(df_features))) * 0.1
            
        df_features['predicted_points'] = preds.clip(lower=0) if hasattr(preds, 'clip') else max(preds, 0)
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
    
    # Run full audit
    if os.path.exists("data/processed/player_features.parquet") and os.path.exists("data/processed/historical_features.parquet"):
        df_features = pd.read_parquet("data/processed/player_features.parquet")
        df_train = pd.read_parquet("data/processed/historical_features.parquet")
        predictor.generate_audit_report(df_train, df_features)
    else:
        print("Run history_builder.py and processor.py first to generate data.")
