import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib

class PointsPredictor:
    PREMIUM_PRICE_THRESHOLD = 8.0
    PREMIUM_BOOST_VALUE = 0.5

    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "lgb_points.pkl")
        self.model = None

    def train(self, df_train):
        """Trains a LightGBM model to predict points."""
        # df_train must have 'actual_points' target
        X = df_train.drop(columns=['actual_points', 'id', 'web_name'])
        y = df_train['actual_points']
        
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05
        }
        self.model = lgb.train(params, train_data, num_boost_round=100)
        joblib.dump(self.model, self.model_path)
        print("Model trained and saved.")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def predict(self, df_features):
        """Predicts points for the next gameweek."""
        # Ensure we have the same columns as training
        # For this skeleton, we'll assume we use a heuristic if no model found
        
        # Heuristic: More aggressive weighting towards Understat data (xG/xA), Form, and Fixtures
        # This replaces the conservative 'cold start' fallback.
        # Feature typical ranges: form (0-9), pts_per_game (~1-10), xGI_per_90 (0.3-1.5).
        # Weights (0.2-2.5) scale these to similar magnitudes, favoring underlying threat.
        # Fixture difficulty is bounded [1-5], so (5 - difficulty) yields a 0-4 boost.
        
        # Calculate Expected Goal Involvement per 90 (if columns exist)
        if 'xG_per_90' in df_features.columns and 'xA_per_90' in df_features.columns:
            xGI_per_90 = df_features['xG_per_90'].fillna(0) + df_features['xA_per_90'].fillna(0)
        else:
            xGI_per_90 = 0
            
        # Ensure base columns exist
        form = df_features['form'].fillna(0) if 'form' in df_features.columns else 0
        ppg = df_features['points_per_game'].fillna(0) if 'points_per_game' in df_features.columns else 0
        diff = df_features['fixture_difficulty'].fillna(3) if 'fixture_difficulty' in df_features.columns else 3
        
        preds = (
            form * 0.3 + 
            ppg * 0.2 + 
            xGI_per_90 * 2.5 + # Heavy weight on underlying threat
            (5 - diff) * 0.6
        )
        
        # Premium player bump (talented players have higher ceilings uncaptured by linear stats)
        price_col = df_features['price'].fillna(0) if 'price' in df_features.columns else 0
        premium_boost = (price_col >= self.PREMIUM_PRICE_THRESHOLD).astype(float) * self.PREMIUM_BOOST_VALUE
        preds = preds + premium_boost
        
        # Adjust for Minutes Risk
        # If minutes_prob is completely missing, default to 100% (1.0).
        # We also enforce a clip to prevent strange values.
        if 'minutes_prob' in df_features.columns:
            minutes_prob = df_features['minutes_prob'].fillna(1.0).clip(lower=0, upper=1)
        else:
            minutes_prob = 1.0
            
        preds = preds * minutes_prob
        
        # If we had a real model:
        # if self.load_model():
        #     X = df_features[self.model.feature_name()]
        #     preds = self.model.predict(X)
        
        # Store predictions
        df_features['predicted_points'] = preds
        return df_features

if __name__ == "__main__":
    # Test
    df = pd.read_parquet("data/processed/player_features.parquet")
    predictor = PointsPredictor()
    result = predictor.predict(df)
    print(result[['web_name', 'predicted_points']].sort_values('predicted_points', ascending=False).head(10))
