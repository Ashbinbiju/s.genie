import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib

class PointsPredictor:
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
        
        # Heuristic: Form * 0.4 + PPG * 0.4 + (5 - FixtureDiff) * 0.2
        # This is a 'cold start' fallback
        
        preds = (
            df_features['form'] * 0.4 + 
            df_features['points_per_game'] * 0.4 + 
            (5 - df_features['fixture_difficulty']) * 0.5
        )
        
        # Adjust for Minutes Risk
        preds = preds * df_features['minutes_prob']
        
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
