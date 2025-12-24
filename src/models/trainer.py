
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def get_model(name: str, config: dict):
        if name == 'xgboost':
            return xgb.XGBRegressor(**config['models']['xgboost'])
        elif name == 'lightgbm':
            return lgb.LGBMRegressor(**config['models']['lightgbm'], verbose=-1)
        elif name == 'catboost':
            return CatBoostRegressor(**config['models']['catboost'], verbose=0)
        else:
            raise ValueError(f"Unknown model: {name}")

class Trainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Starting training...")
        
        output_dir = Path(self.config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train multiple models as per plan (or just one for now, let's do all 3)
        model_names = ['xgboost', 'lightgbm', 'catboost']
        
        results = {}
        
        for name in model_names:
            logger.info(f"Training {name}...")
            model = ModelFactory.get_model(name, self.config)
            
            # Simple CV
            kf = KFold(n_splits=self.config['training']['cv_folds'], shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = np.sqrt(mean_squared_error(y_val, preds))
                cv_scores.append(score)
            
            avg_rmse = np.mean(cv_scores)
            logger.info(f"{name} CV RMSE: {avg_rmse:.4f}")
            results[name] = avg_rmse
            
            # Refit on full data
            model.fit(X, y)
            self.models[name] = model
            
            # Save
            joblib.dump(model, output_dir / f"{name}_model.joblib")
            
        logger.info(f"Training complete. Results: {results}")
        return results
