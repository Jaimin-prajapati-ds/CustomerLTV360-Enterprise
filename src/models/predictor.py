
import pandas as pd
import joblib
from pathlib import Path
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.output_dir = Path(config['training']['output_dir'])
        
    def load_models(self):
        """Load all trained models."""
        model_names = ['xgboost', 'lightgbm', 'catboost']
        for name in model_names:
            model_path = self.output_dir / f"{name}_model.joblib"
            if model_path.exists():
                logger.info(f"Loading {name} model from {model_path}")
                self.models[name] = joblib.load(model_path)
            else:
                logger.warning(f"Model {name} not found at {model_path}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using all loaded models."""
        if not self.models:
            self.load_models()
        
        if not self.models:
            raise ValueError("No models loaded. Please train models first.")
            
        predictions = pd.DataFrame(index=X.index)
        
        for name, model in self.models.items():
            logger.info(f"Predicting with {name}...")
            predictions[f'pred_{name}'] = model.predict(X)
            
        # Ensemble average
        predictions['pred_avg'] = predictions.mean(axis=1)
        return predictions
