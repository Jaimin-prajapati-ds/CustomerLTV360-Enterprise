
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

class FeatureGenerator:
    def __init__(self, config):
        self.config = config
        self.preprocessor = None
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        logger.info("Generating features...")
        
        # Split features and target if present
        target_col = self.config['features']['target_col']
        
        X = df.drop(columns=[target_col, 'customer_id', 'transaction_date'], errors='ignore')
        
        # Define preprocessor
        cat_cols = [c for c in self.config['features']['categorical_cols'] if c in X.columns]
        num_cols = [c for c in self.config['features']['numerical_cols'] if c in X.columns]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        num_names = num_cols
        cat_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        feature_names = list(num_names) + list(cat_names)
        
        return pd.DataFrame(X_processed, columns=feature_names)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data."""
        if self.preprocessor is None:
            raise ValueError("FeatureGenerator must be fitted before transform.")
            
        target_col = self.config['features']['target_col']
        X = df.drop(columns=[target_col, 'customer_id', 'transaction_date'], errors='ignore')
        
        X_processed = self.preprocessor.transform(X)
        
        # Re-construct dataframe (logic repeated for simplicity, in prod should be shared)
        cat_cols = [c for c in self.config['features']['categorical_cols'] if c in X.columns]
        num_cols = [c for c in self.config['features']['numerical_cols'] if c in X.columns]
        num_names = num_cols
        cat_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        feature_names = list(num_names) + list(cat_names)
        
        return pd.DataFrame(X_processed, columns=feature_names)
