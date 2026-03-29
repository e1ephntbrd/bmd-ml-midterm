import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class BankDataTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for the Bank Marketing dataset to handle 
    categorical grouping and feature engineering.
    """
    def __init__(self) -> None:
        """Initializes the transformer."""
        pass
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BankDataTransformer':
        """
        Fits the transformer. This is a stateless transformer.
        
        Args:
            X: Input features.
            y: Target variable (not used).
            
        Returns:
            Self (BankDataTransformer).
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning and feature engineering logic to the dataframe.
        
        Args:
            X: Dataframe to be transformed.
            
        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        X_copy = X.copy()
        
        # 1. Handle 'unknown' by replacing it with the mode of the column
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object':
                valid_data = X_copy[col][X_copy[col] != 'unknown']
                if not valid_data.empty:
                    mode_val = valid_data.mode()[0]
                    X_copy[col] = X_copy[col].replace('unknown', mode_val)
        
        # 2. Group education levels into broader categories
        if 'education' in X_copy.columns:
            X_copy['education'] = X_copy['education'].replace({
                'basic.4y': 'basic', 
                'basic.6y': 'basic', 
                'basic.9y': 'basic'
            })
            
        # 3. Cap the number of contacts to reduce the impact of outliers
        if 'campaign' in X_copy.columns:
            X_copy['campaign'] = X_copy['campaign'].clip(upper=10)
            
        # 4. Create binary flag if the client was previously contacted
        if 'pdays' in X_copy.columns:
            X_copy['was_contacted'] = (X_copy['pdays'] != 999).astype(int)
            
        return X_copy

def get_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Creates a standard ColumnTransformer pipeline.
    
    Args:
        numeric_features: List of numerical column names.
        categorical_features: List of categorical column names.
        
    Returns:
        ColumnTransformer: Configured preprocessor object.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])