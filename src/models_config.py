from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def get_models_dict(ratio: float = 8.0) -> Dict[str, Any]:
    """
    Returns a dictionary of pre-configured machine learning models.
    
    Args:
        ratio: The scale_pos_weight for XGBoost to handle class imbalance. 
               Defaults to 8.0 based on dataset observation.
               
    Returns:
        Dict[str, Any]: Dictionary where keys are model names and values are model objects.
    """
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42
        ),
        'kNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=7, 
            class_weight='balanced', 
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=ratio, 
            eval_metric='logloss', 
            random_state=42
        )
    }
