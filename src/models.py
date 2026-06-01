"""
Machine learning models for fraud detection.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionModels:
    """Factory class for creating fraud detection models."""
    
    @staticmethod
    def logistic_regression(random_state: int = 42, 
                           max_iter: int = 1000) -> LogisticRegression:
        """
        Create Logistic Regression model.
        
        Args:
            random_state: Random state for reproducibility
            max_iter: Maximum iterations
            
        Returns:
            Configured LogisticRegression model
        """
        return LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight='balanced',
            solver='lbfgs',
            n_jobs=-1
        )
    
    @staticmethod
    def random_forest(n_estimators: int = 100, 
                     random_state: int = 42,
                     max_depth: int = 15) -> RandomForestClassifier:
        """
        Create Random Forest model.
        
        Args:
            n_estimators: Number of trees
            random_state: Random state for reproducibility
            max_depth: Maximum tree depth
            
        Returns:
            Configured RandomForestClassifier model
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            class_weight='balanced',
            n_jobs=-1,
            verbose=0
        )
    
    @staticmethod
    def xgboost(n_estimators: int = 100,
               random_state: int = 42,
               max_depth: int = 6,
               learning_rate: float = 0.1) -> XGBClassifier:
        """
        Create XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            random_state: Random state for reproducibility
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            
        Returns:
            Configured XGBClassifier model
        """
        return XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=1,  # Adjust for class imbalance
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=0
        )
    
    @staticmethod
    def isolation_forest(contamination: float = 0.001,
                        random_state: int = 42) -> IsolationForest:
        """
        Create Isolation Forest model for anomaly detection.
        
        Args:
            contamination: Expected fraction of anomalies
            random_state: Random state for reproducibility
            
        Returns:
            Configured IsolationForest model
        """
        return IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
    
    @staticmethod
    def get_model(model_name: str, **kwargs):
        """
        Get model by name.
        
        Args:
            model_name: Name of the model ('lr', 'rf', 'xgb', 'if')
            **kwargs: Additional model parameters
            
        Returns:
            Configured model
        """
        models = {
            'lr': FraudDetectionModels.logistic_regression,
            'rf': FraudDetectionModels.random_forest,
            'xgb': FraudDetectionModels.xgboost,
            'if': FraudDetectionModels.isolation_forest
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(models.keys())}")
        
        model_func = models[model_name]
        
        # Filter kwargs to only include valid parameters
        import inspect
        valid_params = inspect.signature(model_func).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in valid_params}
        
        model = model_func(**filtered_kwargs)
        logger.info(f"Created {model_name} model")
        
        return model
