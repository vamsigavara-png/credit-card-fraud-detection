"""
Model training pipeline.
"""

import numpy as np
import joblib
from typing import List, Dict, Any
import logging

from .models import FraudDetectionModels
from .evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage multiple fraud detection models."""
    
    def __init__(self):
        """Initialize ModelTrainer."""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_model(self, model_name: str, X_train: np.ndarray, 
                   y_train: np.ndarray, **kwargs):
        """
        Train a single model.
        
        Args:
            model_name: Name of the model ('lr', 'rf', 'xgb', 'if')
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional model parameters
        """
        logger.info(f"\nTraining {model_name} model...")
        
        model = FraudDetectionModels.get_model(model_name, **kwargs)
        
        # Check if model is Isolation Forest (unsupervised)
        if model_name == 'if':
            model.fit(X_train)
            logger.info(f"{model_name} model trained")
        else:
            model.fit(X_train, y_train)
            logger.info(f"{model_name} model trained")
        
        self.models[model_name] = model
    
    def train_multiple_models(self, X_train: np.ndarray, y_train: np.ndarray,
                             model_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Train multiple models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_configs: Dictionary of model names to their configs
            
        Example:
            model_configs = {
                'lr': {'max_iter': 1000},
                'rf': {'n_estimators': 100},
                'xgb': {'n_estimators': 100, 'learning_rate': 0.1}
            }
        """
        for model_name, config in model_configs.items():
            self.train_model(model_name, X_train, y_train, **config)
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results for each model
        """
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name} model...")
            
            results = ModelEvaluator.evaluate_model(model, X_test, y_test, 
                                                   model_name)
            self.results[model_name] = results
        
        return self.results
    
    def select_best_model(self, metric: str = 'f1_score') -> str:
        """
        Select best model based on specified metric.
        
        Args:
            metric: Metric to use for selection ('f1_score', 'precision', 
                   'recall', 'roc_auc')
            
        Returns:
            Name of the best model
        """
        best_score = -1
        best_name = None
        
        for model_name, results in self.results.items():
            if metric in results['metrics']:
                score = results['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"\nBest model: {best_name} "
                   f"({metric}: {best_score:.4f})")
        
        return best_name
    
    def get_model(self, model_name: str):
        """
        Get a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. "
                           f"Available: {list(self.models.keys())}")
        
        return self.models[model_name]
    
    def get_best_model(self):
        """Get the best trained model."""
        if self.best_model is None:
            raise ValueError("No best model selected. "
                           "Run evaluate_models() and select_best_model() first.")
        
        return self.best_model
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the saved model
        """
        model = joblib.load(filepath)
        self.models[model_name] = model
        logger.info(f"Model loaded from {filepath}")
    
    def compare_models(self) -> Dict:
        """
        Compare all models side by side.
        
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            comparison[model_name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics.get('roc_auc', 'N/A')
            }
        
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        
        for model_name, metrics in comparison.items():
            logger.info(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric:12s}: {value:.4f}")
                else:
                    logger.info(f"  {metric:12s}: {value}")
        
        return comparison
