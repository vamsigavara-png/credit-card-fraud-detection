"""
Real-time prediction module for credit card fraud detection.
"""

import numpy as np
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudPredictor:
    """Make predictions using trained models."""
    
    def __init__(self, model, scaler=None):
        """
        Initialize FraudPredictor.
        
        Args:
            model: Trained model
            scaler: Feature scaler (optional)
        """
        self.model = model
        self.scaler = scaler
    
    def predict_single(self, features: np.ndarray, 
                      threshold: float = 0.5) -> dict:
        """
        Predict fraud for a single transaction.
        
        Args:
            features: Feature vector for the transaction
            threshold: Probability threshold for fraud classification
            
        Returns:
            Dictionary with prediction results
        """
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Get prediction
        prediction = self.model.predict(features)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            fraud_prob = probabilities[1]
        else:
            fraud_prob = float(prediction)
        
        # Apply threshold
        is_fraud = fraud_prob >= threshold
        
        return {
            'fraud_probability': float(fraud_prob),
            'is_fraud': bool(is_fraud),
            'confidence': float(max(probabilities)) if hasattr(self.model, 'predict_proba') else None,
            'threshold_used': threshold
        }
    
    def predict_batch(self, features: np.ndarray,
                     threshold: float = 0.5) -> np.ndarray:
        """
        Predict fraud for multiple transactions.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            threshold: Probability threshold for fraud classification
            
        Returns:
            Array of fraud predictions
        """
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
        else:
            predictions = self.model.predict(features)
        
        return predictions
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get fraud probabilities for transactions.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of fraud probabilities
        """
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(features)[:, 1]
        else:
            raise ValueError("Model does not support probability predictions")
    
    def predict_with_decision_function(self, features: np.ndarray) -> np.ndarray:
        """
        Get decision function scores (useful for Isolation Forest).
        
        Args:
            features: Feature matrix
            
        Returns:
            Array of decision scores
        """
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(features)
        elif hasattr(self.model, 'score_samples'):
            return self.model.score_samples(features)
        else:
            raise ValueError("Model does not support decision function")
    
    @staticmethod
    def load_predictor(model_path: str, scaler_path: str = None):
        """
        Load predictor from saved files.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler (optional)
            
        Returns:
            FraudPredictor instance
        """
        model = joblib.load(model_path)
        scaler = None
        
        if scaler_path:
            scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded model from {model_path}")
        if scaler_path:
            logger.info(f"Loaded scaler from {scaler_path}")
        
        return FraudPredictor(model, scaler)
    
    def save_predictor(self, model_path: str, scaler_path: str = None) -> None:
        """
        Save predictor to disk.
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler (optional)
        """
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        if self.scaler and scaler_path:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")


class BatchPredictor:
    """Batch prediction with detailed reporting."""
    
    def __init__(self, predictor: FraudPredictor):
        """
        Initialize BatchPredictor.
        
        Args:
            predictor: FraudPredictor instance
        """
        self.predictor = predictor
    
    def predict_dataframe(self, df: pd.DataFrame,
                         feature_cols: list = None,
                         threshold: float = 0.5) -> pd.DataFrame:
        """
        Predict on a DataFrame.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            threshold: Probability threshold
            
        Returns:
            DataFrame with predictions
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != 'Class']
        
        features = df[feature_cols].values
        predictions = self.predictor.predict_batch(features, threshold)
        
        result_df = df.copy()
        result_df['fraud_prediction'] = predictions
        
        if hasattr(self.predictor.model, 'predict_proba'):
            probabilities = self.predictor.predict_proba(features)
            result_df['fraud_probability'] = probabilities
        
        return result_df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get prediction statistics.
        
        Args:
            df: DataFrame with predictions
            
        Returns:
            Dictionary of statistics
        """
        total = len(df)
        fraud_count = df['fraud_prediction'].sum()
        fraud_percentage = (fraud_count / total) * 100
        
        stats = {
            'total_transactions': total,
            'fraud_detected': int(fraud_count),
            'non_fraud': int(total - fraud_count),
            'fraud_percentage': fraud_percentage,
            'avg_fraud_probability': df['fraud_probability'].mean() if 'fraud_probability' in df.columns else None
        }
        
        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  Total transactions: {stats['total_transactions']}")
        logger.info(f"  Fraud detected: {stats['fraud_detected']}")
        logger.info(f"  Fraud rate: {stats['fraud_percentage']:.2f}%")
        
        return stats
