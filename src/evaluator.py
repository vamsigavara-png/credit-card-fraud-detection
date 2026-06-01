"""
Model evaluation and metrics module.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_pred_proba: np.ndarray = None) -> dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': (y_pred == y_true).mean(),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> str:
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        report = classification_report(y_true, y_pred, 
                                       target_names=['Non-Fraud', 'Fraud'])
        logger.info(f"\nClassification Report:\n{report}")
        return report
    
    @staticmethod
    def get_confusion_matrix_dict(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> dict:
        """
        Get confusion matrix components.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with TP, TN, FP, FN
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp),
            'sensitivity': tp / (tp + fn)
        }
    
    @staticmethod
    def calculate_roc_curve(y_true: np.ndarray,
                           y_pred_proba: np.ndarray) -> dict:
        """
        Calculate ROC curve data.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with FPR, TPR, thresholds, and AUC
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc
        }
    
    @staticmethod
    def calculate_precision_recall_curve(y_true: np.ndarray,
                                        y_pred_proba: np.ndarray) -> dict:
        """
        Calculate Precision-Recall curve data.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with precision, recall, thresholds
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'avg_precision': average_precision_score(y_true, y_pred_proba)
        }
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray,
                              y_pred_proba: np.ndarray,
                              metric: str = 'f1') -> float:
        """
        Find optimal probability threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        thresholds = np.arange(0.1, 1.0, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal {metric} threshold: {optimal_threshold:.3f} "
                   f"(score: {scores[optimal_idx]:.3f})")
        
        return optimal_threshold
    
    @staticmethod
    def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = '') -> dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of all evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = ModelEvaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        confusion_dict = ModelEvaluator.get_confusion_matrix_dict(y_test, y_pred)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {model_name or model.__class__.__name__}")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        ModelEvaluator.print_classification_report(y_test, y_pred)
        
        results = {
            'metrics': metrics,
            'confusion_matrix': confusion_dict
        }
        
        if y_pred_proba is not None:
            results['roc_curve'] = ModelEvaluator.calculate_roc_curve(y_test, y_pred_proba)
            results['pr_curve'] = ModelEvaluator.calculate_precision_recall_curve(y_test, y_pred_proba)
            results['optimal_threshold'] = ModelEvaluator.find_optimal_threshold(y_test, y_pred_proba)
        
        return results
