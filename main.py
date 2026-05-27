#!/usr/bin/env python
"""
Main execution script for credit card fraud detection system.
"""

import argparse
import logging
import numpy as np
from pathlib import Path

from src.data_loader import DataLoader, FeatureScaler, ImbalanceHandler
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.predictor import FraudPredictor, BatchPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_pipeline(data_path: str, models_path: str = 'models/'):
    """
    Execute the complete training pipeline.
    
    Args:
        data_path: Path to the credit card dataset
        models_path: Path to save trained models
    """
    logger.info("="*80)
    logger.info("CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    logger.info("="*80)
    
    # Create output directory
    Path(models_path).mkdir(parents=True, exist_ok=True)
    
    # 1. Load and explore data
    logger.info("\n1. Loading and exploring data...")
    loader = DataLoader(data_path)
    df = loader.load_data()
    stats = loader.explore_data()
    
    # 2. Handle outliers
    logger.info("\n2. Handling outliers...")
    loader.handle_outliers(threshold=3.0)
    
    # 3. Prepare features
    logger.info("\n3. Preparing features...")
    X_train, X_test, y_train, y_test = loader.prepare_features()
    
    # 4. Scale features
    logger.info("\n4. Scaling features...")
    scaler = FeatureScaler(method='standard')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Handle class imbalance
    logger.info("\n5. Handling class imbalance (SMOTE + Undersampling)...")
    imbalance_handler = ImbalanceHandler(sampling_strategy=0.5)
    X_train_balanced, y_train_balanced = imbalance_handler.fit_resample(
        X_train_scaled, y_train.values
    )
    
    # 6. Train models
    logger.info("\n6. Training models...")
    trainer = ModelTrainer()
    
    model_configs = {
        'lr': {'max_iter': 1000},
        'rf': {'n_estimators': 100, 'max_depth': 15},
        'xgb': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    }
    
    trainer.train_multiple_models(X_train_balanced, y_train_balanced, model_configs)
    
    # 7. Evaluate models
    logger.info("\n7. Evaluating models...")
    trainer.evaluate_models(X_test_scaled, y_test.values)
    
    # 8. Compare models
    logger.info("\n8. Comparing models...")
    comparison = trainer.compare_models()
    
    # 9. Select best model
    logger.info("\n9. Selecting best model...")
    best_model_name = trainer.select_best_model(metric='f1_score')
    
    # 10. Save models and scaler
    logger.info("\n10. Saving models and scaler...")
    trainer.save_model(best_model_name, f'{models_path}{best_model_name}_model.pkl')
    import joblib
    joblib.dump(scaler.scaler, f'{models_path}scaler.pkl')
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return trainer, scaler, X_test_scaled, y_test


def predict_pipeline(models_path: str = 'models/', model_name: str = 'xgb'):
    """
    Execute the prediction pipeline.
    
    Args:
        models_path: Path to load trained models
        model_name: Name of the model to use
    """
    logger.info("\n" + "="*80)
    logger.info("CREDIT CARD FRAUD DETECTION - PREDICTION PIPELINE")
    logger.info("="*80)
    
    # Load predictor
    logger.info(f"\nLoading {model_name} model...")
    predictor = FraudPredictor.load_predictor(
        f'{models_path}{model_name}_model.pkl',
        f'{models_path}scaler.pkl'
    )
    
    # Example prediction
    logger.info("\nMaking example prediction...")
    example_features = np.random.randn(30)  # 30 features
    result = predictor.predict_single(example_features, threshold=0.5)
    
    logger.info(f"\nPrediction Result:")
    logger.info(f"  Fraud Probability: {result['fraud_probability']:.4f}")
    logger.info(f"  Is Fraud: {result['is_fraud']}")
    logger.info(f"  Confidence: {result['confidence']:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Credit Card Fraud Detection System'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'predict'],
        default='train',
        help='Mode to run (default: train)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/creditcard.csv',
        help='Path to credit card dataset (for training)'
    )
    parser.add_argument(
        '--models-path',
        type=str,
        default='models/',
        help='Path to save/load models'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='xgb',
        help='Model name to use for prediction'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_pipeline(args.data, args.models_path)
    elif args.mode == 'predict':
        predict_pipeline(args.models_path, args.model_name)


if __name__ == '__main__':
    main()
