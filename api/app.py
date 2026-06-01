"""
Flask API for credit card fraud detection.
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import logging

from src.predictor import FraudPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MODELS_PATH = os.getenv('MODELS_PATH', 'models/')
MODEL_NAME = os.getenv('MODEL_NAME', 'xgb')

# Global variables for loaded models
predictor = None


def load_model():
    """Load the trained model and scaler."""
    global predictor
    
    model_path = f'{MODELS_PATH}{MODEL_NAME}_model.pkl'
    scaler_path = f'{MODELS_PATH}scaler.pkl'
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return False
    
    try:
        predictor = FraudPredictor.load_predictor(model_path, scaler_path)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_NAME,
        'models_path': MODELS_PATH
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud probability for a transaction.
    
    Request JSON:
    {
        "features": [v1, v2, ..., v30],  # 30 feature values
        "threshold": 0.5  # Optional, default 0.5
    }
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please ensure the model has been trained'
        }), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        if 'features' not in data:
            return jsonify({
                'error': 'Missing features',
                'message': 'Please provide features array with 30 values'
            }), 400
        
        features = np.array(data['features'], dtype=np.float32)
        threshold = data.get('threshold', 0.5)
        
        # Validate features shape
        if len(features) != 30:
            return jsonify({
                'error': 'Invalid features shape',
                'message': f'Expected 30 features, got {len(features)}'
            }), 400
        
        # Make prediction
        result = predictor.predict_single(features, threshold=threshold)
        
        return jsonify({
            'success': True,
            'fraud_probability': result['fraud_probability'],
            'is_fraud': result['is_fraud'],
            'confidence': result['confidence'],
            'threshold': threshold
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict fraud for multiple transactions.
    
    Request JSON:
    {
        "transactions": [
            [v1, v2, ..., v30],
            [v1, v2, ..., v30],
            ...
        ],
        "threshold": 0.5  # Optional
    }
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please ensure the model has been trained'
        }), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        if 'transactions' not in data:
            return jsonify({
                'error': 'Missing transactions',
                'message': 'Please provide transactions array'
            }), 400
        
        transactions = np.array(data['transactions'], dtype=np.float32)
        threshold = data.get('threshold', 0.5)
        
        # Validate shape
        if transactions.shape[1] != 30:
            return jsonify({
                'error': 'Invalid transaction shape',
                'message': f'Expected 30 features per transaction, got {transactions.shape[1]}'
            }), 400
        
        # Make predictions
        predictions = predictor.predict_batch(transactions, threshold=threshold)
        probabilities = predictor.predict_proba(transactions)
        
        return jsonify({
            'success': True,
            'total_transactions': len(transactions),
            'fraud_detected': int(np.sum(predictions)),
            'non_fraud': int(len(transactions) - np.sum(predictions)),
            'fraud_rate': float(np.mean(predictions)),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'threshold': threshold
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'model_name': MODEL_NAME,
        'model_type': predictor.model.__class__.__name__,
        'has_scaler': predictor.scaler is not None,
        'input_features': 30,
        'output_type': 'binary_classification'
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            'GET /health',
            'GET /model-info',
            'POST /predict',
            'POST /predict-batch'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.warning("Could not load model, API will be available but predictions will fail")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
