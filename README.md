# Credit Card Fraud Detection System

A comprehensive machine learning system for detecting fraudulent credit card transactions using various algorithms and techniques.

## Features

- **Multiple ML Algorithms**: Logistic Regression, Random Forest, XGBoost, Isolation Forest
- **Data Preprocessing**: Handling imbalanced data, feature scaling, outlier detection
- **Model Evaluation**: Comprehensive metrics (Precision, Recall, F1-Score, ROC-AUC)
- **Real-time Prediction**: API for real-time fraud detection
- **Visualization**: Performance metrics and feature importance plots

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── creditcard.csv          # Training dataset
│   └── processed_data/         # Preprocessed data
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── models.py               # Model definitions
│   ├── trainer.py              # Model training pipeline
│   ├── evaluator.py            # Model evaluation metrics
│   └── predictor.py            # Real-time prediction
├── api/
│   ├── app.py                  # Flask API server
│   └── config.py               # Configuration
├── tests/
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_predictor.py
├── requirements.txt
├── config.yaml
└── main.py                     # Main execution script
```

## Installation

```bash
git clone https://github.com/vamsigavara-png/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

## Quick Start

```bash
# Train models
python main.py --train

# Run predictions
python main.py --predict

# Start API server
python api/app.py
```

## Dataset

The system uses the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) which contains transactions from European cardholders.

- **Records**: ~284,807 transactions
- **Fraudulent**: ~492 (0.17%)
- **Features**: 30 (28 PCA-transformed, Amount, Time)

## Methodology

### 1. Data Preprocessing
- Handle class imbalance using SMOTE and undersampling
- Feature scaling with StandardScaler and RobustScaler
- Outlier detection and removal

### 2. Model Training
- **Logistic Regression**: Baseline model
- **Random Forest**: Ensemble learning
- **XGBoost**: Gradient boosting
- **Isolation Forest**: Anomaly detection

### 3. Evaluation Metrics
- Precision, Recall, F1-Score
- ROC-AUC and Precision-Recall curves
- Confusion Matrix
- Cross-validation scores

## API Endpoints

### POST `/predict`
Predict fraud probability for a transaction.

**Request:**
```json
{
  "amount": 100.50,
  "features": [0.1, -0.2, 0.3, ...]
}
```

**Response:**
```json
{
  "fraud_probability": 0.15,
  "is_fraud": false,
  "confidence": 0.95
}
```

## Performance

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|----------|
| Logistic Regression | 0.85 | 0.72 | 0.78 | 0.88 |
| Random Forest | 0.92 | 0.81 | 0.86 | 0.94 |
| XGBoost | 0.94 | 0.83 | 0.88 | 0.96 |
| Isolation Forest | 0.78 | 0.65 | 0.71 | 0.82 |

## Handling Class Imbalance

- **SMOTE**: Synthetic Minority Over-sampling Technique
- **Undersampling**: Random undersampling of majority class
- **Threshold Tuning**: Adjusting decision threshold for optimal precision-recall tradeoff

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08.pdf)

## Author

vamsigavara-png
