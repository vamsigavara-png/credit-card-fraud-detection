"""
Data loading and preprocessing module for credit card fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and initial preprocessing."""
    
    def __init__(self, filepath: str):
        """
        Initialize DataLoader.
        
        Args:
            filepath: Path to CSV file containing transaction data
        """
        self.filepath = filepath
        self.df = None
        self.X = None
        self.y = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            self.df = pd.read_csv(self.filepath)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Fraud distribution:\n{self.df['Class'].value_counts()}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.filepath}")
            raise
    
    def explore_data(self) -> dict:
        """Explore data characteristics."""
        if self.df is None:
            self.load_data()
        
        stats = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'fraud_percentage': (self.df['Class'].sum() / len(self.df)) * 100,
            'data_types': self.df.dtypes.to_dict(),
            'description': self.df.describe().to_dict()
        }
        
        logger.info(f"Dataset has {stats['missing_values']} missing values")
        logger.info(f"Fraud cases: {stats['fraud_percentage']:.2f}%")
        
        return stats
    
    def handle_outliers(self, threshold: float = 3.0) -> None:
        """
        Remove outliers using Z-score method.
        
        Args:
            threshold: Z-score threshold (default: 3.0)
        """
        if self.df is None:
            self.load_data()
        
        numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns if col != 'Class']
        z_scores = np.abs((self.df[numeric_cols] - self.df[numeric_cols].mean()) / 
                         self.df[numeric_cols].std())
        
        outlier_mask = (z_scores > threshold).any(axis=1)
        initial_shape = self.df.shape
        self.df = self.df[~outlier_mask]
        
        logger.info(f"Removed {initial_shape[0] - len(self.df)} outliers")
    
    def prepare_features(self, test_size: float = 0.2, 
                        random_state: int = 42) -> tuple:
        """
        Prepare features and labels for modeling.
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            self.load_data()
        
        # Separate features and target
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test


class FeatureScaler:
    """Handle feature scaling and normalization."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize FeatureScaler.
        
        Args:
            method: 'standard' for StandardScaler or 'robust' for RobustScaler
        """
        self.method = method
        self.scaler = None
        self._init_scaler()
    
    def _init_scaler(self):
        """Initialize the appropriate scaler."""
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
    
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training data and transform."""
        return self.scaler.fit_transform(X_train)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler."""
        return self.scaler.transform(X)


class ImbalanceHandler:
    """Handle class imbalance using SMOTE and undersampling."""
    
    def __init__(self, sampling_strategy: float = 0.5, 
                 random_state: int = 42):
        """
        Initialize ImbalanceHandler.
        
        Args:
            sampling_strategy: Target ratio of minority to majority class
            random_state: Random state for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.pipeline = None
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize SMOTE + Undersampling pipeline."""
        self.pipeline = ImbPipeline([
            ('smote', SMOTE(sampling_strategy=self.sampling_strategy,
                           random_state=self.random_state)),
            ('undersampler', RandomUnderSampler(sampling_strategy=1.0,
                                               random_state=self.random_state))
        ])
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Apply SMOTE and undersampling to training data."""
        X_resampled, y_resampled = self.pipeline.fit_resample(X, y)
        
        logger.info(f"Resampled data shape: {X_resampled.shape}")
        logger.info(f"Class distribution after resampling: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
