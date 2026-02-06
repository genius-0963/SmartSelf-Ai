"""
SmartShelf AI - Base ML Model Class

Abstract base class for all ML models with common functionality
including training, evaluation, serialization, and monitoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all SmartShelf AI ML models.
    
    Provides common functionality for:
    - Model training and evaluation
    - Serialization and deserialization
    - Performance monitoring
    - Feature engineering pipeline
    - Prediction with confidence intervals
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Unique identifier for the model
            config: Model configuration parameters
        """
        self.model_name = model_name
        self.model_version = "1.0.0"
        self.config = config or {}
        self.model = None
        self.is_trained = False
        
        # Performance metrics
        self.training_metrics = {}
        self.validation_metrics = {}
        self.feature_importance = {}
        
        # Metadata
        self.created_at = datetime.utcnow()
        self.last_trained = None
        self.training_data_shape = None
        self.feature_columns = []
        self.target_column = None
        
        # Model paths
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / f"{model_name}.pkl"
        self.metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        logger.info(f"Initialized {model_name} model")
    
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/prediction.
        
        Args:
            data: Raw input data
            
        Returns:
            Tuple of (features, target)
        """
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the ML model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        pass
    
    def predict_with_confidence(self, X: pd.DataFrame, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Feature matrix
            confidence_level: Confidence interval level (0-1)
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Default implementation - just return point predictions
        predictions = self.predict(X)
        std_error = np.std(predictions) * np.ones_like(predictions)
        
        # Simple confidence interval calculation
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin_error = z_score * std_error
        
        lower_bound = predictions - margin_error
        upper_bound = predictions + margin_error
        
        return predictions, lower_bound, upper_bound
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate common regression metrics
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y))
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y - predictions) / np.maximum(y, 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        }
        
        logger.info(f"Evaluation metrics for {self.model_name}: {metrics}")
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with lists of metrics for each fold
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train and evaluate
            self.train(X_train, y_train)
            metrics = self.evaluate(X_val, y_val)
            fold_metrics.append(metrics)
        
        # Aggregate metrics
        cv_results = {}
        for metric_name in fold_metrics[0].keys():
            cv_results[metric_name] = [fold[metric_name] for fold in fold_metrics]
        
        return cv_results
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Optional custom path to save model
            
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_path = Path(path) if path else self.model_path
        
        # Save model using joblib (better for sklearn objects)
        joblib.dump(self.model, save_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'config': self.config,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'feature_importance': self.feature_importance,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'training_data_shape': self.training_data_shape,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
        return str(save_path)
    
    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Optional custom path to load model from
        """
        load_path = Path(path) if path else self.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Load model
        self.model = joblib.load(load_path)
        self.is_trained = True
        
        # Load metadata
        metadata_path = load_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_name = metadata.get('model_name', self.model_name)
            self.model_version = metadata.get('model_version', '1.0.0')
            self.config = metadata.get('config', {})
            self.training_metrics = metadata.get('training_metrics', {})
            self.validation_metrics = metadata.get('validation_metrics', {})
            self.feature_importance = metadata.get('feature_importance', {})
            self.created_at = datetime.fromisoformat(metadata.get('created_at', '2024-01-01'))
            self.last_trained = datetime.fromisoformat(metadata['last_trained']) if metadata.get('last_trained') else None
            self.training_data_shape = metadata.get('training_data_shape')
            self.feature_columns = metadata.get('feature_columns', [])
            self.target_column = metadata.get('target_column')
        
        logger.info(f"Model loaded from {load_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.feature_importance
    
    def explain_prediction(self, X: pd.DataFrame, prediction_index: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction (XAI - Explainable AI).
        
        Args:
            X: Feature matrix
            prediction_index: Index of the prediction to explain
            
        Returns:
            Dictionary with explanation components
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before explaining predictions")
        
        # Default implementation - return basic info
        explanation = {
            'prediction': self.predict(X.iloc[[prediction_index]])[0],
            'feature_values': X.iloc[prediction_index].to_dict(),
            'feature_importance': self.feature_importance,
            'model_name': self.model_name,
            'explanation_method': 'basic_feature_importance'
        }
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata and status
        """
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'training_data_shape': self.training_data_shape,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'feature_importance': self.feature_importance,
            'model_path': str(self.model_path),
            'metadata_path': str(self.metadata_path)
        }
    
    def log_training_step(self, step: str, metrics: Dict[str, float]) -> None:
        """
        Log training step metrics for monitoring.
        
        Args:
            step: Description of training step
            metrics: Dictionary of metrics for this step
        """
        logger.info(f"Training step '{step}': {metrics}")
        
        # Store in training metrics
        if 'training_steps' not in self.training_metrics:
            self.training_metrics['training_steps'] = {}
        
        self.training_metrics['training_steps'][step] = {
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format and quality.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid, raises exception otherwise
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        if data.isnull().any().any():
            logger.warning("Input data contains missing values")
        
        # Check for required columns if model is trained
        if self.is_trained and self.feature_columns:
            missing_cols = set(self.feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        return True
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before prediction.
        
        Args:
            X: Raw feature matrix
            
        Returns:
            Preprocessed feature matrix
        """
        # Default implementation - return as-is
        # Override in subclasses for specific preprocessing
        return X
    
    def postprocess_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Postprocess predictions after model output.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Postprocessed predictions
        """
        # Default implementation - return as-is
        # Override in subclasses for specific postprocessing
        return predictions
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.model_name}', status='{status}')"
