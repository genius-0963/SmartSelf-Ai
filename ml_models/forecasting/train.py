"""
SmartShelf AI - Training utilities for forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseModel
from .prophet_model import ProphetDemandForecaster
from .lstm_model import LSTMDemandForecaster
from .evaluate import ModelEvaluator
from .features import FeatureEngineer

logger = logging.getLogger(__name__)


def train_forecasting_models(data: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Train multiple forecasting models and compare performance.
    
    Args:
        data: Training data with datetime index and target column
        config: Training configuration
        
    Returns:
        Dictionary with trained models and performance metrics
    """
    logger.info("Training forecasting models...")
    
    if config is None:
        config = {
            'target_column': 'demand',
            'date_column': 'date',
            'models': ['prophet', 'lstm'],
            'test_size': 0.2,
            'cv_folds': 5,
            'save_models': True
        }
    
    # Validate input data
    if data.empty:
        raise ValueError("Training data is empty")
    
    target_col = config['target_column']
    date_col = config.get('date_column')
    
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Prepare data
    data = data.copy()
    if date_col and date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.set_index(date_col)
    
    # Initialize results
    results = {
        'models_trained': [],
        'performance_metrics': {},
        'training_time': None,
        'best_model': None,
        'model_paths': {}
    }
    
    start_time = datetime.utcnow()
    
    # Split data
    test_size = config.get('test_size', 0.2)
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    logger.info(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")
    
    # Feature engineering
    feature_engineer = FeatureEngineer(config.get('feature_config', {}))
    
    # Train each model
    for model_name in config['models']:
        logger.info(f"Training {model_name} model...")
        
        try:
            if model_name == 'prophet':
                model = _train_prophet_model(train_data, test_data, config)
            elif model_name == 'lstm':
                model = _train_lstm_model(train_data, test_data, config, feature_engineer)
            else:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            # Evaluate model
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(model, test_data, config)
            
            results['models_trained'].append(model_name)
            results['performance_metrics'][model_name] = metrics
            
            # Save model if requested
            if config.get('save_models', True):
                model_path = model.save_model()
                results['model_paths'][model_name] = model_path
            
            logger.info(f"{model_name} training completed. Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Determine best model
    if results['performance_metrics']:
        best_model = min(results['performance_metrics'].keys(), 
                        key=lambda x: results['performance_metrics'][x].get('rmse', float('inf')))
        results['best_model'] = best_model
        logger.info(f"Best model: {best_model}")
    
    results['training_time'] = (datetime.utcnow() - start_time).total_seconds()
    
    logger.info(f"Training pipeline completed in {results['training_time']:.2f} seconds")
    return results


def _train_prophet_model(train_data: pd.DataFrame, test_data: pd.DataFrame, config: Dict[str, Any]) -> ProphetDemandForecaster:
    """Train Prophet forecasting model."""
    prophet_config = config.get('prophet_config', {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'seasonality_mode': 'multiplicative'
    })
    
    model = ProphetDemandForecaster(prophet_config)
    
    # Prepare data for Prophet
    prophet_train = train_data.reset_index()
    if prophet_train.columns[0].lower() in ['date', 'datetime', 'timestamp']:
        prophet_train = prophet_train.rename(columns={prophet_train.columns[0]: 'ds'})
    
    if config['target_column'] != 'y':
        prophet_train = prophet_train.rename(columns={config['target_column']: 'y'})
    
    # Train model
    X = prophet_train[['ds']]
    y = prophet_train['y']
    model.train(X, y)
    
    return model


def _train_lstm_model(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                     config: Dict[str, Any], feature_engineer: FeatureEngineer) -> LSTMDemandForecaster:
    """Train LSTM forecasting model."""
    lstm_config = config.get('lstm_config', {
        'sequence_length': 30,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2,
        'early_stopping_patience': 10
    })
    
    model = LSTMDemandForecaster(lstm_config)
    
    # Prepare data
    target_col = config['target_column']
    
    # Use only target column for basic LSTM, or add features if available
    if len(train_data.columns) > 1:
        # Multivariate case
        feature_cols = [col for col in train_data.columns if col != target_col]
        X_train = train_data[feature_cols + [target_col]]
    else:
        # Univariate case
        X_train = train_data[[target_col]]
    
    # Prepare data and train
    X, y = model.prepare_data(X_train)
    metrics = model.train(X, y)
    
    return model


def cross_validate_model(model: BaseModel, data: pd.DataFrame, cv_folds: int = 5) -> Dict[str, List[float]]:
    """
    Perform cross-validation on a forecasting model.
    
    Args:
        model: The model to validate
        data: Training data
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Performing {cv_folds}-fold cross-validation...")
    
    try:
        # Use the model's built-in cross-validation method
        target_col = data.columns[-1]  # Assume last column is target
        
        if len(data.columns) == 1:
            # Univariate case
            X = pd.DataFrame(index=data.index)
            y = data.iloc[:, 0]
        else:
            # Multivariate case
            X = data.drop(columns=[target_col])
            y = data[target_col]
        
        cv_results = model.cross_validate(X, y, cv_folds)
        
        # Calculate summary statistics
        summary = {}
        for metric, values in cv_results.items():
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
        
        results = {
            'fold_metrics': cv_results,
            'summary': summary,
            'cv_folds': cv_folds
        }
        
        logger.info(f"Cross-validation completed. RMSE: {summary.get('rmse_mean', 'N/A'):.4f} ± {summary.get('rmse_std', 'N/A'):.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Cross-validation failed: {str(e)}")
        return {
            'fold_metrics': {},
            'summary': {},
            'error': str(e)
        }


def hyperparameter_tuning(data: pd.DataFrame, model_type: str = 'lstm', 
                         param_grid: Dict[str, List[Any]] = None) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for forecasting models.
    
    Args:
        data: Training data
        model_type: Type of model to tune ('prophet' or 'lstm')
        param_grid: Dictionary of parameters to search
        
    Returns:
        Dictionary with tuning results and best parameters
    """
    logger.info(f"Starting hyperparameter tuning for {model_type}...")
    
    if param_grid is None:
        param_grid = _get_default_param_grid(model_type)
    
    best_score = float('inf')
    best_params = None
    best_model = None
    results = []
    
    # Generate parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))
        
        try:
            # Train model with current parameters
            if model_type == 'prophet':
                model = ProphetDemandForecaster(params)
            elif model_type == 'lstm':
                model = LSTMDemandForecaster(params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Evaluate using cross-validation
            cv_results = cross_validate_model(model, data, cv_folds=3)
            
            if 'error' not in cv_results:
                score = cv_results['summary'].get('rmse_mean', float('inf'))
                
                result = {
                    'params': params,
                    'score': score,
                    'cv_results': cv_results
                }
                results.append(result)
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
                logger.info(f"Params: {params} - RMSE: {score:.4f}")
            
        except Exception as e:
            logger.error(f"Error with params {params}: {str(e)}")
            continue
    
    tuning_results = {
        'best_params': best_params,
        'best_score': best_score,
        'best_model': best_model,
        'all_results': results,
        'model_type': model_type
    }
    
    logger.info(f"Hyperparameter tuning completed. Best params: {best_params}, Best RMSE: {best_score:.4f}")
    return tuning_results


def _get_default_param_grid(model_type: str) -> Dict[str, List[Any]]:
    """Get default parameter grid for hyperparameter tuning."""
    if model_type == 'prophet':
        return {
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False],
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_prior_scale': [0.01, 0.05, 0.1],
            'seasonality_prior_scale': [0.01, 0.1, 1.0]
        }
    elif model_type == 'lstm':
        return {
            'sequence_length': [15, 30, 60],
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32, 64]
        }
    else:
        return {}


def ensemble_forecast(models: List[BaseModel], data: pd.DataFrame, 
                     weights: List[float] = None) -> np.ndarray:
    """
    Create ensemble forecast from multiple models.
    
    Args:
        models: List of trained models
        data: Data for prediction
        weights: Optional weights for each model (default: equal weights)
        
    Returns:
        Ensemble predictions
    """
    if not models:
        raise ValueError("No models provided for ensemble")
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    if len(models) != len(weights):
        raise ValueError("Number of models must match number of weights")
    
    logger.info(f"Creating ensemble forecast with {len(models)} models")
    
    predictions = []
    
    for i, model in enumerate(models):
        try:
            pred = model.predict(data)
            predictions.append(pred * weights[i])
            logger.info(f"Model {i+1} prediction shape: {pred.shape}")
        except Exception as e:
            logger.error(f"Error in model {i+1}: {str(e)}")
            continue
    
    if not predictions:
        raise ValueError("No valid predictions from any model")
    
    # Average predictions
    ensemble_pred = np.sum(predictions, axis=0)
    
    logger.info(f"Ensemble forecast completed. Shape: {ensemble_pred.shape}")
    return ensemble_pred


def forecast_pipeline(data: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Complete forecasting pipeline with training, evaluation, and prediction.
    
    Args:
        data: Historical data
        config: Pipeline configuration
        
    Returns:
        Complete pipeline results
    """
    logger.info("Starting complete forecasting pipeline...")
    
    if config is None:
        config = {
            'target_column': 'demand',
            'forecast_horizon': 30,
            'models': ['prophet', 'lstm'],
            'use_ensemble': True,
            'save_models': True
        }
    
    # Train models
    training_results = train_forecasting_models(data, config)
    
    # Generate forecasts
    forecasts = {}
    
    for model_name in training_results['models_trained']:
        try:
            # Load the trained model
            model_path = training_results['model_paths'][model_name]
            
            if model_name == 'prophet':
                model = ProphetDemandForecaster()
            elif model_name == 'lstm':
                model = LSTMDemandForecaster()
            else:
                continue
            
            model.load_model(model_path)
            
            # Make future predictions
            future_data = _create_future_data(data, config['forecast_horizon'], config)
            forecast = model.predict(future_data)
            forecasts[model_name] = forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast for {model_name}: {str(e)}")
            continue
    
    # Create ensemble if requested
    if config.get('use_ensemble', True) and len(forecasts) > 1:
        try:
            # Load models for ensemble
            models = []
            for model_name in forecasts.keys():
                if model_name == 'prophet':
                    model = ProphetDemandForecaster()
                elif model_name == 'lstm':
                    model = LSTMDemandForecaster()
                else:
                    continue
                
                model_path = training_results['model_paths'][model_name]
                model.load_model(model_path)
                models.append(model)
            
            future_data = _create_future_data(data, config['forecast_horizon'], config)
            ensemble_forecast = ensemble_forecast(models, future_data)
            forecasts['ensemble'] = ensemble_forecast
            
        except Exception as e:
            logger.error(f"Error creating ensemble forecast: {str(e)}")
    
    pipeline_results = {
        'training_results': training_results,
        'forecasts': forecasts,
        'config': config,
        'forecast_horizon': config['forecast_horizon']
    }
    
    logger.info("Forecasting pipeline completed successfully")
    return pipeline_results


def _create_future_data(historical_data: pd.DataFrame, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
    """Create future data for forecasting."""
    # This is a simplified implementation
    # In practice, you'd create proper future dates and features
    
    last_date = historical_data.index[-1] if hasattr(historical_data.index, 'date') else None
    
    if last_date:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        future_data = pd.DataFrame(index=future_dates)
        
        # Add basic features
        future_data['day_of_week'] = future_data.index.dayofweek
        future_data['month'] = future_data.index.month
        future_data['day_of_month'] = future_data.index.day
        
        # Fill with last known values for other features
        for col in historical_data.columns:
            if col not in future_data.columns:
                future_data[col] = historical_data[col].iloc[-1]
    else:
        # Simple case - just repeat the last row
        future_data = pd.concat([historical_data.iloc[-1:]] * horizon, ignore_index=True)
    
    return future_data
