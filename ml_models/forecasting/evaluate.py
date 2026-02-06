"""
SmartShelf AI - Evaluation utilities for forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def evaluate_forecasts(predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """
    Evaluate forecast accuracy with multiple metrics.
    
    Args:
        predictions: Forecast predictions
        actual: Actual values
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating forecast accuracy...")
    
    # Calculate metrics
    mse = np.mean((actual - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predictions))
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predictions) / np.maximum(actual, 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2)
    }
    
    logger.info(f"Evaluation metrics: MAPE={mape:.2f}%, RMSE={rmse:.2f}")
    
    return metrics
