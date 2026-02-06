"""
SmartShelf AI - Demand Forecasting Module

Advanced time-series forecasting for retail demand prediction.
Supports multiple algorithms including Prophet and LSTM.
"""

from .prophet_model import DemandForecaster
from .features import FeatureEngineer
from .train import train_forecasting_models
from .evaluate import evaluate_forecasts

# Optional LSTM import
try:
    from .lstm_model import LSTMDemandForecaster
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTM_AVAILABLE = False
    LSTMDemandForecaster = None

__all__ = [
    'DemandForecaster',
    'FeatureEngineer',
    'train_forecasting_models',
    'evaluate_forecasts'
]

if _LSTM_AVAILABLE:
    __all__.append('LSTMDemandForecaster')
