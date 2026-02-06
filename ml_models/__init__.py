"""
SmartShelf AI - Machine Learning Pipeline

Comprehensive ML system for retail analytics including:
- Demand forecasting with Prophet and LSTM
- Pricing optimization with elasticity analysis
- Inventory intelligence with stockout prediction
- Model orchestration and monitoring
"""

from .base import BaseModel
from .pipeline import MLPipeline
from .utils import *

__version__ = "1.0.0"
__author__ = "SmartShelf AI Team"

# Import all model modules
from .forecasting import *
from .pricing import *
from .inventory import *

__all__ = [
    'BaseModel',
    'MLPipeline',
    'DemandForecaster',
    'PricingOptimizer', 
    'InventoryAnalyzer'
]
