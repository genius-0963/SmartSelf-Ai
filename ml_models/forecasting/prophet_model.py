"""
SmartShelf AI - Prophet Demand Forecasting Model

Facebook Prophet implementation for retail demand forecasting.
Handles seasonality, trends, holidays, and promotional effects.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings

# Suppress Prophet warnings
warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseModel


class DemandForecaster(BaseModel):
    """
    Prophet-based demand forecasting model for retail products.
    
    Features:
    - Automatic seasonality detection (weekly, monthly, yearly)
    - Holiday and promotional effects
    - Trend changepoint detection
    - Confidence intervals
    - Cross-validation and backtesting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Prophet demand forecaster.
        
        Args:
            config: Configuration dictionary with Prophet parameters
        """
        default_config = {
            # Prophet core parameters
            'yearly_seasonality': 'auto',
            'weekly_seasonality': 'auto',
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative',
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'changepoint_prior_scale': 0.05,
            'mcmc_samples': 0,
            'interval_width': 0.95,
            'uncertainty_samples': 1000,
            
            # Custom parameters
            'growth': 'linear',
            'cap': None,  # For logistic growth
            'floor': None,
            'n_changepoints': 25,
            'changepoint_range': 0.8,
            
            # SmartShelf specific
            'include_promotions': True,
            'include_holidays': True,
            'promotion_effect_prior_scale': 5.0,
            'min_historical_days': 30,
            'forecast_horizon_days': 30
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("prophet_demand_forecaster", default_config)
        
        # Prophet-specific attributes
        self.prophet_model = None
        self.holidays_df = None
        self.promo_events = None
        self.feature_engineer = None
        
        # Training data info
        self.training_start_date = None
        self.training_end_date = None
        self.product_ids = []
        
        logger.info("Initialized Prophet Demand Forecaster")
    
    def prepare_holidays(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Prepare holidays dataframe for Prophet.
        
        Args:
            start_date: Start date for holiday data
            end_date: End date for holiday data
            
        Returns:
            DataFrame with holidays for Prophet
        """
        holidays = []
        
        # Major US retail holidays (can be extended for other regions)
        holiday_dates = {
            'New Year': [(1, 1)],
            'Valentine': [(2, 14)],
            'Presidents Day': [(2, 20)],  # Third Monday in February (simplified)
            'St. Patrick': [(3, 17)],
            'Easter': [(4, 9)],  # Simplified (changes annually)
            'Mothers Day': [(5, 14)],  # Second Sunday in May (simplified)
            'Memorial Day': [(5, 29)],  # Last Monday in May (simplified)
            'Fathers Day': [(6, 18)],  # Third Sunday in June (simplified)
            'Independence Day': [(7, 4)],
            'Labor Day': [(9, 4)],  # First Monday in September (simplified)
            'Halloween': [(10, 31)],
            'Veterans Day': [(11, 11)],
            'Thanksgiving': [(11, 23)],  # Fourth Thursday in November (simplified)
            'Black Friday': [(11, 24)],
            'Cyber Monday': [(11, 27)],
            'Christmas': [(12, 25)],
            'Christmas Eve': [(12, 24)],
            'New Year Eve': [(12, 31)]
        }
        
        current_year = start_date.year
        end_year = end_date.year
        
        for year in range(current_year, end_year + 1):
            for holiday_name, dates in holiday_dates.items():
                for month, day in dates:
                    holiday_date = datetime(year, month, day)
                    
                    # Extend some holidays to multi-day events
                    if holiday_name in ['Black Friday', 'Cyber Monday', 'Christmas', 'Thanksgiving']:
                        # Create 2-3 day holiday windows
                        for offset in range(-1, 2):  # -1, 0, 1
                            extended_date = holiday_date + timedelta(days=offset)
                            if start_date <= extended_date <= end_date:
                                holidays.append({
                                    'holiday': holiday_name,
                                    'ds': extended_date,
                                    'lower_window': 0,
                                    'upper_window': 1
                                })
                    else:
                        if start_date <= holiday_date <= end_date:
                            holidays.append({
                                'holiday': holiday_name,
                                'ds': holiday_date,
                                'lower_window': 0,
                                'upper_window': 1
                            })
        
        holidays_df = pd.DataFrame(holidays)
        logger.info(f"Prepared {len(holidays_df)} holiday events")
        
        return holidays_df
    
    def prepare_promotions(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare promotional events from sales data.
        
        Args:
            sales_data: Sales data with promotion information
            
        Returns:
            DataFrame with promotional events
        """
        if not self.config.get('include_promotions', True):
            return pd.DataFrame()
        
        # Find dates with significant promotional activity
        promo_data = sales_data[sales_data['promotion_active'] == True].copy()
        
        if promo_data.empty:
            logger.info("No promotional data found")
            return pd.DataFrame()
        
        # Aggregate promotions by date
        daily_promos = promo_data.groupby('date').agg({
            'total_revenue': 'sum',
            'quantity_sold': 'sum',
            'promotion_discount': 'mean'
        }).reset_index()
        
        # Create promotion events
        promo_events = []
        for _, row in daily_promos.iterrows():
            promo_events.append({
                'holiday': f'promotion_{row["date"].strftime("%Y%m%d")}',
                'ds': row['date'],
                'lower_window': 0,
                'upper_window': 1
            })
        
        promo_df = pd.DataFrame(promo_events)
        logger.info(f"Prepared {len(promo_df)} promotional events")
        
        return promo_df
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for Prophet training.
        
        Args:
            data: Sales data with date and quantity columns
            
        Returns:
            Tuple of (features DataFrame, target Series) in Prophet format
        """
        self.validate_input_data(data)
        
        # Ensure required columns
        required_cols = ['date', 'quantity_sold']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Aggregate data by date (Prophet expects one row per date)
        if 'product_id' in data.columns:
            # Multi-product forecasting
            prophet_data = data.groupby(['product_id', 'date']).agg({
                'quantity_sold': 'sum',
                'total_revenue': 'sum',
                'promotion_active': 'any',
                'promotion_discount': 'mean'
            }).reset_index()
        else:
            # Single product forecasting
            prophet_data = data.groupby('date').agg({
                'quantity_sold': 'sum',
                'total_revenue': 'sum',
                'promotion_active': 'any',
                'promotion_discount': 'mean'
            }).reset_index()
        
        # Rename columns for Prophet
        prophet_data = prophet_data.rename(columns={
            'date': 'ds',
            'quantity_sold': 'y'
        })
        
        # Sort by date
        prophet_data = prophet_data.sort_values('ds')
        
        # Store training data info
        self.training_start_date = prophet_data['ds'].min()
        self.training_end_date = prophet_data['ds'].max()
        self.training_data_shape = prophet_data.shape
        
        if 'product_id' in prophet_data.columns:
            self.product_ids = prophet_data['product_id'].unique().tolist()
        
        # Prophet expects just the ds and y columns for training
        features = prophet_data[['ds', 'y']].copy()
        target = prophet_data['y']
        
        logger.info(f"Prepared {len(features)} training days from {self.training_start_date} to {self.training_end_date}")
        
        return features, target
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the Prophet model.
        
        Args:
            X: Features DataFrame (must include 'ds' column)
            y: Target Series (demand values)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting Prophet model training...")
        
        # Prepare training data
        train_data = X.copy()
        train_data['y'] = y
        
        # Initialize Prophet model with configuration
        prophet_params = {
            'yearly_seasonality': self.config['yearly_seasonality'],
            'weekly_seasonality': self.config['weekly_seasonality'],
            'daily_seasonality': self.config['daily_seasonality'],
            'seasonality_mode': self.config['seasonality_mode'],
            'seasonality_prior_scale': self.config['seasonality_prior_scale'],
            'holidays_prior_scale': self.config['holidays_prior_scale'],
            'changepoint_prior_scale': self.config['changepoint_prior_scale'],
            'interval_width': self.config['interval_width'],
            'uncertainty_samples': self.config['uncertainty_samples'],
            'growth': self.config['growth'],
            'n_changepoints': self.config['n_changepoints'],
            'changepoint_range': self.config['changepoint_range']
        }
        
        # Add capacity constraints if logistic growth
        if self.config['growth'] == 'logistic':
            if self.config['cap']:
                train_data['cap'] = self.config['cap']
            else:
                train_data['cap'] = train_data['y'].max() * 1.5
            
            if self.config['floor']:
                train_data['floor'] = self.config['floor']
            else:
                train_data['floor'] = 0
        
        self.prophet_model = Prophet(**prophet_params)
        
        # Add holidays if enabled
        if self.config.get('include_holidays', True):
            if self.training_start_date and self.training_end_date:
                self.holidays_df = self.prepare_holidays(self.training_start_date, self.training_end_date)
                if not self.holidays_df.empty:
                    self.prophet_model.add_holidays(self.holidays_df)
        
        # Add custom seasonalities if needed
        if len(train_data) >= 365:  # Only add yearly if we have enough data
            self.prophet_model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=8,
                prior_scale=self.config['seasonality_prior_scale']
            )
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prophet_model.fit(train_data)
        
        self.is_trained = True
        self.last_trained = datetime.utcnow()
        
        # Calculate training metrics
        training_metrics = self._calculate_training_metrics(train_data)
        self.training_metrics = training_metrics
        
        # Store feature importance (Prophet components)
        self.feature_importance = self._get_feature_importance()
        
        logger.info(f"Prophet model training completed. Metrics: {training_metrics}")
        
        return training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained Prophet model.
        
        Args:
            X: DataFrame with 'ds' column (dates to predict)
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare prediction data
        future_data = X.copy()
        
        # Add capacity constraints if logistic growth
        if self.config['growth'] == 'logistic':
            if self.config['cap']:
                future_data['cap'] = self.config['cap']
            else:
                future_data['cap'] = self.training_metrics.get('max_observed_value', 100) * 1.5
            
            if self.config['floor']:
                future_data['floor'] = self.config['floor']
            else:
                future_data['floor'] = 0
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.prophet_model.predict(future_data)
        
        return forecast['yhat'].values
    
    def predict_with_confidence(self, X: pd.DataFrame, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: DataFrame with 'ds' column
            confidence_level: Confidence interval level
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare prediction data
        future_data = X.copy()
        
        # Add capacity constraints if logistic growth
        if self.config['growth'] == 'logistic':
            if self.config['cap']:
                future_data['cap'] = self.config['cap']
            else:
                future_data['cap'] = self.training_metrics.get('max_observed_value', 100) * 1.5
            
            if self.config['floor']:
                future_data['floor'] = self.config['floor']
            else:
                future_data['floor'] = 0
        
        # Make predictions with confidence intervals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.prophet_model.predict(future_data)
        
        predictions = forecast['yhat'].values
        
        # Adjust confidence intervals if requested
        if confidence_level != self.config['interval_width']:
            # Prophet uses fixed interval width, so we'll approximate
            scaling_factor = confidence_level / self.config['interval_width']
            lower_bound = forecast['yhat_lower'].values
            upper_bound = forecast['yhat_upper'].values
            
            # Scale the intervals (approximation)
            margin_lower = predictions - lower_bound
            margin_upper = upper_bound - predictions
            
            lower_bound = predictions - margin_lower * scaling_factor
            upper_bound = predictions + margin_upper * scaling_factor
        else:
            lower_bound = forecast['yhat_lower'].values
            upper_bound = forecast['yhat_upper'].values
        
        return predictions, lower_bound, upper_bound
    
    def forecast_future(self, periods: int, freq: str = 'D') -> pd.DataFrame:
        """
        Generate future forecast for specified periods.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with forecast including dates, predictions, and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
        
        # Add capacity constraints if logistic growth
        if self.config['growth'] == 'logistic':
            if self.config['cap']:
                future['cap'] = self.config['cap']
            else:
                future['cap'] = self.training_metrics.get('max_observed_value', 100) * 1.5
            
            if self.config['floor']:
                future['floor'] = self.config['floor']
            else:
                future['floor'] = 0
        
        # Make forecast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.prophet_model.predict(future)
        
        # Select relevant columns
        result_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        if 'trend' in forecast.columns:
            result_columns.extend(['trend', 'seasonal', 'yearly', 'weekly'])
        
        return forecast[result_columns]
    
    def get_forecast_components(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get forecast components (trend, seasonalities, holidays).
        
        Args:
            forecast_df: Forecast dataframe from Prophet
            
        Returns:
            DataFrame with forecast components
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting components")
        
        # Get components from the forecast
        component_columns = ['ds', 'trend']
        
        # Add seasonal components if available
        for col in ['seasonal', 'yearly', 'weekly', 'monthly', 'holidays']:
            if col in forecast_df.columns:
                component_columns.append(col)
        
        return forecast_df[component_columns]
    
    def cross_validate_model(self, initial: str, period: str, horizon: str) -> pd.DataFrame:
        """
        Perform time-series cross-validation.
        
        Args:
            initial: Initial training period (e.g., '180 days')
            period: Spacing between cutoff dates (e.g., '30 days')
            horizon: Forecast horizon (e.g., '30 days')
            
        Returns:
            DataFrame with cross-validation performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")
        
        logger.info(f"Performing cross-validation: initial={initial}, period={period}, horizon={horizon}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_validation(
                self.prophet_model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel='threads'
            )
        
        # Calculate performance metrics
        performance = performance_metrics(cv_results)
        
        logger.info(f"Cross-validation completed. MAPE: {performance['mape'].mean():.2f}%")
        
        return performance
    
    def _calculate_training_metrics(self, train_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate training-specific metrics."""
        metrics = {
            'training_days': len(train_data),
            'min_observed_value': float(train_data['y'].min()),
            'max_observed_value': float(train_data['y'].max()),
            'mean_observed_value': float(train_data['y'].mean()),
            'std_observed_value': float(train_data['y'].std()),
            'total_observed_demand': float(train_data['y'].sum())
        }
        
        # Calculate in-sample predictions for training metrics
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self.prophet_model.predict(train_data)
            
            y_true = train_data['y'].values
            y_pred = forecast['yhat'].values
            
            # Calculate metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
            
            metrics.update({
                'train_mse': float(mse),
                'train_rmse': float(rmse),
                'train_mae': float(mae),
                'train_mape': float(mape)
            })
            
        except Exception as e:
            logger.warning(f"Could not calculate in-sample metrics: {e}")
        
        return metrics
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from Prophet model."""
        importance = {}
        
        if self.prophet_model:
            # Get component contributions (approximate feature importance)
            try:
                # Use seasonal parameters as proxy for importance
                if hasattr(self.prophet_model, 'params'):
                    params = self.prophet_model.params
                    
                    # Trend importance
                    if 'k' in params:  # Growth rate
                        importance['trend'] = abs(float(params['k'][0]))
                    
                    # Seasonality importance
                    if 'seasonal' in params:
                        importance['seasonality'] = float(np.mean(np.abs(params['seasonal'])))
                    
                    # Holiday importance
                    if 'holidays' in params and self.holidays_df is not None:
                        importance['holidays'] = float(np.mean(np.abs(params['holidays'])))
                
                # Normalize importance scores
                if importance:
                    total = sum(importance.values())
                    if total > 0:
                        importance = {k: v/total for k, v in importance.items()}
                
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {e}")
        
        return importance
    
    def plot_forecast(self, forecast_df: pd.DataFrame, include_components: bool = True):
        """
        Plot forecast using Prophet's built-in plotting.
        
        Args:
            forecast_df: Forecast dataframe
            include_components: Whether to plot components
        """
        try:
            import matplotlib.pyplot as plt
            
            # Plot forecast
            fig1 = self.prophet_model.plot(forecast_df)
            plt.title(f'Demand Forecast - {self.model_name}')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.show()
            
            # Plot components if requested
            if include_components:
                fig2 = self.prophet_model.plot_components(forecast_df)
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = self.get_model_info()
        
        if self.is_trained:
            summary.update({
                'training_start_date': self.training_start_date.isoformat() if self.training_start_date else None,
                'training_end_date': self.training_end_date.isoformat() if self.training_end_date else None,
                'product_ids': self.product_ids,
                'holidays_included': len(self.holidays_df) if self.holidays_df is not None else 0,
                'prophet_config': {
                    'seasonality_mode': self.config['seasonality_mode'],
                    'yearly_seasonality': self.config['yearly_seasonality'],
                    'weekly_seasonality': self.config['weekly_seasonality'],
                    'interval_width': self.config['interval_width']
                }
            })
        
        return summary
