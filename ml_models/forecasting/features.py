"""
SmartShelf AI - Feature Engineering for Demand Forecasting

Feature engineering utilities for time-series forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for demand forecasting.
    
    Creates time-based features, lag features, rolling statistics,
    and external regressors for ML models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize feature engineer with configuration."""
        self.config = config or {}
        
        # Feature configuration
        self.include_lag_features = self.config.get('include_lag_features', True)
        self.include_rolling_features = self.config.get('include_rolling_features', True)
        self.include_time_features = self.config.get('include_time_features', True)
        self.include_promotion_features = self.config.get('include_promotion_features', True)
        
        # Lag configuration
        self.lag_periods = self.config.get('lag_periods', [1, 7, 14, 30])
        
        # Rolling window configuration
        self.rolling_windows = self.config.get('rolling_windows', [7, 14, 30])
        
        logger.info("Initialized FeatureEngineer")
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create time-based features from date column.
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with time features added
        """
        if not self.include_time_features:
            return df
        
        df = df.copy()
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic time features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['dayofyear'] = df[date_col].dt.dayofyear
        df['week'] = df[date_col].dt.isocalendar().week
        df['quarter'] = df[date_col].dt.quarter
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
        
        # Weekend indicator
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Month start/end indicators
        df['is_month_start'] = df['day'] == 1
        df['is_month_end'] = df['day'] == df[date_col].dt.days_in_month
        
        # Season indicators (Northern Hemisphere)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        logger.info(f"Created {len([col for col in df.columns if col not in df.columns[:len(df.columns)]])} time features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, date_col: str = 'date') -> pd.DataFrame:
        """
        Create lag features for time series.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            date_col: Name of date column
            
        Returns:
            DataFrame with lag features added
        """
        if not self.include_lag_features:
            return df
        
        df = df.copy()
        df = df.sort_values(date_col)
        
        for lag in self.lag_periods:
            lag_col = f'{target_col}_lag_{lag}'
            df[lag_col] = df[target_col].shift(lag)
        
        logger.info(f"Created {len(self.lag_periods)} lag features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str, date_col: str = 'date') -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            date_col: Name of date column
            
        Returns:
            DataFrame with rolling features added
        """
        if not self.include_rolling_features:
            return df
        
        df = df.copy()
        df = df.sort_values(date_col)
        
        for window in self.rolling_windows:
            # Rolling statistics
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window=window).median()
            
            # Rolling trends
            df[f'{target_col}_rolling_trend_{window}'] = df[target_col].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan, raw=False
            )
        
        logger.info(f"Created {len(self.rolling_windows) * 6} rolling features")
        
        return df
    
    def create_promotion_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create promotion-related features.
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with promotion features added
        """
        if not self.include_promotion_features or 'promotion_active' not in df.columns:
            return df
        
        df = df.copy()
        df = df.sort_values(date_col)
        
        # Promotion indicators
        df['promotion_active'] = df['promotion_active'].astype(int)
        
        # Rolling promotion features
        for window in [7, 14, 30]:
            df[f'promotion_count_rolling_{window}'] = df['promotion_active'].rolling(window=window).sum()
            df[f'promotion_rate_rolling_{window}'] = df['promotion_active'].rolling(window=window).mean()
        
        # Days since last promotion
        df['days_since_last_promo'] = df.groupby((df['promotion_active'] != df['promotion_active'].shift()).cumsum()).cumcount()
        df['days_since_last_promo'] = df['days_since_last_promo'] * df['promotion_active'].shift(1).fillna(0)
        
        # Days until next promotion
        df['days_until_next_promo'] = df[::-1].groupby((df['promotion_active'] != df['promotion_active'].shift(-1)).cumsum()).cumcount()[::-1]
        df['days_until_next_promo'] = df['days_until_next_promo'] * df['promotion_active'].shift(-1).fillna(0)
        
        logger.info("Created promotion features")
        
        return df
    
    def create_all_features(self, df: pd.DataFrame, target_col: str, date_col: str = 'date') -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            date_col: Name of date column
            
        Returns:
            DataFrame with all features added
        """
        logger.info("Creating all features...")
        
        # Create different feature types
        df = self.create_time_features(df, date_col)
        df = self.create_lag_features(df, target_col, date_col)
        df = self.create_rolling_features(df, target_col, date_col)
        df = self.create_promotion_features(df, date_col)
        
        # Remove rows with NaN values created by lag/rolling features
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"Feature engineering complete. Removed {initial_rows - final_rows} rows with NaN values")
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for importance analysis.
        
        Returns:
            Dictionary mapping feature groups to feature names
        """
        feature_groups = {
            'time_features': [],
            'lag_features': [],
            'rolling_features': [],
            'promotion_features': [],
            'cyclical_features': []
        }
        
        # Time features
        time_features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter', 
                        'is_weekend', 'is_month_start', 'is_month_end', 
                        'is_spring', 'is_summer', 'is_fall', 'is_winter']
        feature_groups['time_features'].extend(time_features)
        
        # Cyclical features
        cyclical_features = ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 
                           'dayofyear_sin', 'dayofyear_cos']
        feature_groups['cyclical_features'].extend(cyclical_features)
        
        # Lag features
        for lag in self.lag_periods:
            feature_groups['lag_features'].append(f'quantity_sold_lag_{lag}')
        
        # Rolling features
        for window in self.rolling_windows:
            rolling_features = [f'quantity_sold_rolling_mean_{window}',
                              f'quantity_sold_rolling_std_{window}',
                              f'quantity_sold_rolling_min_{window}',
                              f'quantity_sold_rolling_max_{window}',
                              f'quantity_sold_rolling_median_{window}',
                              f'quantity_sold_rolling_trend_{window}']
            feature_groups['rolling_features'].extend(rolling_features)
        
        # Promotion features
        promotion_features = ['promotion_active', 'days_since_last_promo', 'days_until_next_promo']
        for window in [7, 14, 30]:
            promotion_features.extend([f'promotion_count_rolling_{window}', 
                                     f'promotion_rate_rolling_{window}'])
        feature_groups['promotion_features'].extend(promotion_features)
        
        return feature_groups
