"""
SmartShelf AI - LSTM Demand Forecasting Model

LSTM-based neural network for demand forecasting.
Provides deep learning alternative to Prophet.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base import BaseModel

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM neural network architecture for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out.squeeze()


class LSTMDemandForecaster(BaseModel):
    """
    LSTM-based demand forecasting model.
    
    Features:
    - Deep learning approach for complex patterns
    - Sequence modeling for time series
    - Customizable architecture
    - GPU acceleration support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LSTM forecaster."""
        default_config = {
            'sequence_length': 30,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("lstm_demand_forecaster", default_config)
        
        # LSTM-specific attributes
        self.sequence_length = self.config['sequence_length']
        self.device = torch.device(self.config['device'])
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        
        # Initialize model
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        
        logger.info(f"Initialized LSTM Demand Forecaster with device: {self.device}")
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for LSTM training."""
        # Validate input
        self.validate_input_data(data)
        
        # Handle missing values
        data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        
        # Assume last column is target, rest are features
        if len(data.columns) == 1:
            # Univariate case - create lag features
            target_col = data.columns[0]
            data = self._create_lag_features(data[target_col])
            target = data[target_col].iloc[self.sequence_length:]
            features = data.drop(columns=[target_col]).iloc[self.sequence_length:]
        else:
            # Multivariate case
            target_col = data.columns[-1]
            target = data[target_col]
            features = data.drop(columns=[target_col])
        
        # Store feature info
        self.feature_columns = features.columns.tolist()
        self.target_column = target_col
        
        logger.info(f"Prepared data: {features.shape} features, {target.shape} target")
        return features, target
    
    def _create_lag_features(self, series: pd.Series, max_lags: int = None) -> pd.DataFrame:
        """Create lag features for univariate time series."""
        if max_lags is None:
            max_lags = self.sequence_length
        
        df = pd.DataFrame({'target': series})
        
        # Create lag features
        for i in range(1, max_lags + 1):
            df[f'lag_{i}'] = series.shift(i)
        
        # Create rolling statistics
        df['rolling_mean_7'] = series.rolling(window=7, min_periods=1).mean()
        df['rolling_std_7'] = series.rolling(window=7, min_periods=1).std()
        df['rolling_mean_30'] = series.rolling(window=30, min_periods=1).mean()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train LSTM model."""
        logger.info("Starting LSTM training...")
        
        # Store training data shape
        self.training_data_shape = (X.shape[0], X.shape[1] + 1)
        
        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Split data
        split_idx = int(len(X_seq) * (1 - self.config['validation_split']))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model
        input_size = X_train.shape[2]  # Number of features
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            # Create batches
            for i in range(0, len(X_train), self.config['batch_size']):
                batch_X = X_train[i:i + self.config['batch_size']]
                batch_y = y_train[i:i + self.config['batch_size']]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / (len(X_train) // self.config['batch_size'])
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(X_val), self.config['batch_size']):
                    batch_X = X_val[i:i + self.config['batch_size']]
                    batch_y = y_val[i:i + self.config['batch_size']]
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / (len(X_val) // self.config['batch_size'])
            val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.models_dir / f"{self.model_name}_best.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Load best model
        best_model_path = self.models_dir / f"{self.model_name}_best.pth"
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train).cpu().numpy()
            val_pred = self.model(X_val).cpu().numpy()
            
            # Inverse transform predictions
            train_pred = self.scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            val_pred = self.scaler_y.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            y_train_orig = self.scaler_y.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
            y_val_orig = self.scaler_y.inverse_transform(y_val.cpu().numpy().reshape(-1, 1)).flatten()
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train_orig, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred))
            val_mae = mean_absolute_error(y_val_orig, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred))
        
        metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'epochs_trained': epoch + 1
        }
        
        self.training_metrics = metrics
        self.is_trained = True
        self.last_trained = datetime.utcnow()
        
        logger.info(f"LSTM training completed. Final metrics: {metrics}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Scale input
        X_scaled = self.scaler_X.transform(X)
        
        # For prediction, we need the last sequence_length points
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:i + self.sequence_length])
        
        if not X_seq:
            # If we don't have enough for full sequences, pad with the last available values
            last_sequence = X_scaled[-self.sequence_length:].copy()
            X_seq = [last_sequence]
        
        X_tensor = torch.FloatTensor(np.array(X_seq)).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform predictions
        predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained LSTM model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_path = Path(path) if path else self.model_path
        
        # Save model state dict
        model_save_path = save_path.with_suffix('.pth')
        torch.save(self.model.state_dict(), model_save_path)
        
        # Save scalers
        joblib.dump(self.scaler_X, save_path.with_suffix('_scaler_X.pkl'))
        joblib.dump(self.scaler_y, save_path.with_suffix('_scaler_y.pkl'))
        
        # Save metadata using parent method
        super().save_model(str(save_path))
        
        logger.info(f"LSTM model saved to {model_save_path}")
        return str(model_save_path)
    
    def load_model(self, path: Optional[str] = None) -> None:
        """Load a trained LSTM model from disk."""
        load_path = Path(path) if path else self.model_path
        
        model_load_path = load_path.with_suffix('.pth')
        if not model_load_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_load_path}")
        
        # Load scalers
        scaler_X_path = load_path.with_suffix('_scaler_X.pkl')
        scaler_y_path = load_path.with_suffix('_scaler_y.pkl')
        
        if scaler_X_path.exists():
            self.scaler_X = joblib.load(scaler_X_path)
        if scaler_y_path.exists():
            self.scaler_y = joblib.load(scaler_y_path)
        
        # Load metadata first to get config
        metadata_path = load_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.config = metadata.get('config', self.config)
            self.feature_columns = metadata.get('feature_columns', [])
            self.target_column = metadata.get('target_column')
        
        # Initialize and load model
        input_size = len(self.feature_columns) if self.feature_columns else 1
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_load_path, map_location=self.device))
        self.is_trained = True
        
        # Load remaining metadata
        if metadata_path.exists():
            self.model_name = metadata.get('model_name', self.model_name)
            self.model_version = metadata.get('model_version', '1.0.0')
            self.training_metrics = metadata.get('training_metrics', {})
            self.validation_metrics = metadata.get('validation_metrics', {})
            self.feature_importance = metadata.get('feature_importance', {})
            self.created_at = datetime.fromisoformat(metadata.get('created_at', '2024-01-01'))
            self.last_trained = datetime.fromisoformat(metadata['last_trained']) if metadata.get('last_trained') else None
            self.training_data_shape = metadata.get('training_data_shape')
        
        logger.info(f"LSTM model loaded from {model_load_path}")
