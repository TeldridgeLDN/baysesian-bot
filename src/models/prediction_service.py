"""
Prediction service for real-time cryptocurrency price forecasting.
Provides production-ready interface for the Bayesian LSTM model with caching and error handling.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import threading
import time
from pathlib import Path

from .bayesian_lstm import BayesianLSTM, ModelManager
from ..data.storage import DatabaseManager, PredictionData, ModelMetrics
from ..utils.config import ModelConfig

logger = logging.getLogger(__name__)

class ModelCache:
    """Thread-safe cache for model predictions."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items() 
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached prediction if valid."""
        with self.lock:
            self._cleanup_expired()
            if key in self.cache:
                return self.cache[key].copy()
        return None
    
    def put(self, key: str, value: Dict):
        """Cache a prediction result."""
        with self.lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                self.cache.pop(oldest_key, None)
                self.timestamps.pop(oldest_key, None)
            
            self.cache[key] = value.copy()
            self.timestamps[key] = time.time()

class PredictionService:
    """
    Production-ready prediction service for cryptocurrency trading.
    
    Features:
    - Real-time Bayesian LSTM predictions with uncertainty quantification
    - Automatic model loading and validation
    - Prediction caching for performance
    - Model degradation detection and alerts
    - Integration with database for storing predictions and metrics
    """
    
    def __init__(self, 
                 model_config: ModelConfig,
                 db_manager: Optional[DatabaseManager] = None,
                 model_path: Optional[str] = None,
                 enable_caching: bool = True):
        """
        Initialize prediction service.
        
        Args:
            model_config: Model configuration
            db_manager: Database manager for storing predictions
            model_path: Path to pre-trained model (optional)
            enable_caching: Enable prediction caching
        """
        self.config = model_config
        self.db_manager = db_manager
        self.model_path = model_path
        self.enable_caching = enable_caching
        
        # Initialize components
        self.model_manager = ModelManager()
        self.model = None
        self.cache = ModelCache() if enable_caching else None
        self.data_buffer = deque(maxlen=self.config.sequence_length * 2)  # Buffer for sequence data
        self.prediction_stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0,
            'last_prediction_time': None,
            'model_version': None
        }
        
        # Feature engineering setup (simplified for demonstration)
        self.feature_columns = [
            'price', 'volume', 'high', 'low', 'open',
            'price_ma_5', 'price_ma_20', 'volume_ma_5',
            'rsi', 'macd', 'bollinger_upper', 'bollinger_lower'
        ]
        
        self.is_initialized = False
        logger.info("PredictionService initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the prediction service by loading or creating a model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to load existing model
            if self.model_path and Path(self.model_path + ".keras").exists():
                logger.info(f"Loading existing model from {self.model_path}")
                self.model = self.model_manager.load_model(self.model_path)
                self.prediction_stats['model_version'] = self.model.model_version
                logger.info(f"Loaded model version: {self.model.model_version}")
            else:
                logger.info("Creating new model - no existing model found")
                # Create new model with config
                model_config_dict = {
                    'sequence_length': self.config.sequence_length,
                    'feature_count': len(self.feature_columns),
                    'lstm_units': self.config.lstm_units,
                    'dropout_rate': self.config.dropout_rate,
                    'recurrent_dropout_rate': self.config.recurrent_dropout_rate,
                    'dense_dropout_rate': self.config.dense_dropout_rate,
                    'learning_rate': self.config.learning_rate,
                    'monte_carlo_samples': self.config.monte_carlo_samples,
                    'confidence_interval': self.config.confidence_interval
                }
                
                self.model = self.model_manager.create_model(model_config_dict)
                self.prediction_stats['model_version'] = self.model.model_version
                logger.warning("Created new untrained model - predictions will be random until training")
            
            self.is_initialized = True
            logger.info("PredictionService initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PredictionService: {str(e)}")
            return False
    
    def _prepare_features(self, price_data: List[Dict]) -> np.ndarray:
        """
        Prepare feature matrix from price data.
        
        Args:
            price_data: List of price data dictionaries
            
        Returns:
            Feature matrix (samples, features)
        """
        if len(price_data) < self.config.sequence_length:
            raise ValueError(f"Need at least {self.config.sequence_length} data points")
        
        # Convert to DataFrame for easier feature engineering
        df = pd.DataFrame(price_data)
        
        # Ensure required columns exist (use price as fallback)
        for col in ['high', 'low', 'open', 'volume']:
            if col not in df.columns:
                df[col] = df['price']  # Fallback to price
        
        # Feature engineering
        features = {}
        
        # Basic price features
        features['price'] = df['price'].values
        features['volume'] = df['volume'].values
        features['high'] = df['high'].values
        features['low'] = df['low'].values
        features['open'] = df['open'].values
        
        # Moving averages
        features['price_ma_5'] = df['price'].rolling(5, min_periods=1).mean().values
        features['price_ma_20'] = df['price'].rolling(20, min_periods=1).mean().values
        features['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean().values
        
        # Technical indicators (simplified)
        features['rsi'] = self._calculate_rsi(df['price'].values)
        features['macd'] = self._calculate_macd(df['price'].values)
        
        # Bollinger bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(df['price'].values)
        features['bollinger_upper'] = bb_upper
        features['bollinger_lower'] = bb_lower
        
        # Combine features into matrix
        feature_matrix = np.column_stack([features[col] for col in self.feature_columns])
        
        # Normalize features (simple min-max scaling)
        feature_matrix = self._normalize_features(feature_matrix)
        
        return feature_matrix
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='same')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='same')
        
        rs = avg_gains / (avg_losses + 1e-8)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Pad to match input length
        return np.concatenate([[50], rsi])[:len(prices)]
    
    def _calculate_macd(self, prices: np.ndarray) -> np.ndarray:
        """Calculate MACD indicator."""
        ema12 = pd.Series(prices).ewm(span=12).mean().values
        ema26 = pd.Series(prices).ewm(span=26).mean().values
        macd = ema12 - ema26
        return macd
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        rolling_mean = pd.Series(prices).rolling(period, min_periods=1).mean().values
        rolling_std = pd.Series(prices).rolling(period, min_periods=1).std().values
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        return upper_band, lower_band
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Simple feature normalization."""
        # Avoid division by zero
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)
        feature_range = feature_max - feature_min
        feature_range = np.where(feature_range == 0, 1, feature_range)  # Avoid division by zero
        
        normalized = (features - feature_min) / feature_range
        return normalized
    
    def predict(self, 
                price_data: List[Dict], 
                store_prediction: bool = True) -> Optional[Dict[str, Any]]:
        """
        Make a price prediction with uncertainty quantification.
        
        Args:
            price_data: Recent price data (must contain at least sequence_length points)
            store_prediction: Whether to store prediction in database
            
        Returns:
            Prediction result dictionary or None if error
        """
        if not self.is_initialized:
            logger.error("PredictionService not initialized")
            return None
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(price_data)
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.prediction_stats['cache_hits'] += 1
                    logger.debug("Returning cached prediction")
                    return cached_result
                else:
                    self.prediction_stats['cache_misses'] += 1
            
            # Prepare features
            features = self._prepare_features(price_data)
            
            # Create sequence for prediction (last sequence_length points)
            if len(features) < self.config.sequence_length:
                raise ValueError(f"Insufficient data: need {self.config.sequence_length}, got {len(features)}")
            
            sequence = features[-self.config.sequence_length:].reshape(1, self.config.sequence_length, -1)
            
            # Make prediction with uncertainty
            pred_result = self.model.predict_with_uncertainty(sequence, n_samples=self.config.monte_carlo_samples)
            
            # Extract results for single prediction
            current_price = price_data[-1]['price']
            predicted_price = float(pred_result['predictions'][0])
            uncertainty = float(pred_result['uncertainties'][0])
            confidence_lower = float(pred_result['confidence_lower'][0])
            confidence_upper = float(pred_result['confidence_upper'][0])
            
            # Calculate additional metrics
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            interval_width_pct = ((confidence_upper - confidence_lower) / current_price) * 100
            
            # Compile result
            prediction = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change_pct,
                'uncertainty': uncertainty,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper,
                'interval_width_pct': interval_width_pct,
                'confidence_interval': self.config.confidence_interval,
                'monte_carlo_samples': self.config.monte_carlo_samples,
                'model_version': self.model.model_version,
                'feature_count': len(self.feature_columns)
            }
            
            # Store in cache
            if self.cache:
                self.cache.put(cache_key, prediction)
            
            # Store in database
            if store_prediction and self.db_manager:
                self._store_prediction(prediction)
            
            # Update stats
            self.prediction_stats['total_predictions'] += 1
            self.prediction_stats['last_prediction_time'] = datetime.now()
            
            logger.info(f"Prediction: ${predicted_price:,.0f} ({price_change_pct:+.2f}%) "
                       f"CI: [{confidence_lower:,.0f}, {confidence_upper:,.0f}] "
                       f"Width: {interval_width_pct:.1f}%")
            
            return prediction
            
        except Exception as e:
            self.prediction_stats['error_count'] += 1
            logger.error(f"Prediction error: {str(e)}")
            return None
    
    def _generate_cache_key(self, price_data: List[Dict]) -> str:
        """Generate cache key from price data."""
        # Use last few prices and timestamps
        recent_data = price_data[-5:] if len(price_data) >= 5 else price_data
        key_parts = []
        for data_point in recent_data:
            key_parts.append(f"{data_point['price']:.2f}")
        return "_".join(key_parts)
    
    def _store_prediction(self, prediction: Dict[str, Any]):
        """Store prediction in database."""
        try:
            prediction_data = PredictionData(
                timestamp=int(prediction['timestamp'].timestamp()),
                predicted_price=prediction['predicted_price'],
                confidence_lower=prediction['confidence_lower'],
                confidence_upper=prediction['confidence_upper'],
                model_version=prediction['model_version']
            )
            
            self.db_manager.store_prediction(prediction_data)
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {str(e)}")
    
    def evaluate_recent_performance(self, 
                                   actual_prices: List[Dict],
                                   lookback_hours: int = 24) -> Optional[Dict[str, float]]:
        """
        Evaluate model performance on recent predictions.
        
        Args:
            actual_prices: Recent actual price data
            lookback_hours: How far back to evaluate
            
        Returns:
            Performance metrics dictionary
        """
        if not self.db_manager:
            logger.warning("Cannot evaluate performance without database manager")
            return None
        
        try:
            # Get recent predictions from database
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            recent_predictions = self.db_manager.get_recent_predictions(hours=lookback_hours)
            
            if not recent_predictions:
                logger.info("No recent predictions found for evaluation")
                return None
            
            # Match predictions with actual prices
            matched_pairs = []
            
            for pred in recent_predictions:
                pred_time = datetime.fromtimestamp(pred['timestamp'])
                
                # Find closest actual price (within 1 hour)
                closest_actual = None
                min_time_diff = timedelta(hours=1)
                
                for actual in actual_prices:
                    actual_time = actual.get('timestamp', datetime.now())
                    if isinstance(actual_time, int):
                        actual_time = datetime.fromtimestamp(actual_time)
                    
                    time_diff = abs(actual_time - pred_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_actual = actual
                
                if closest_actual:
                    matched_pairs.append({
                        'predicted': pred['predicted_price'],
                        'actual': closest_actual['price'],
                        'confidence_lower': pred['confidence_lower'],
                        'confidence_upper': pred['confidence_upper']
                    })
            
            if not matched_pairs:
                logger.info("No matching prediction-actual pairs found")
                return None
            
            # Calculate metrics
            predictions = np.array([p['predicted'] for p in matched_pairs])
            actuals = np.array([p['actual'] for p in matched_pairs])
            
            mae = np.mean(np.abs(predictions - actuals))
            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            
            # Directional accuracy
            pred_direction = np.sign(np.diff(predictions))
            actual_direction = np.sign(np.diff(actuals))
            directional_accuracy = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0.0
            
            # Interval coverage
            in_interval = np.array([
                p['confidence_lower'] <= p['actual'] <= p['confidence_upper']
                for p in matched_pairs
            ])
            interval_coverage = np.mean(in_interval)
            
            metrics = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'directional_accuracy': float(directional_accuracy),
                'interval_coverage': float(interval_coverage),
                'expected_coverage': self.config.confidence_interval,
                'n_samples': len(matched_pairs),
                'evaluation_period_hours': lookback_hours,
                'model_version': self.model.model_version if self.model else None
            }
            
            # Store model metrics in database
            if self.db_manager:
                model_metrics = ModelMetrics(
                    timestamp=int(datetime.now().timestamp()),
                    mse=mse,
                    mae=mae,
                    directional_accuracy=directional_accuracy,
                    model_version=self.model.model_version if self.model else "unknown",
                    sharpe_ratio=None  # Could be calculated if we have returns data
                )
                
                self.db_manager.store_model_metrics(model_metrics)
            
            logger.info(f"Model evaluation: MAE={mae:.2f}, Accuracy={directional_accuracy:.3f}, "
                       f"Coverage={interval_coverage:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {str(e)}")
            return None
    
    def check_model_health(self) -> Dict[str, Any]:
        """
        Check model health and performance status.
        
        Returns:
            Health status dictionary
        """
        health = {
            'is_initialized': self.is_initialized,
            'model_available': self.model is not None,
            'model_version': self.model.model_version if self.model else None,
            'prediction_stats': self.prediction_stats.copy(),
            'cache_enabled': self.cache is not None,
            'database_connected': self.db_manager is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.cache:
            health['cache_stats'] = {
                'size': len(self.cache.cache),
                'hit_rate': self.prediction_stats['cache_hits'] / max(1, self.prediction_stats['total_predictions']),
                'max_size': self.cache.max_size,
                'ttl_seconds': self.cache.ttl_seconds
            }
        
        return health
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        stats = {
            'service_info': {
                'initialized': self.is_initialized,
                'model_version': self.model.model_version if self.model else None,
                'feature_count': len(self.feature_columns),
                'sequence_length': self.config.sequence_length
            },
            'prediction_stats': self.prediction_stats.copy(),
            'performance_config': {
                'monte_carlo_samples': self.config.monte_carlo_samples,
                'confidence_interval': self.config.confidence_interval,
                'caching_enabled': self.enable_caching
            }
        }
        
        if self.model:
            model_summary = self.model.get_model_summary()
            stats['model_info'] = model_summary
        
        return stats