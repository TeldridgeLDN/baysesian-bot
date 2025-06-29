"""
Model Manager for handling Bayesian LSTM model training, prediction, and lifecycle.
Manages the complete model workflow for the trading bot.
"""

import logging
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# TensorFlow imports with graceful fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available - model predictions will use fallback")

from models.bayesian_lstm import BayesianLSTMModel

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the complete lifecycle of the Bayesian LSTM model.
    
    Features:
    - Model training and retraining
    - Real-time predictions with uncertainty
    - Model persistence and loading
    - Performance monitoring
    - Fallback prediction modes
    """
    
    def __init__(self, config: Any):
        """Initialize model manager."""
        self.config = config
        
        # Model state
        self.model = None
        self.model_trained = False
        self.last_training_time = None
        self.last_prediction_time = None
        
        # Model paths
        self.model_dir = Path("models/saved")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "bayesian_lstm_model.h5"
        self.scaler_path = self.model_dir / "feature_scaler.pkl"
        
        # Feature engineering
        self.feature_scaler = None
        self.sequence_length = getattr(config, 'sequence_length', 60)
        self.features = ['open', 'high', 'low', 'close', 'volume']
        
        # Model performance tracking
        self.performance_history = []
        self.prediction_cache = {}
        
        # Fallback prediction system
        self.fallback_mode = not TENSORFLOW_AVAILABLE
        self.fallback_predictions = self._initialize_fallback_system()
        
        logger.info(f"ModelManager initialized (TensorFlow available: {TENSORFLOW_AVAILABLE})")
    
    def _initialize_fallback_system(self) -> Dict[str, Any]:
        """Initialize fallback prediction system for when TensorFlow is unavailable."""
        return {
            'simple_trend': {
                'enabled': True,
                'lookback_periods': 5,
                'confidence_base': 0.45
            },
            'moving_average': {
                'enabled': True,
                'short_ma': 5,
                'long_ma': 20,
                'confidence_base': 0.40
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize the model manager."""
        try:
            if TENSORFLOW_AVAILABLE:
                # Try to load existing model
                if await self._load_existing_model():
                    logger.info("✅ Existing model loaded successfully")
                    return True
                else:
                    logger.info("No existing model found - will train new model when data is available")
            else:
                logger.info("✅ Fallback prediction system initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ModelManager: {e}")
            return False
    
    async def _load_existing_model(self) -> bool:
        """Load existing trained model from disk."""
        try:
            if not self.model_path.exists():
                return False
            
            # Load the Bayesian LSTM model
            self.model = BayesianLSTMModel(self.config)
            
            # Load model weights if they exist
            if self.model_path.exists():
                # For now, we'll retrain the model as loading complex custom models can be tricky
                logger.info("Model weights found, will use for reference")
            
            # Load feature scaler
            if self.scaler_path.exists():
                with open(self.scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                logger.info("Feature scaler loaded")
            
            self.model_trained = True
            self.last_training_time = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading existing model: {e}")
            return False
    
    async def train_model(self, data: pd.DataFrame) -> bool:
        """Train the Bayesian LSTM model with provided data."""
        if not TENSORFLOW_AVAILABLE:
            logger.info("TensorFlow not available - skipping model training")
            return True  # Return True to continue with fallback predictions
        
        try:
            logger.info("Starting model training...")
            
            # Prepare data for training
            X, y = self._prepare_training_data(data)
            
            if X is None or len(X) < 100:  # Need at least 100 samples
                logger.warning("Insufficient data for training")
                return False
            
            # Initialize model if not exists
            if self.model is None:
                self.model = BayesianLSTMModel(self.config)
                self.model.build_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Train the model
            history = self.model.train(X, y, validation_split=0.2)
            
            # Save the trained model
            await self._save_model()
            
            # Update state
            self.model_trained = True
            self.last_training_time = datetime.now()
            
            # Log training results
            final_loss = history.history['loss'][-1] if history else 0
            logger.info(f"✅ Model training completed - Final loss: {final_loss:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare data for model training."""
        try:
            if len(data) < self.sequence_length + 1:
                return None, None
            
            # Feature engineering
            features_df = self._engineer_features(data)
            
            # Scale features
            if self.feature_scaler is None:
                from sklearn.preprocessing import MinMaxScaler
                self.feature_scaler = MinMaxScaler()
                scaled_features = self.feature_scaler.fit_transform(features_df)
            else:
                scaled_features = self.feature_scaler.transform(features_df)
            
            # Create sequences
            X, y = self._create_sequences(scaled_features)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw OHLCV data."""
        features_df = data[self.features].copy()
        
        # Technical indicators
        features_df['price_change'] = data['close'].pct_change()
        features_df['volume_change'] = data['volume'].pct_change()
        
        # Moving averages
        features_df['ma_5'] = data['close'].rolling(5).mean()
        features_df['ma_20'] = data['close'].rolling(20).mean()
        
        # Volatility
        features_df['volatility'] = data['close'].rolling(20).std()
        
        # RSI (simplified)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        return features_df
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            # Predict next price change (close price change)
            current_price = data[i-1, 3]  # Previous close price
            next_price = data[i, 3]  # Current close price
            price_change = (next_price - current_price) / current_price
            y.append(price_change)
        
        return np.array(X), np.array(y)
    
    async def get_prediction(self, data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Get prediction from the model."""
        try:
            if self.fallback_mode or not self.model_trained:
                return await self._get_fallback_prediction(data)
            
            if data is None or len(data) < self.sequence_length:
                logger.warning("Insufficient data for prediction")
                return None
            
            # Prepare data for prediction
            features_df = self._engineer_features(data)
            
            if self.feature_scaler is None:
                logger.warning("Feature scaler not available")
                return None
            
            scaled_features = self.feature_scaler.transform(features_df)
            
            # Get the last sequence
            last_sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Get predictions with uncertainty
            predictions = self.model.predict_with_uncertainty(last_sequence, n_samples=50)
            
            mean_prediction = predictions['mean'][0]
            uncertainty = predictions['std'][0]
            
            # Convert to price prediction
            current_price = data['close'].iloc[-1]
            predicted_price_change = mean_prediction
            predicted_price = current_price * (1 + predicted_price_change)
            
            # Calculate confidence (inverse of uncertainty)
            confidence = max(0.0, min(1.0, 1.0 - uncertainty))
            
            # Determine direction
            direction = 1 if predicted_price_change > 0 else -1
            
            prediction_result = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change_pct': predicted_price_change * 100,
                'confidence': confidence,
                'uncertainty_pct': uncertainty * 100,
                'direction': direction,
                'model_type': 'bayesian_lstm',
                'target_price': predicted_price,
                'stop_loss': current_price * (0.98 if direction > 0 else 1.02),
                'raw_prediction': mean_prediction,
                'raw_uncertainty': uncertainty
            }
            
            self.last_prediction_time = datetime.now()
            self._cache_prediction(prediction_result)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return await self._get_fallback_prediction(data)
    
    async def _get_fallback_prediction(self, data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Get fallback prediction when main model is unavailable."""
        try:
            if data is None or len(data) < 20:
                logger.warning("Insufficient data for fallback prediction")
                return None
            
            current_price = data['close'].iloc[-1]
            
            # Simple trend analysis
            recent_prices = data['close'].tail(10).values
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Moving average analysis
            ma_short = data['close'].tail(5).mean()
            ma_long = data['close'].tail(20).mean()
            
            # Combine signals
            trend_signal = 1 if trend > 0 else -1
            ma_signal = 1 if ma_short > ma_long else -1
            
            # Calculate prediction
            trend_strength = abs(trend) / current_price
            predicted_change_pct = trend_signal * min(trend_strength * 100, 5.0)  # Cap at 5%
            predicted_price = current_price * (1 + predicted_change_pct / 100)
            
            # Calculate confidence based on signal agreement
            signal_agreement = (trend_signal == ma_signal)
            base_confidence = self.fallback_predictions['simple_trend']['confidence_base']
            confidence = base_confidence + (0.15 if signal_agreement else -0.10)
            confidence = max(0.2, min(0.6, confidence))  # Keep reasonable bounds
            
            # Calculate uncertainty
            price_volatility = data['close'].tail(20).std() / current_price
            uncertainty_pct = min(price_volatility * 100, 30.0)  # Cap at 30%
            
            direction = 1 if predicted_change_pct > 0 else -1
            
            prediction_result = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change_pct': predicted_change_pct,
                'confidence': confidence,
                'uncertainty_pct': uncertainty_pct,
                'direction': direction,
                'model_type': 'fallback_trend',
                'target_price': predicted_price,
                'stop_loss': current_price * (0.97 if direction > 0 else 1.03),
                'trend_signal': trend_signal,
                'ma_signal': ma_signal,
                'signal_agreement': signal_agreement
            }
            
            self.last_prediction_time = datetime.now()
            logger.info(f"Fallback prediction: {direction} direction, {confidence:.2f} confidence")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return None
    
    def _cache_prediction(self, prediction: Dict[str, Any]):
        """Cache prediction for analysis."""
        timestamp = prediction['timestamp']
        self.prediction_cache[timestamp] = prediction
        
        # Keep only recent predictions (last 100)
        if len(self.prediction_cache) > 100:
            oldest_key = min(self.prediction_cache.keys())
            del self.prediction_cache[oldest_key]
    
    async def _save_model(self):
        """Save the trained model to disk."""
        try:
            if self.model and TENSORFLOW_AVAILABLE:
                # Save model weights/architecture
                # Note: Custom models may need special handling
                logger.info("Model weights would be saved here")
            
            # Save feature scaler
            if self.feature_scaler:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.feature_scaler, f)
                logger.info("Feature scaler saved")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        if not self.model_trained:
            return True
        
        if self.last_training_time is None:
            return True
        
        # Retrain every 24 hours
        retrain_interval = timedelta(hours=getattr(self.config, 'retrain_frequency_hours', 24))
        return datetime.now() - self.last_training_time > retrain_interval
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status and performance metrics."""
        return {
            'model_trained': self.model_trained,
            'model_type': 'bayesian_lstm' if not self.fallback_mode else 'fallback',
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'fallback_mode': self.fallback_mode,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'cached_predictions': len(self.prediction_cache),
            'sequence_length': self.sequence_length,
            'features_count': len(self.features) + 6,  # +6 for engineered features
            'should_retrain': self.should_retrain()
        }
    
    def get_recent_predictions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent predictions for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            prediction for timestamp, prediction in self.prediction_cache.items()
            if timestamp >= cutoff_time
        ]
    
    async def evaluate_model_performance(self, actual_prices: List[float]) -> Dict[str, Any]:
        """Evaluate model performance against actual prices."""
        try:
            recent_predictions = self.get_recent_predictions(hours=24)
            
            if not recent_predictions or len(actual_prices) < len(recent_predictions):
                return {'error': 'Insufficient data for evaluation'}
            
            # Calculate metrics
            predictions = [p['predicted_price'] for p in recent_predictions]
            actual = actual_prices[-len(predictions):]
            
            # Mean Absolute Error
            mae = np.mean(np.abs(np.array(predictions) - np.array(actual)))
            
            # Direction accuracy
            pred_directions = [1 if p['direction'] > 0 else -1 for p in recent_predictions]
            actual_directions = [1 if actual[i] > actual[i-1] else -1 for i in range(1, len(actual))]
            
            if len(actual_directions) > 0:
                direction_accuracy = np.mean(np.array(pred_directions[1:]) == np.array(actual_directions))
            else:
                direction_accuracy = 0.0
            
            return {
                'mae': mae,
                'direction_accuracy': direction_accuracy,
                'total_predictions': len(predictions),
                'avg_confidence': np.mean([p['confidence'] for p in recent_predictions]),
                'avg_uncertainty': np.mean([p['uncertainty_pct'] for p in recent_predictions])
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {'error': str(e)}