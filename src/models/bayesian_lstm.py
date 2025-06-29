"""
Bayesian LSTM Model Implementation for cryptocurrency price prediction.
Implements Monte Carlo Dropout for uncertainty quantification per PRD Section 4.2.
Automatically falls back to TensorFlow-free ensemble approach if TensorFlow unavailable.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

# Try to import TensorFlow, fallback to ensemble approach if unavailable
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, optimizers, callbacks
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow available - using TensorFlow implementation")
except ImportError as e:
    logger.warning(f"TensorFlow not available ({str(e)}) - falling back to ensemble implementation")
    # Import the alternative implementation
    try:
        from .bayesian_lstm_alternative import EnsembleBayesianModel, BayesianLSTM as EnsembleBayesianLSTM, ModelManager as EnsembleModelManager
        logger.info("Successfully loaded TensorFlow-free ensemble implementation")
    except ImportError as e2:
        logger.error(f"Failed to load ensemble fallback: {str(e2)}")
        raise ImportError("Neither TensorFlow nor ensemble fallback is available")

# TensorFlow-specific implementations (only available when TensorFlow is imported)
if TENSORFLOW_AVAILABLE:
    class MonteCarloDropout(layers.Layer):
        """Custom Monte Carlo Dropout layer that stays active during inference."""
        
        def __init__(self, rate: float, **kwargs):
            super(MonteCarloDropout, self).__init__(**kwargs)
            self.rate = rate
            
        def call(self, inputs, training=None):
            """Apply dropout regardless of training mode for uncertainty estimation."""
            return tf.nn.dropout(inputs, rate=self.rate)
        
        def get_config(self):
            config = super().get_config()
            config.update({'rate': self.rate})
            return config

    class BayesianLSTM:
        """
        Bayesian LSTM model for cryptocurrency price prediction with uncertainty quantification.
        
        Architecture:
        - LSTM layers: 128 → 64 → 32 units
        - Dropout: 0.2 for LSTM layers, 0.3 for dense layer
        - Monte Carlo Dropout for uncertainty estimation
        - 95% confidence intervals from 100 forward passes
        """
        
        def __init__(self, config: Dict[str, Any]):
            """
            Initialize Bayesian LSTM model.
            
            Args:
                config: Model configuration dictionary containing:
                    - sequence_length: Input sequence length
                    - feature_count: Number of features per timestep
                    - lstm_units: List of LSTM layer units
                    - dropout_rate: Dropout rate for LSTM layers
                    - recurrent_dropout_rate: Recurrent dropout rate
                    - dense_dropout_rate: Dense layer dropout rate
                    - learning_rate: Learning rate
                    - monte_carlo_samples: Number of MC samples for uncertainty
                    - confidence_interval: Confidence interval (e.g., 0.95)
            """
            self.config = config
            self.model = None
            self.history = None
            self.model_version = None
            self.training_stats = {}
            
            # Extract configuration (handle both dict and object)
            def get_config_value(key, default):
                if isinstance(config, dict):
                    return config.get(key, default)
                else:
                    return getattr(config, key, default)
            
            self.sequence_length = get_config_value('sequence_length', 21)  # Optimized based on research findings
            self.feature_count = get_config_value('feature_count', 47)  # Default from TASK-005
            self.lstm_units = get_config_value('lstm_units', [128, 64, 32])
            self.dropout_rate = get_config_value('dropout_rate', 0.2)
            self.recurrent_dropout_rate = get_config_value('recurrent_dropout_rate', 0.2)
            self.dense_dropout_rate = get_config_value('dense_dropout_rate', 0.3)
            self.learning_rate = get_config_value('learning_rate', 0.001)
            self.monte_carlo_samples = get_config_value('monte_carlo_samples', 100)
            self.confidence_interval = get_config_value('confidence_interval', 0.95)
            
            logger.info(f"BayesianLSTM initialized with {len(self.lstm_units)} LSTM layers: {self.lstm_units}")
            
        def build_model(self) -> Model:
            """
            Build the Bayesian LSTM model architecture.
            
            Returns:
                Compiled Keras model
            """
            logger.info("Building Bayesian LSTM model architecture")
            
            # Input layer
            inputs = keras.Input(shape=(self.sequence_length, self.feature_count), name='price_sequence')
            
            # LSTM layers with dropout
            x = inputs
            for i, units in enumerate(self.lstm_units):
                return_sequences = i < len(self.lstm_units) - 1  # All but last layer return sequences
                
                x = layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    name=f'lstm_{i+1}'
                )(x)
                
                logger.info(f"Added LSTM layer {i+1}: {units} units, return_sequences={return_sequences}")
            
            # Dense layers with Monte Carlo dropout
            x = layers.Dense(16, activation='relu', name='dense_1')(x)
            x = MonteCarloDropout(rate=self.dense_dropout_rate, name='mc_dropout')(x)
            
            # Output layer for price prediction
            outputs = layers.Dense(1, activation='linear', name='price_prediction')(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs, name='bayesian_lstm')
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            self.model = model
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            logger.info(f"Model compiled with version: {self.model_version}")
            logger.info(f"Model summary: {model.count_params()} parameters")
            
            return model
        
        def train(self, 
                  X_train: np.ndarray, 
                  y_train: np.ndarray,
                  X_val: np.ndarray, 
                  y_val: np.ndarray,
                  epochs: int = 100,
                  batch_size: int = 32,
                  early_stopping_patience: int = 10,
                  verbose: int = 1) -> Dict[str, Any]:
            """
            Train the Bayesian LSTM model.
            
            Args:
                X_train: Training sequences (samples, sequence_length, features)
                y_train: Training targets (samples,)
                X_val: Validation sequences
                y_val: Validation targets
                epochs: Maximum number of epochs
                batch_size: Training batch size
                early_stopping_patience: Early stopping patience
                verbose: Verbosity level
                
            Returns:
                Training statistics dictionary
            """
            if self.model is None:
                self.build_model()
            
            logger.info(f"Starting model training: {len(X_train)} train, {len(X_val)} val samples")
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=verbose
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=verbose
                ),
                callbacks.ModelCheckpoint(
                    filepath=f'models/bayesian_lstm_{self.model_version}_best.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=verbose
                )
            ]
            
            # Train model
            start_time = datetime.now()
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=verbose,
                shuffle=True
            )
            
            training_time = datetime.now() - start_time
            
            # Calculate training statistics
            self.training_stats = {
                'model_version': self.model_version,
                'training_time_seconds': training_time.total_seconds(),
                'epochs_trained': len(self.history.history['loss']),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'final_train_mae': float(self.history.history['mae'][-1]),
                'final_val_mae': float(self.history.history['val_mae'][-1]),
                'best_val_loss': float(min(self.history.history['val_loss'])),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'config': self.config.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Training completed in {training_time}")
            logger.info(f"Final validation loss: {self.training_stats['final_val_loss']:.6f}")
            logger.info(f"Final validation MAE: {self.training_stats['final_val_mae']:.6f}")
            
            return self.training_stats
        
        def predict_with_uncertainty(self, 
                                    X: np.ndarray, 
                                    n_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
            """
            Make predictions with uncertainty quantification using Monte Carlo Dropout.
            
            Args:
                X: Input sequences (samples, sequence_length, features)
                n_samples: Number of Monte Carlo samples (default: from config)
                
            Returns:
                Dictionary containing:
                    - predictions: Mean predictions
                    - uncertainties: Prediction standard deviations
                    - confidence_lower: Lower confidence bounds
                    - confidence_upper: Upper confidence bounds
                    - all_samples: All MC samples (for analysis)
            """
            if self.model is None:
                raise ValueError("Model must be trained before making predictions")
            
            if n_samples is None:
                n_samples = self.monte_carlo_samples
            
            logger.info(f"Generating predictions with {n_samples} MC samples for {len(X)} sequences")
            
            # Collect predictions from multiple forward passes
            mc_predictions = []
            
            for i in range(n_samples):
                # Forward pass with dropout active (Monte Carlo sampling)
                pred = self.model(X, training=True)  # training=True keeps dropout active
                mc_predictions.append(pred.numpy().flatten())
            
            # Convert to numpy array: (n_samples, n_predictions)
            mc_predictions = np.array(mc_predictions)
            
            # Calculate statistics
            predictions = np.mean(mc_predictions, axis=0)
            uncertainties = np.std(mc_predictions, axis=0)
            
            # Calculate confidence intervals
            alpha = 1 - self.confidence_interval
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            confidence_lower = np.percentile(mc_predictions, lower_percentile, axis=0)
            confidence_upper = np.percentile(mc_predictions, upper_percentile, axis=0)
            
            # Calculate additional uncertainty metrics
            prediction_variance = np.var(mc_predictions, axis=0)
            epistemic_uncertainty = uncertainties  # Model uncertainty
            
            results = {
                'predictions': predictions,
                'uncertainties': uncertainties,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper,
                'prediction_variance': prediction_variance,
                'epistemic_uncertainty': epistemic_uncertainty,
                'all_samples': mc_predictions,
                'confidence_interval': self.confidence_interval,
                'n_samples': n_samples
            }
            
            logger.info(f"Predictions generated: mean uncertainty = {np.mean(uncertainties):.4f}")
            
            return results
        
        def predict_single(self, X: np.ndarray) -> Dict[str, float]:
            """
            Make a single prediction with uncertainty for the latest sequence.
            
            Args:
                X: Single input sequence (1, sequence_length, features)
                
            Returns:
                Dictionary with prediction results
            """
            if len(X.shape) == 2:
                X = X.reshape(1, X.shape[0], X.shape[1])
            
            results = self.predict_with_uncertainty(X)
            
            return {
                'prediction': float(results['predictions'][0]),
                'uncertainty': float(results['uncertainties'][0]),
                'confidence_lower': float(results['confidence_lower'][0]),
                'confidence_upper': float(results['confidence_upper'][0]),
                'confidence_interval': self.confidence_interval,
                'timestamp': datetime.now().isoformat()
            }
        
        def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
            """
            Evaluate model performance on test data.
            
            Args:
                X_test: Test sequences
                y_test: Test targets
                
            Returns:
                Performance metrics dictionary
            """
            if self.model is None:
                raise ValueError("Model must be trained before evaluation")
            
            logger.info(f"Evaluating model on {len(X_test)} test samples")
            
            # Standard model evaluation
            test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Uncertainty-aware predictions
            pred_results = self.predict_with_uncertainty(X_test)
            predictions = pred_results['predictions']
            uncertainties = pred_results['uncertainties']
            
            # Calculate additional metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(mse)
            
            # Directional accuracy
            pred_direction = np.sign(np.diff(predictions))
            true_direction = np.sign(np.diff(y_test))
            directional_accuracy = np.mean(pred_direction == true_direction) if len(pred_direction) > 0 else 0.0
            
            # Uncertainty metrics
            mean_uncertainty = np.mean(uncertainties)
            uncertainty_std = np.std(uncertainties)
            
            # Confidence interval coverage
            in_interval = ((y_test >= pred_results['confidence_lower']) & 
                          (y_test <= pred_results['confidence_upper']))
            interval_coverage = np.mean(in_interval)
            
            # Interval width
            interval_width = pred_results['confidence_upper'] - pred_results['confidence_lower']
            mean_interval_width = np.mean(interval_width)
            
            metrics = {
                'test_loss': float(test_loss),
                'test_mae': float(test_mae),
                'test_mse': float(test_mse),
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'directional_accuracy': float(directional_accuracy),
                'mean_uncertainty': float(mean_uncertainty),
                'uncertainty_std': float(uncertainty_std),
                'interval_coverage': float(interval_coverage),
                'mean_interval_width': float(mean_interval_width),
                'expected_coverage': self.confidence_interval,
                'n_test_samples': len(X_test),
                'model_version': self.model_version
            }
            
            logger.info(f"Model evaluation completed:")
            logger.info(f"  MAE: {mae:.6f}")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  Directional Accuracy: {directional_accuracy:.3f}")
            logger.info(f"  Interval Coverage: {interval_coverage:.3f} (expected: {self.confidence_interval:.3f})")
            
            return metrics
        
        def save_model(self, filepath: str, include_history: bool = True):
            """
            Save the trained model and metadata.
            
            Args:
                filepath: Path to save the model (without extension)
                include_history: Whether to save training history
            """
            if self.model is None:
                raise ValueError("No model to save")
            
            # Save model architecture and weights
            model_path = f"{filepath}.keras"
            self.model.save(model_path)
            
            # Save metadata
            metadata = {
                'model_version': self.model_version,
                'config': self.config,
                'training_stats': self.training_stats,
                'model_architecture': {
                    'sequence_length': self.sequence_length,
                    'feature_count': self.feature_count,
                    'lstm_units': self.lstm_units,
                    'dropout_rate': self.dropout_rate,
                    'recurrent_dropout_rate': self.recurrent_dropout_rate,
                    'dense_dropout_rate': self.dense_dropout_rate
                },
                'saved_at': datetime.now().isoformat()
            }
            
            if include_history and self.history is not None:
                metadata['training_history'] = {
                    'loss': [float(x) for x in self.history.history['loss']],
                    'val_loss': [float(x) for x in self.history.history['val_loss']],
                    'mae': [float(x) for x in self.history.history['mae']],
                    'val_mae': [float(x) for x in self.history.history['val_mae']]
                }
            
            metadata_path = f"{filepath}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")
        
        def load_model(self, filepath: str) -> bool:
            """
            Load a saved model and metadata.
            
            Args:
                filepath: Path to the saved model (without extension)
                
            Returns:
                True if successful, False otherwise
            """
            try:
                # Load model
                model_path = f"{filepath}.keras"
                
                # Register custom layer
                custom_objects = {'MonteCarloDropout': MonteCarloDropout}
                self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
                
                # Load metadata
                metadata_path = f"{filepath}_metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.model_version = metadata['model_version']
                self.config.update(metadata['config'])
                self.training_stats = metadata.get('training_stats', {})
                
                # Update configuration from metadata
                arch = metadata.get('model_architecture', {})
                self.sequence_length = arch.get('sequence_length', self.sequence_length)
                self.feature_count = arch.get('feature_count', self.feature_count)
                self.lstm_units = arch.get('lstm_units', self.lstm_units)
                
                logger.info(f"Model loaded successfully: {self.model_version}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                return False
        
        def get_model_summary(self) -> Dict[str, Any]:
            """
            Get comprehensive model summary.
            
            Returns:
                Dictionary with model information
            """
            if self.model is None:
                return {'error': 'No model available'}
            
            summary = {
                'model_version': self.model_version,
                'architecture': {
                    'total_params': self.model.count_params(),
                    'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                    'sequence_length': self.sequence_length,
                    'feature_count': self.feature_count,
                    'lstm_layers': len(self.lstm_units),
                    'lstm_units': self.lstm_units
                },
                'configuration': {
                    'dropout_rate': self.dropout_rate,
                    'recurrent_dropout_rate': self.recurrent_dropout_rate,
                    'dense_dropout_rate': self.dense_dropout_rate,
                    'learning_rate': self.learning_rate,
                    'monte_carlo_samples': self.monte_carlo_samples,
                    'confidence_interval': self.confidence_interval
                },
                'training_stats': self.training_stats,
                'created_at': datetime.now().isoformat()
            }
            
            return summary
        
        def check_model_degradation(self, 
                                   current_metrics: Dict[str, float],
                                   threshold_mae_increase: float = 0.2,
                                   threshold_accuracy_decrease: float = 0.1) -> Dict[str, Any]:
            """
            Check if model performance has degraded significantly.
            
            Args:
                current_metrics: Current model performance metrics
                threshold_mae_increase: MAE increase threshold for degradation
                threshold_accuracy_decrease: Accuracy decrease threshold
                
            Returns:
                Degradation analysis results
            """
            if not self.training_stats:
                return {'degradation_detected': False, 'reason': 'No baseline metrics available'}
            
            baseline_mae = self.training_stats.get('final_val_mae', float('inf'))
            current_mae = current_metrics.get('mae', 0)
            
            baseline_accuracy = self.training_stats.get('directional_accuracy', 0)
            current_accuracy = current_metrics.get('directional_accuracy', 0)
            
            # Check for degradation
            mae_increase = (current_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0
            accuracy_decrease = baseline_accuracy - current_accuracy
            
            degradation_detected = (
                mae_increase > threshold_mae_increase or 
                accuracy_decrease > threshold_accuracy_decrease
            )
            
            analysis = {
                'degradation_detected': degradation_detected,
                'mae_increase_pct': mae_increase * 100,
                'accuracy_decrease_pct': accuracy_decrease * 100,
                'baseline_mae': baseline_mae,
                'current_mae': current_mae,
                'baseline_accuracy': baseline_accuracy,
                'current_accuracy': current_accuracy,
                'thresholds': {
                    'mae_increase_threshold': threshold_mae_increase,
                    'accuracy_decrease_threshold': threshold_accuracy_decrease
                },
                'recommendation': 'retrain' if degradation_detected else 'continue',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            if degradation_detected:
                logger.warning(f"Model degradation detected: MAE +{mae_increase*100:.1f}%, Accuracy -{accuracy_decrease*100:.1f}%")
            
            return analysis

    class ModelManager:
        """Manager class for handling multiple model versions and operations."""
        
        def __init__(self, base_path: str = "models/"):
            """
            Initialize model manager.
            
            Args:
                base_path: Base directory for storing models
            """
            self.base_path = base_path
            self.current_model = None
            self.model_registry = {}
            
            # Create models directory if it doesn't exist
            import os
            os.makedirs(base_path, exist_ok=True)
            
        def create_model(self, config: Dict[str, Any]) -> BayesianLSTM:
            """
            Create a new Bayesian LSTM model.
            
            Args:
                config: Model configuration
                
            Returns:
                BayesianLSTM instance
            """
            model = BayesianLSTM(config)
            model.build_model()
            
            self.current_model = model
            self.model_registry[model.model_version] = model
            
            logger.info(f"Created new model: {model.model_version}")
            return model
        
        def save_current_model(self, name: Optional[str] = None) -> str:
            """
            Save the current model.
            
            Args:
                name: Optional custom name for the model
                
            Returns:
                Path where model was saved
            """
            if self.current_model is None:
                raise ValueError("No current model to save")
            
            if name is None:
                name = f"bayesian_lstm_{self.current_model.model_version}"
            
            filepath = f"{self.base_path}{name}"
            self.current_model.save_model(filepath)
            
            return filepath
        
        def load_model(self, filepath: str) -> BayesianLSTM:
            """
            Load a model from file.
            
            Args:
                filepath: Path to the model file
                
            Returns:
                Loaded BayesianLSTM instance
            """
            # Create temporary model to load into
            temp_config = {'sequence_length': 60}  # Will be updated from metadata
            model = BayesianLSTM(temp_config)
            
            if model.load_model(filepath):
                self.current_model = model
                self.model_registry[model.model_version] = model
                logger.info(f"Loaded model: {model.model_version}")
                return model
            else:
                raise ValueError(f"Failed to load model from {filepath}")
        
        def get_model_list(self) -> List[Dict[str, Any]]:
            """
            Get list of all registered models.
            
            Returns:
                List of model information dictionaries
            """
            model_list = []
            for version, model in self.model_registry.items():
                summary = model.get_model_summary()
                model_list.append(summary)
            
            return model_list

# Conditional exports based on TensorFlow availability
if not TENSORFLOW_AVAILABLE:
    # If TensorFlow is not available, replace classes with ensemble versions
    logger.info("Exporting TensorFlow-free ensemble implementations")
    BayesianLSTM = EnsembleBayesianLSTM
    ModelManager = EnsembleModelManager
    
    # Also export the ensemble model directly
    __all__ = ['BayesianLSTM', 'ModelManager', 'EnsembleBayesianModel']
else:
    # TensorFlow is available, export normal classes
    __all__ = ['BayesianLSTM', 'ModelManager', 'MonteCarloDropout']