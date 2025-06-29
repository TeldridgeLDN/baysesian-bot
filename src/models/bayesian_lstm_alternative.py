"""
TensorFlow-free Bayesian LSTM Alternative for cryptocurrency price prediction.
Uses scikit-learn and numpy to provide similar functionality without TF dependencies.
Implements ensemble-based uncertainty quantification.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Quantile loss function for better tail distribution prediction.
    More robust to outliers than MSE, especially for volatile financial data.
    """
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

def pinball_loss(y_true, y_pred_lower, y_pred_upper, alpha=0.05):
    """
    Pinball loss for interval prediction evaluation.
    Used to evaluate confidence interval quality.
    """
    lower_loss = quantile_loss(y_true, y_pred_lower, alpha/2)
    upper_loss = quantile_loss(y_true, y_pred_upper, 1 - alpha/2)
    return (lower_loss + upper_loss) / 2

def add_gaussian_noise(data, noise_variance=0.1, random_state=42):
    """
    Add Gaussian noise to data for robustness training.
    Based on research findings with variance levels 0.1, 0.2, 0.3.
    """
    np.random.seed(random_state)
    noise = np.random.normal(0, np.sqrt(noise_variance), data.shape)
    return data + noise

def augment_training_data(X, y, noise_levels=[0.1, 0.2], random_state=42):
    """
    Augment training data with different noise levels for robustness.
    """
    X_augmented = [X]
    y_augmented = [y]
    
    for noise_var in noise_levels:
        X_noisy = add_gaussian_noise(X, noise_variance=noise_var, random_state=random_state)
        X_augmented.append(X_noisy)
        y_augmented.append(y)  # Target doesn't change
    
    return np.concatenate(X_augmented, axis=0), np.concatenate(y_augmented, axis=0)

class EnsembleBayesianModel:
    """
    Ensemble-based Bayesian model for cryptocurrency price prediction.
    Uses multiple neural networks and tree-based models for uncertainty quantification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ensemble Bayesian model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.models = []
        self.scalers = []
        self.model_version = None
        self.training_stats = {}
        self.is_trained = False
        
        # Extract configuration
        def get_config_value(key, default):
            if isinstance(config, dict):
                return config.get(key, default)
            else:
                return getattr(config, key, default)
        
        self.sequence_length = get_config_value('sequence_length', 21)  # Optimized based on research findings
        self.feature_count = get_config_value('feature_count', 47)
        self.n_models = get_config_value('ensemble_size', 10)  # Number of models in ensemble
        self.confidence_interval = get_config_value('confidence_interval', 0.95)
        self.random_state = get_config_value('random_state', 42)
        self.use_quantile_loss = get_config_value('use_quantile_loss', True)  # Enable quantile loss by default
        self.noise_augmentation = get_config_value('noise_augmentation', True)  # Enable noise augmentation
        self.noise_levels = get_config_value('noise_levels', [0.1, 0.2])  # Research-based noise levels
        
        # Model configurations
        self.mlp_configs = [
            {'hidden_layer_sizes': (128, 64, 32), 'alpha': 0.001, 'learning_rate_init': 0.001},
            {'hidden_layer_sizes': (100, 50), 'alpha': 0.01, 'learning_rate_init': 0.005},
            {'hidden_layer_sizes': (64, 32, 16), 'alpha': 0.005, 'learning_rate_init': 0.001},
            {'hidden_layer_sizes': (150, 75), 'alpha': 0.001, 'learning_rate_init': 0.002},
        ]
        
        self.tree_configs = [
            {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.1},
            {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.05},
            {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.08},
        ]
        
        logger.info(f"EnsembleBayesianModel initialized with {self.n_models} models")
        
    def _create_ensemble(self):
        """Create ensemble of diverse models."""
        models = []
        scalers = []
        
        # Add MLPs with different architectures
        for i, config in enumerate(self.mlp_configs):
            if len(models) >= self.n_models:
                break
                
            model = MLPRegressor(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                alpha=config['alpha'],
                learning_rate_init=config['learning_rate_init'],
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state + i,
                activation='relu',
                solver='adam'
            )
            scaler = StandardScaler()
            
            models.append(('mlp', model, scaler))
            scalers.append(scaler)
        
        # Add Gradient Boosting models
        for i, config in enumerate(self.tree_configs):
            if len(models) >= self.n_models:
                break
                
            model = GradientBoostingRegressor(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                learning_rate=config['learning_rate'],
                random_state=self.random_state + len(models),
                subsample=0.8,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
            scaler = MinMaxScaler()
            
            models.append(('gbr', model, scaler))
            scalers.append(scaler)
        
        # Add Random Forest models with different parameters
        remaining_slots = self.n_models - len(models)
        for i in range(remaining_slots):
            model = RandomForestRegressor(
                n_estimators=100 + i * 50,
                max_depth=8 + i * 2,
                random_state=self.random_state + len(models),
                n_jobs=-1,
                bootstrap=True,
                oob_score=True
            )
            scaler = StandardScaler()
            
            models.append(('rf', model, scaler))
            scalers.append(scaler)
        
        self.models = models[:self.n_models]
        self.scalers = scalers[:self.n_models]
        
        logger.info(f"Created ensemble with {len(self.models)} models")
        
    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert sequences to flattened features for sklearn models."""
        if len(X.shape) == 3:
            # Reshape from (samples, sequence_length, features) to (samples, sequence_length * features)
            return X.reshape(X.shape[0], -1)
        return X
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray, 
              y_val: np.ndarray,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Verbosity level
            
        Returns:
            Training statistics dictionary
        """
        logger.info(f"Starting ensemble training: {len(X_train)} train, {len(X_val)} val samples")
        
        # Create ensemble
        self._create_ensemble()
        
        # Prepare data
        X_train_flat = self._prepare_sequences(X_train)
        X_val_flat = self._prepare_sequences(X_val)
        
        # Apply noise augmentation if enabled
        if self.noise_augmentation:
            logger.info(f"Applying noise augmentation with levels: {self.noise_levels}")
            X_train_flat, y_train = augment_training_data(
                X_train_flat, y_train, 
                noise_levels=self.noise_levels, 
                random_state=self.random_state
            )
            logger.info(f"Training data augmented: {X_train_flat.shape[0]} samples")
        
        start_time = datetime.now()
        training_scores = []
        validation_scores = []
        
        # Train each model in the ensemble
        for i, (model_type, model, scaler) in enumerate(self.models):
            try:
                if verbose:
                    logger.info(f"Training model {i+1}/{len(self.models)} ({model_type})")
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train_flat)
                X_val_scaled = scaler.transform(X_val_flat)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_pred = model.predict(X_train_scaled)
                val_pred = model.predict(X_val_scaled)
                
                train_mse = mean_squared_error(y_train, train_pred)
                val_mse = mean_squared_error(y_val, val_pred)
                
                training_scores.append(train_mse)
                validation_scores.append(val_mse)
                
                if verbose:
                    logger.info(f"  Model {i+1} - Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
                
            except Exception as e:
                logger.error(f"Failed to train model {i+1}: {str(e)}")
                # Remove failed model
                self.models[i] = None
        
        # Remove failed models
        self.models = [m for m in self.models if m is not None]
        
        training_time = datetime.now() - start_time
        
        # Calculate ensemble performance manually without using predict method
        X_val_flat = self._prepare_sequences(X_val)
        ensemble_predictions = []
        
        for model_type, model, scaler in self.models:
            try:
                X_val_scaled = scaler.transform(X_val_flat)
                pred = model.predict(X_val_scaled)
                ensemble_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed during training evaluation: {str(e)}")
                continue
        
        if ensemble_predictions:
            ensemble_val_pred = np.mean(ensemble_predictions, axis=0)
            ensemble_val_mse = mean_squared_error(y_val, ensemble_val_pred)
            ensemble_val_mae = mean_absolute_error(y_val, ensemble_val_pred)
        else:
            ensemble_val_mse = float('inf')
            ensemble_val_mae = float('inf')
        
        # Training statistics
        self.training_stats = {
            'model_version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_time_seconds': training_time.total_seconds(),
            'n_models_trained': len(self.models),
            'ensemble_val_mse': float(ensemble_val_mse),
            'ensemble_val_mae': float(ensemble_val_mae),
            'individual_train_mse': [float(s) for s in training_scores],
            'individual_val_mse': [float(s) for s in validation_scores],
            'mean_individual_val_mse': float(np.mean(validation_scores)),
            'std_individual_val_mse': float(np.std(validation_scores)),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'config': self.config.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.model_version = self.training_stats['model_version']
        self.is_trained = True
        
        logger.info(f"Ensemble training completed in {training_time}")
        logger.info(f"Ensemble validation MSE: {ensemble_val_mse:.6f}")
        logger.info(f"Ensemble validation MAE: {ensemble_val_mae:.6f}")
        
        return self.training_stats
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty quantification using ensemble disagreement.
        
        Args:
            X: Input sequences
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        if not self.is_trained or not self.models:
            raise ValueError("Model must be trained before making predictions")
        
        X_flat = self._prepare_sequences(X)
        predictions_ensemble = []
        
        # Get predictions from each model
        for model_type, model, scaler in self.models:
            try:
                X_scaled = scaler.transform(X_flat)
                pred = model.predict(X_scaled)
                predictions_ensemble.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed: {str(e)}")
                continue
        
        if not predictions_ensemble:
            raise ValueError("No models available for prediction")
        
        # Convert to numpy array: (n_models, n_predictions)
        predictions_ensemble = np.array(predictions_ensemble)
        
        # Calculate ensemble statistics
        predictions = np.mean(predictions_ensemble, axis=0)
        uncertainties = np.std(predictions_ensemble, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_lower = np.percentile(predictions_ensemble, lower_percentile, axis=0)
        confidence_upper = np.percentile(predictions_ensemble, upper_percentile, axis=0)
        
        # Additional uncertainty metrics
        prediction_variance = np.var(predictions_ensemble, axis=0)
        epistemic_uncertainty = uncertainties  # Model disagreement
        
        results = {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'prediction_variance': prediction_variance,
            'epistemic_uncertainty': epistemic_uncertainty,
            'all_predictions': predictions_ensemble,
            'confidence_interval': self.confidence_interval,
            'n_models': len(predictions_ensemble)
        }
        
        logger.info(f"Predictions generated: mean uncertainty = {np.mean(uncertainties):.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Simple prediction without full uncertainty analysis."""
        return self.predict_with_uncertainty(X)
    
    def predict_single(self, X: np.ndarray) -> Dict[str, float]:
        """
        Make a single prediction with uncertainty.
        
        Args:
            X: Single input sequence
            
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
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating ensemble on {len(X_test)} test samples")
        
        # Get predictions with uncertainty
        pred_results = self.predict_with_uncertainty(X_test)
        predictions = pred_results['predictions']
        uncertainties = pred_results['uncertainties']
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        # Quantile loss (better for financial volatility)
        q_loss = quantile_loss(y_test, predictions, quantile=0.5)
        
        # Pinball loss for interval evaluation
        pinball = pinball_loss(y_test, pred_results['confidence_lower'], 
                              pred_results['confidence_upper'], alpha=1-self.confidence_interval)
        
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
        
        # Interval width and quality metrics
        interval_width = pred_results['confidence_upper'] - pred_results['confidence_lower']
        mean_interval_width = np.mean(interval_width)
        normalized_interval_width = mean_interval_width / np.mean(np.abs(y_test))  # Relative to price scale
        
        # Interval efficiency (narrower intervals are better if coverage is maintained)
        interval_efficiency = interval_coverage / normalized_interval_width if normalized_interval_width > 0 else 0
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'quantile_loss': float(q_loss),
            'pinball_loss': float(pinball),
            'directional_accuracy': float(directional_accuracy),
            'mean_uncertainty': float(mean_uncertainty),
            'uncertainty_std': float(uncertainty_std),
            'interval_coverage': float(interval_coverage),
            'mean_interval_width': float(mean_interval_width),
            'normalized_interval_width': float(normalized_interval_width),
            'interval_efficiency': float(interval_efficiency),
            'expected_coverage': self.confidence_interval,
            'n_test_samples': len(X_test),
            'model_version': self.model_version,
            'n_models_used': pred_results['n_models']
        }
        
        logger.info(f"Enhanced Bayesian evaluation completed:")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Quantile Loss: {q_loss:.6f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.3f}")
        logger.info(f"  Interval Coverage: {interval_coverage:.3f} (expected: {self.confidence_interval:.3f})")
        logger.info(f"  Interval Efficiency: {interval_efficiency:.3f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained ensemble model.
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Save models and scalers
        model_data = {
            'models': [],
            'scalers': [],
            'model_types': []
        }
        
        for model_type, model, scaler in self.models:
            model_data['models'].append(model)
            model_data['scalers'].append(scaler)
            model_data['model_types'].append(model_type)
        
        model_path = f"{filepath}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        metadata = {
            'model_version': self.model_version,
            'config': self.config,
            'training_stats': self.training_stats,
            'model_architecture': {
                'sequence_length': self.sequence_length,
                'feature_count': self.feature_count,
                'n_models': len(self.models),
                'ensemble_size': self.n_models
            },
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Ensemble model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a saved ensemble model.
        
        Args:
            filepath: Path to the saved model (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load models
            model_path = f"{filepath}.pkl"
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Reconstruct models list
            self.models = []
            for i, (model, scaler, model_type) in enumerate(zip(
                model_data['models'], 
                model_data['scalers'], 
                model_data['model_types']
            )):
                self.models.append((model_type, model, scaler))
            
            # Load metadata
            metadata_path = f"{filepath}_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.model_version = metadata['model_version']
            self.config.update(metadata['config'])
            self.training_stats = metadata.get('training_stats', {})
            
            # Update configuration
            arch = metadata.get('model_architecture', {})
            self.sequence_length = arch.get('sequence_length', self.sequence_length)
            self.feature_count = arch.get('feature_count', self.feature_count)
            self.n_models = arch.get('n_models', len(self.models))
            
            self.is_trained = True
            
            logger.info(f"Ensemble model loaded successfully: {self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensemble model: {str(e)}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        if not self.is_trained:
            return {'error': 'No trained model available'}
        
        summary = {
            'model_version': self.model_version,
            'model_type': 'EnsembleBayesian',
            'architecture': {
                'n_models': len(self.models),
                'sequence_length': self.sequence_length,
                'feature_count': self.feature_count,
                'model_types': [model_type for model_type, _, _ in self.models]
            },
            'configuration': {
                'ensemble_size': self.n_models,
                'confidence_interval': self.confidence_interval,
                'random_state': self.random_state
            },
            'training_stats': self.training_stats,
            'created_at': datetime.now().isoformat()
        }
        
        return summary

# Backwards compatibility adapter
class BayesianLSTM:
    """Adapter class to maintain compatibility with existing code."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with EnsembleBayesianModel backend."""
        self.backend = EnsembleBayesianModel(config)
        self.config = config
        self.model = None  # For compatibility
        self.history = None  # For compatibility
        self.model_version = None
        self.training_stats = {}
    
    def build_model(self):
        """Build model (no-op for compatibility)."""
        self.model = "ensemble_backend"  # Dummy value
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        """Train the model."""
        stats = self.backend.train(X_train, y_train, X_val, y_val, kwargs.get('verbose', 1))
        self.training_stats = stats
        self.model_version = stats['model_version']
        return stats
    
    def predict_with_uncertainty(self, X, n_samples=None):
        """Predict with uncertainty."""
        return self.backend.predict_with_uncertainty(X)
    
    def predict_single(self, X):
        """Single prediction."""
        return self.backend.predict_single(X)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model."""
        return self.backend.evaluate_model(X_test, y_test)
    
    def save_model(self, filepath, include_history=True):
        """Save model."""
        self.backend.save_model(filepath)
    
    def load_model(self, filepath):
        """Load model."""
        success = self.backend.load_model(filepath)
        if success:
            self.model_version = self.backend.model_version
            self.training_stats = self.backend.training_stats
        return success
    
    def get_model_summary(self):
        """Get model summary."""
        return self.backend.get_model_summary()
    
    def check_model_degradation(self, current_metrics, **kwargs):
        """Check model degradation."""
        if not self.training_stats:
            return {'degradation_detected': False, 'reason': 'No baseline metrics available'}
        
        baseline_mae = self.training_stats.get('ensemble_val_mae', float('inf'))
        current_mae = current_metrics.get('mae', 0)
        
        mae_increase = (current_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0
        degradation_detected = mae_increase > kwargs.get('threshold_mae_increase', 0.2)
        
        return {
            'degradation_detected': degradation_detected,
            'mae_increase_pct': mae_increase * 100,
            'baseline_mae': baseline_mae,
            'current_mae': current_mae,
            'recommendation': 'retrain' if degradation_detected else 'continue',
            'analysis_timestamp': datetime.now().isoformat()
        }

class ModelManager:
    """Model manager for ensemble models."""
    
    def __init__(self, base_path: str = "models/"):
        self.base_path = base_path
        self.current_model = None
        self.model_registry = {}
        
        import os
        os.makedirs(base_path, exist_ok=True)
    
    def create_model(self, config: Dict[str, Any]) -> BayesianLSTM:
        """Create new model."""
        model = BayesianLSTM(config)
        model.build_model()
        
        self.current_model = model
        return model
    
    def save_current_model(self, name: Optional[str] = None) -> str:
        """Save current model."""
        if self.current_model is None:
            raise ValueError("No current model to save")
        
        if name is None:
            name = f"ensemble_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = f"{self.base_path}{name}"
        self.current_model.save_model(filepath)
        
        return filepath
    
    def load_model(self, filepath: str) -> BayesianLSTM:
        """Load model."""
        temp_config = {'sequence_length': 60}
        model = BayesianLSTM(temp_config)
        
        if model.load_model(filepath):
            self.current_model = model
            return model
        else:
            raise ValueError(f"Failed to load model from {filepath}")
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """Get model list."""
        return [self.current_model.get_model_summary()] if self.current_model else []