"""
Unit tests for Bayesian LSTM Model Implementation.
Tests all core functionality including model architecture, training, uncertainty quantification.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import tensorflow as tf

# Import modules to test
import sys
project_root = '/Users/tomeldridge/telegram_bot'
sys.path = [project_root] + [p for p in sys.path if 'Momentum_dashboard' not in p]

from src.models.bayesian_lstm import BayesianLSTM, MonteCarloDropout, ModelManager
from src.models.uncertainty import UncertaintyQuantifier, ConfidenceScorer
from src.models.training import TrainingPipeline, AutoTrainer

class TestMonteCarloDropout:
    """Test Monte Carlo Dropout layer."""
    
    def test_monte_carlo_dropout_initialization(self):
        """Test MC dropout layer initialization."""
        dropout_rate = 0.3
        layer = MonteCarloDropout(rate=dropout_rate)
        
        assert layer.rate == dropout_rate
        
    def test_monte_carlo_dropout_config(self):
        """Test MC dropout layer configuration."""
        dropout_rate = 0.2
        layer = MonteCarloDropout(rate=dropout_rate)
        
        config = layer.get_config()
        assert config['rate'] == dropout_rate
        
    def test_monte_carlo_dropout_call(self):
        """Test MC dropout layer call method."""
        dropout_rate = 0.3
        layer = MonteCarloDropout(rate=dropout_rate)
        
        # Create test input
        inputs = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # Call layer
        output = layer(inputs)
        
        # Check output shape matches input
        assert output.shape == inputs.shape

class TestBayesianLSTM:
    """Test Bayesian LSTM model."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample model configuration."""
        return {
            'sequence_length': 60,
            'feature_count': 47,
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.2,
            'dense_dropout_rate': 0.3,
            'learning_rate': 0.001,
            'monte_carlo_samples': 100,
            'confidence_interval': 0.95
        }
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 60, 47)  # 100 samples, 60 timesteps, 47 features
        y = np.random.randn(100)  # 100 targets
        return X, y
    
    def test_bayesian_lstm_initialization(self, sample_config):
        """Test Bayesian LSTM initialization."""
        model = BayesianLSTM(sample_config)
        
        assert model.sequence_length == 60
        assert model.feature_count == 47
        assert model.lstm_units == [128, 64, 32]
        assert model.dropout_rate == 0.2
        assert model.monte_carlo_samples == 100
        assert model.model is None  # Not built yet
        
    def test_build_model(self, sample_config):
        """Test model building."""
        model = BayesianLSTM(sample_config)
        keras_model = model.build_model()
        
        # Check model was built
        assert model.model is not None
        assert keras_model is not None
        assert model.model_version is not None
        
        # Check model architecture
        assert len(keras_model.layers) >= 6  # Input + 3 LSTM + Dense + Dropout + Output
        
        # Check input shape
        expected_input_shape = (None, 60, 47)
        assert keras_model.input_shape == expected_input_shape
        
        # Check output shape
        expected_output_shape = (None, 1)
        assert keras_model.output_shape == expected_output_shape
        
    def test_model_summary(self, sample_config):
        """Test model summary generation."""
        model = BayesianLSTM(sample_config)
        model.build_model()
        
        summary = model.get_model_summary()
        
        assert 'model_version' in summary
        assert 'architecture' in summary
        assert 'configuration' in summary
        assert summary['architecture']['sequence_length'] == 60
        assert summary['architecture']['feature_count'] == 47
        assert summary['architecture']['lstm_layers'] == 3
        
    def test_train_model(self, sample_config, sample_data):
        """Test model training."""
        X, y = sample_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = BayesianLSTM(sample_config)
        
        # Train with minimal epochs for testing
        training_stats = model.train(
            X_train, y_train, X_val, y_val,
            epochs=2, batch_size=16, verbose=0
        )
        
        # Check training stats
        assert 'model_version' in training_stats
        assert 'training_time_seconds' in training_stats
        assert 'epochs_trained' in training_stats
        assert 'final_train_loss' in training_stats
        assert 'final_val_loss' in training_stats
        assert training_stats['epochs_trained'] <= 2
        
    def test_predict_with_uncertainty(self, sample_config, sample_data):
        """Test uncertainty prediction."""
        X, y = sample_data
        
        # Use smaller sample for faster testing
        X_test = X[:10]
        
        model = BayesianLSTM(sample_config)
        model.build_model()
        
        # Make predictions with reduced MC samples for testing
        results = model.predict_with_uncertainty(X_test, n_samples=10)
        
        # Check results structure
        assert 'predictions' in results
        assert 'uncertainties' in results
        assert 'confidence_lower' in results
        assert 'confidence_upper' in results
        assert 'all_samples' in results
        
        # Check shapes
        assert len(results['predictions']) == len(X_test)
        assert len(results['uncertainties']) == len(X_test)
        assert results['all_samples'].shape == (10, len(X_test))
        
        # Check uncertainty bounds
        assert np.all(results['confidence_lower'] <= results['predictions'])
        assert np.all(results['predictions'] <= results['confidence_upper'])
        
    def test_predict_single(self, sample_config, sample_data):
        """Test single prediction."""
        X, _ = sample_data
        
        model = BayesianLSTM(sample_config)
        model.build_model()
        
        # Single prediction
        result = model.predict_single(X[0])
        
        # Check result structure
        assert 'prediction' in result
        assert 'uncertainty' in result
        assert 'confidence_lower' in result
        assert 'confidence_upper' in result
        assert 'timestamp' in result
        
        # Check types
        assert isinstance(result['prediction'], float)
        assert isinstance(result['uncertainty'], float)
        assert result['uncertainty'] >= 0
        
    def test_evaluate_model(self, sample_config, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        model = BayesianLSTM(sample_config)
        model.build_model()
        
        # Quick training for evaluation
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model.train(X_train, y_train, X_test, y_test, epochs=1, verbose=0)
        
        # Evaluate model
        metrics = model.evaluate_model(X_test, y_test)
        
        # Check metrics
        required_metrics = [
            'test_loss', 'test_mae', 'test_mse', 'mse', 'mae', 'rmse',
            'directional_accuracy', 'mean_uncertainty', 'interval_coverage'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check reasonable values
        assert metrics['directional_accuracy'] >= 0
        assert metrics['directional_accuracy'] <= 1
        assert metrics['interval_coverage'] >= 0
        assert metrics['interval_coverage'] <= 1
        
    def test_save_load_model(self, sample_config, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        
        # Train a model
        model = BayesianLSTM(sample_config)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model.train(X_train, y_train, X_val, y_val, epochs=1, verbose=0)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model')
            model.save_model(model_path)
            
            # Check files exist
            assert os.path.exists(f"{model_path}.keras")
            assert os.path.exists(f"{model_path}_metadata.json")
            
            # Load model
            new_model = BayesianLSTM(sample_config)
            success = new_model.load_model(model_path)
            
            assert success
            assert new_model.model is not None
            assert new_model.model_version == model.model_version
            
            # Test loaded model can make predictions
            predictions = new_model.predict_with_uncertainty(X_val[:5], n_samples=5)
            assert len(predictions['predictions']) == 5
            
    def test_check_model_degradation(self, sample_config):
        """Test model degradation detection."""
        model = BayesianLSTM(sample_config)
        
        # Test without baseline metrics
        current_metrics = {'mae': 0.1, 'directional_accuracy': 0.6}
        analysis = model.check_model_degradation(current_metrics)
        
        assert 'degradation_detected' in analysis
        assert analysis['degradation_detected'] == False  # No baseline
        
        # Set baseline metrics
        model.training_stats = {
            'final_val_mae': 0.05,
            'directional_accuracy': 0.7
        }
        
        # Test with degraded performance
        degraded_metrics = {'mae': 0.15, 'directional_accuracy': 0.5}
        analysis = model.check_model_degradation(degraded_metrics)
        
        assert analysis['degradation_detected'] == True
        assert analysis['mae_increase_pct'] > 0
        assert analysis['recommendation'] == 'retrain'

class TestModelManager:
    """Test Model Manager functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample model configuration."""
        return {
            'sequence_length': 60,
            'feature_count': 47,
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'monte_carlo_samples': 50
        }
    
    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(base_path=temp_dir)
            
            assert manager.base_path == temp_dir
            assert manager.current_model is None
            assert len(manager.model_registry) == 0
            
    def test_create_model(self, sample_config):
        """Test model creation through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(base_path=temp_dir)
            
            model = manager.create_model(sample_config)
            
            assert model is not None
            assert manager.current_model == model
            assert model.model_version in manager.model_registry
            
    def test_save_load_model(self, sample_config):
        """Test model save/load through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(base_path=temp_dir)
            
            # Create and save model
            model = manager.create_model(sample_config)
            model_path = manager.save_current_model("test_model")
            
            assert os.path.exists(f"{model_path}.keras")
            
            # Load model
            loaded_model = manager.load_model(model_path)
            
            assert loaded_model is not None
            assert loaded_model.model_version == model.model_version
            assert manager.current_model == loaded_model
            
    def test_get_model_list(self, sample_config):
        """Test getting model list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(base_path=temp_dir)
            
            # Create multiple models
            model1 = manager.create_model(sample_config)
            model2 = manager.create_model(sample_config)
            
            model_list = manager.get_model_list()
            
            assert len(model_list) == 2
            assert all('model_version' in model for model in model_list)
            assert all('architecture' in model for model in model_list)

class TestUncertaintyQuantifier:
    """Test Uncertainty Quantifier."""
    
    @pytest.fixture
    def sample_mc_data(self):
        """Sample Monte Carlo predictions."""
        np.random.seed(42)
        # 50 MC samples, 20 predictions
        mc_samples = np.random.randn(50, 20) + np.linspace(-1, 1, 20)
        true_values = np.linspace(-1, 1, 20) + np.random.randn(20) * 0.1
        return mc_samples, true_values
    
    def test_uncertainty_quantifier_initialization(self):
        """Test UncertaintyQuantifier initialization."""
        uq = UncertaintyQuantifier(confidence_level=0.95)
        
        assert uq.confidence_level == 0.95
        assert uq.alpha == 0.05
        
    def test_analyze_mc_samples(self, sample_mc_data):
        """Test Monte Carlo sample analysis."""
        mc_samples, _ = sample_mc_data
        
        uq = UncertaintyQuantifier()
        analysis = uq.analyze_mc_samples(mc_samples)
        
        # Check required keys
        required_keys = [
            'n_samples', 'n_predictions', 'mean_predictions',
            'std_predictions', 'confidence_lower', 'confidence_upper',
            'interval_width', 'skewness', 'kurtosis'
        ]
        
        for key in required_keys:
            assert key in analysis
        
        # Check shapes
        assert analysis['n_samples'] == 50
        assert analysis['n_predictions'] == 20
        assert len(analysis['mean_predictions']) == 20
        assert len(analysis['std_predictions']) == 20
        
        # Check confidence intervals
        assert np.all(analysis['confidence_lower'] <= analysis['mean_predictions'])
        assert np.all(analysis['mean_predictions'] <= analysis['confidence_upper'])
        
    def test_calculate_prediction_intervals(self, sample_mc_data):
        """Test prediction interval calculation."""
        mc_samples, _ = sample_mc_data
        
        uq = UncertaintyQuantifier()
        
        # Test percentile method
        intervals_pct = uq.calculate_prediction_intervals(mc_samples, 'percentile')
        
        assert 'lower_bound' in intervals_pct
        assert 'upper_bound' in intervals_pct
        assert 'width' in intervals_pct
        assert intervals_pct['method'] == 'percentile'
        
        # Test Gaussian method
        intervals_gauss = uq.calculate_prediction_intervals(mc_samples, 'gaussian')
        
        assert intervals_gauss['method'] == 'gaussian'
        assert len(intervals_gauss['lower_bound']) == 20
        
    def test_assess_calibration(self, sample_mc_data):
        """Test calibration assessment."""
        mc_samples, true_values = sample_mc_data
        
        uq = UncertaintyQuantifier()
        
        # Get predictions and uncertainties
        predictions = np.mean(mc_samples, axis=0)
        uncertainties = np.std(mc_samples, axis=0)
        
        calibration = uq.assess_calibration(predictions, uncertainties, true_values)
        
        # Check required keys
        required_keys = [
            'bins', 'expected_coverage', 'observed_coverage',
            'calibration_error', 'reliability'
        ]
        
        for key in required_keys:
            assert key in calibration
        
        # Check value ranges
        assert 0 <= calibration['reliability'] <= 1
        assert 0 <= calibration['calibration_error'] <= 1
        
    def test_decompose_uncertainty(self, sample_mc_data):
        """Test uncertainty decomposition."""
        mc_samples, true_values = sample_mc_data
        
        uq = UncertaintyQuantifier()
        decomposition = uq.decompose_uncertainty(mc_samples, true_values)
        
        # Check required keys
        required_keys = [
            'epistemic_uncertainty', 'epistemic_std',
            'aleatoric_uncertainty', 'aleatoric_std',
            'total_uncertainty', 'total_std'
        ]
        
        for key in required_keys:
            assert key in decomposition
        
        # Check non-negative uncertainties
        assert np.all(decomposition['epistemic_uncertainty'] >= 0)
        assert np.all(decomposition['aleatoric_uncertainty'] >= 0)
        assert np.all(decomposition['total_uncertainty'] >= 0)
        
    def test_calculate_risk_metrics(self, sample_mc_data):
        """Test risk metrics calculation."""
        mc_samples, _ = sample_mc_data
        
        uq = UncertaintyQuantifier()
        
        predictions = np.mean(mc_samples, axis=0)
        uncertainties = np.std(mc_samples, axis=0)
        confidence_lower = np.percentile(mc_samples, 2.5, axis=0)
        confidence_upper = np.percentile(mc_samples, 97.5, axis=0)
        
        risk_metrics = uq.calculate_risk_metrics(
            predictions, uncertainties, confidence_lower, confidence_upper
        )
        
        # Check required keys
        required_keys = [
            'relative_uncertainty', 'interval_width_pct',
            'low_risk_count', 'medium_risk_count', 'high_risk_count',
            'tradeable_count', 'position_sizes'
        ]
        
        for key in required_keys:
            assert key in risk_metrics
        
        # Check counts sum correctly
        total_risk = (risk_metrics['low_risk_count'] + 
                     risk_metrics['medium_risk_count'] + 
                     risk_metrics['high_risk_count'])
        assert total_risk == 20
        
    def test_generate_uncertainty_report(self, sample_mc_data):
        """Test comprehensive uncertainty report generation."""
        mc_samples, true_values = sample_mc_data
        
        uq = UncertaintyQuantifier()
        report = uq.generate_uncertainty_report(mc_samples, true_values)
        
        # Check main sections
        required_sections = [
            'summary', 'monte_carlo_analysis', 'prediction_intervals',
            'uncertainty_decomposition', 'risk_metrics', 'calibration_assessment'
        ]
        
        for section in required_sections:
            assert section in report
        
        # Check summary
        summary = report['summary']
        assert 'n_predictions' in summary
        assert 'mean_uncertainty' in summary
        assert 'tradeable_signals_pct' in summary

class TestConfidenceScorer:
    """Test Confidence Scorer."""
    
    def test_confidence_scorer_initialization(self):
        """Test ConfidenceScorer initialization."""
        scorer = ConfidenceScorer(
            base_threshold=0.6,
            uncertainty_penalty=2.0,
            interval_width_penalty=1.5
        )
        
        assert scorer.base_threshold == 0.6
        assert scorer.uncertainty_penalty == 2.0
        assert scorer.interval_width_penalty == 1.5
        
    def test_calculate_confidence_score(self):
        """Test single confidence score calculation."""
        scorer = ConfidenceScorer()
        
        result = scorer.calculate_confidence_score(
            prediction=100.0,
            uncertainty=5.0,
            confidence_lower=90.0,
            confidence_upper=110.0
        )
        
        # Check required keys
        required_keys = [
            'confidence_score', 'relative_uncertainty', 'interval_width_pct',
            'uncertainty_factor', 'width_factor', 'recommendation', 'is_tradeable'
        ]
        
        for key in required_keys:
            assert key in result
        
        # Check value ranges
        assert 0 <= result['confidence_score'] <= 1
        assert result['relative_uncertainty'] >= 0
        assert result['recommendation'] in ['TRADE', 'MONITOR', 'HOLD']
        
    def test_batch_confidence_scores(self):
        """Test batch confidence score calculation."""
        scorer = ConfidenceScorer()
        
        # Sample data
        predictions = np.array([100.0, 200.0, 50.0])
        uncertainties = np.array([5.0, 20.0, 2.0])
        confidence_lower = np.array([90.0, 170.0, 47.0])
        confidence_upper = np.array([110.0, 230.0, 53.0])
        
        results = scorer.batch_confidence_scores(
            predictions, uncertainties, confidence_lower, confidence_upper
        )
        
        # Check required keys
        required_keys = [
            'confidence_scores', 'relative_uncertainties', 'recommendations',
            'tradeable_count', 'mean_confidence'
        ]
        
        for key in required_keys:
            assert key in results
        
        # Check shapes
        assert len(results['confidence_scores']) == 3
        assert len(results['recommendations']) == 3
        assert results['tradeable_count'] >= 0

class TestIntegration:
    """Integration tests for complete workflow."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for integration tests."""
        return {
            'sequence_length': 10,  # Small for testing
            'feature_count': 5,
            'lstm_units': [16, 8],  # Small architecture
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.2,
            'dense_dropout_rate': 0.3,
            'learning_rate': 0.01,
            'monte_carlo_samples': 10,  # Reduced for speed
            'confidence_interval': 0.95
        }
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for integration tests."""
        np.random.seed(42)
        X = np.random.randn(50, 10, 5)  # 50 samples, 10 timesteps, 5 features
        y = np.random.randn(50)
        return X, y
    
    def test_full_model_workflow(self, sample_config, sample_training_data):
        """Test complete model training and prediction workflow."""
        X, y = sample_training_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create and train model
        model = BayesianLSTM(sample_config)
        training_stats = model.train(
            X_train, y_train, X_val, y_val,
            epochs=2, batch_size=8, verbose=0
        )
        
        # Evaluate model
        evaluation_metrics = model.evaluate_model(X_val, y_val)
        
        # Generate predictions with uncertainty
        uncertainty_results = model.predict_with_uncertainty(X_val)
        
        # Analyze uncertainty
        uq = UncertaintyQuantifier()
        uncertainty_report = uq.generate_uncertainty_report(
            uncertainty_results['all_samples'], y_val
        )
        
        # Calculate confidence scores
        scorer = ConfidenceScorer()
        confidence_results = scorer.batch_confidence_scores(
            uncertainty_results['predictions'],
            uncertainty_results['uncertainties'],
            uncertainty_results['confidence_lower'],
            uncertainty_results['confidence_upper']
        )
        
        # Verify complete workflow
        assert training_stats['epochs_trained'] <= 2
        assert 'directional_accuracy' in evaluation_metrics
        assert 'summary' in uncertainty_report
        assert 'confidence_scores' in confidence_results
        
        # Check data consistency
        n_predictions = len(X_val)
        assert len(uncertainty_results['predictions']) == n_predictions
        assert len(confidence_results['confidence_scores']) == n_predictions
        assert uncertainty_report['summary']['n_predictions'] == n_predictions
        
    def test_model_persistence_workflow(self, sample_config, sample_training_data):
        """Test model save/load workflow."""
        X, y = sample_training_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train model
            model = BayesianLSTM(sample_config)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            model.train(X_train, y_train, X_val, y_val, epochs=1, verbose=0)
            
            # Get predictions from original model
            original_predictions = model.predict_with_uncertainty(X_val[:3])
            
            # Save model
            model_path = os.path.join(temp_dir, 'test_model')
            model.save_model(model_path)
            
            # Load model
            loaded_model = BayesianLSTM(sample_config)
            success = loaded_model.load_model(model_path)
            
            assert success
            
            # Get predictions from loaded model
            loaded_predictions = loaded_model.predict_with_uncertainty(X_val[:3])
            
            # Predictions should be similar (not exactly equal due to dropout)
            prediction_diff = np.abs(
                original_predictions['predictions'] - 
                loaded_predictions['predictions']
            )
            assert np.mean(prediction_diff) < 1.0  # Reasonable tolerance

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 