"""
Unit tests for configuration validation system.
Tests all configuration parameters, validation logic, and safety bounds.
"""

import pytest
import tempfile
import yaml
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.config import (
    ConfigManager, TradingConfig, ModelConfig, DataConfig,
    TradingConfigModel, ModelConfigModel, DataConfigModel,
    TelegramConfigModel, DatabaseConfigModel, APIConfigModel,
    LoggingConfigModel, SecurityConfigModel, MonitoringConfigModel
)
from pydantic import ValidationError

class TestTradingConfigValidation:
    """Test trading configuration validation."""
    
    def test_valid_trading_config(self):
        """Test valid trading configuration passes validation."""
        valid_config = {
            'max_position_size': 0.10,
            'confidence_threshold': 0.45,
            'stop_loss_multiplier': 2.0,
            'take_profit_ratio': 1.5,
            'max_drawdown': 0.15,
            'position_timeout_hours': 24,
            'min_price_change_pct': 1.2,
            'max_interval_width_pct': 20.0,
            'max_active_positions_pct': 0.30,
            'paper_trading': True,
            'historical_days': 180,
            'trading_fee': 0.001,
            'slippage': 0.0002,
            'risk_grading_enabled': True,
            'adaptive_parameters': False
        }
        
        # Should not raise any exceptions
        trading_model = TradingConfigModel(**valid_config)
        assert trading_model.confidence_threshold == 0.45
        assert trading_model.max_position_size == 0.10
    
    def test_invalid_position_size(self):
        """Test invalid position size raises validation error."""
        invalid_configs = [
            {'max_position_size': 0.0},  # Too low
            {'max_position_size': 1.5},  # Too high (>100%)
            {'max_position_size': -0.1}, # Negative
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValidationError):
                TradingConfigModel(**config)
    
    def test_confidence_threshold_bounds(self):
        """Test confidence threshold validation bounds."""
        # Valid values
        valid_values = [0.25, 0.45, 0.65, 1.0]
        for value in valid_values:
            config = TradingConfigModel(confidence_threshold=value)
            assert config.confidence_threshold == value
        
        # Invalid values
        invalid_values = [0.0, 1.1, -0.1]
        for value in invalid_values:
            with pytest.raises(ValidationError):
                TradingConfigModel(confidence_threshold=value)
    
    def test_price_change_validation(self):
        """Test minimum price change percentage validation."""
        # Valid values
        valid_values = [0.5, 1.0, 1.5, 5.0]
        for value in valid_values:
            config = TradingConfigModel(min_price_change_pct=value)
            assert config.min_price_change_pct == value
        
        # Invalid values
        invalid_values = [0.0, 5.1, -1.0]
        for value in invalid_values:
            with pytest.raises(ValidationError):
                TradingConfigModel(min_price_change_pct=value)
    
    def test_uncertainty_tolerance_validation(self):
        """Test uncertainty tolerance validation."""
        # Valid values
        valid_values = [15.0, 20.0, 25.0, 30.0]
        for value in valid_values:
            config = TradingConfigModel(max_interval_width_pct=value)
            assert config.max_interval_width_pct == value
        
        # Invalid values
        invalid_values = [0.0, 30.1, -5.0]
        for value in invalid_values:
            with pytest.raises(ValidationError):
                TradingConfigModel(max_interval_width_pct=value)
    
    def test_drawdown_limits(self):
        """Test maximum drawdown validation."""
        # Valid values
        valid_values = [0.05, 0.10, 0.15, 0.25]
        for value in valid_values:
            config = TradingConfigModel(max_drawdown=value)
            assert config.max_drawdown == value
        
        # Invalid values
        invalid_values = [0.0, 0.26, -0.1]
        for value in invalid_values:
            with pytest.raises(ValidationError):
                TradingConfigModel(max_drawdown=value)
    
    def test_position_size_warning(self):
        """Test position size warning for high values."""
        with patch('utils.config.logger') as mock_logger:
            # High but valid position size should trigger warning
            config = TradingConfigModel(max_position_size=0.20)
            mock_logger.warning.assert_called_once()
            assert "exceeds recommended 15% limit" in mock_logger.warning.call_args[0][0]

class TestModelConfigValidation:
    """Test model configuration validation."""
    
    def test_valid_model_config(self):
        """Test valid model configuration."""
        valid_config = {
            'sequence_length': 60,
            'retrain_frequency_hours': 24,
            'monte_carlo_samples': 100,
            'confidence_interval': 0.95,
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.2,
            'dense_dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.2
        }
        
        model = ModelConfigModel(**valid_config)
        assert model.sequence_length == 60
        assert model.lstm_units == [128, 64, 32]
    
    def test_lstm_units_validation(self):
        """Test LSTM units validation."""
        # Valid configurations
        valid_configs = [
            {'lstm_units': [32]},
            {'lstm_units': [64, 32]},
            {'lstm_units': [128, 64, 32]},
        ]
        
        for config in valid_configs:
            model = ModelConfigModel(**config)
            assert len(model.lstm_units) >= 1
        
        # Invalid configurations
        with pytest.raises(ValidationError):
            ModelConfigModel(lstm_units=[])  # Empty list
        
        with pytest.raises(ValidationError):
            ModelConfigModel(lstm_units=[0, 32])  # Zero units
        
        with pytest.raises(ValidationError):
            ModelConfigModel(lstm_units=[-10, 32])  # Negative units
    
    def test_monte_carlo_samples_bounds(self):
        """Test Monte Carlo samples validation."""
        # Valid values
        valid_values = [10, 50, 100, 500, 1000]
        for value in valid_values:
            config = ModelConfigModel(monte_carlo_samples=value)
            assert config.monte_carlo_samples == value
        
        # Invalid values
        invalid_values = [0, 1001, -50]
        for value in invalid_values:
            with pytest.raises(ValidationError):
                ModelConfigModel(monte_carlo_samples=value)
    
    def test_dropout_rate_validation(self):
        """Test dropout rate validation."""
        # Valid values
        valid_values = [0.0, 0.1, 0.3, 0.5, 1.0]
        for rate_type in ['dropout_rate', 'recurrent_dropout_rate', 'dense_dropout_rate']:
            for value in valid_values:
                config = ModelConfigModel(**{rate_type: value})
                assert getattr(config, rate_type) == value
        
        # Invalid values
        invalid_values = [-0.1, 1.1]
        for rate_type in ['dropout_rate', 'recurrent_dropout_rate', 'dense_dropout_rate']:
            for value in invalid_values:
                with pytest.raises(ValidationError):
                    ModelConfigModel(**{rate_type: value})

class TestDataConfigValidation:
    """Test data configuration validation."""
    
    def test_valid_data_config(self):
        """Test valid data configuration."""
        valid_config = {
            'historical_days': 180,
            'update_frequency_minutes': 60,
            'backup_retention_days': 30,
            'price_data_sources': {'primary': 'coingecko', 'backup': 'binance'},
            'features': ['open', 'high', 'low', 'close', 'volume'],
            'data_validation': {'max_price_change_pct': 20.0, 'min_volume_threshold': 1000}
        }
        
        config = DataConfigModel(**valid_config)
        assert config.historical_days == 180
        assert 'close' in config.features
    
    def test_historical_days_minimum(self):
        """Test historical days minimum requirement."""
        # Valid values
        valid_values = [30, 60, 180, 365]
        for value in valid_values:
            config = DataConfigModel(
                features=['open', 'high', 'low', 'close', 'volume'],
                historical_days=value
            )
            assert config.historical_days == value
        
        # Invalid values
        invalid_values = [0, 15, 29]
        for value in invalid_values:
            with pytest.raises(ValidationError):
                DataConfigModel(
                    features=['open', 'high', 'low', 'close', 'volume'],
                    historical_days=value
                )
    
    def test_required_features_validation(self):
        """Test required features validation."""
        required_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Valid - has all required features
        valid_config = DataConfigModel(features=required_features + ['rsi', 'macd'])
        assert all(feature in valid_config.features for feature in required_features)
        
        # Invalid - missing required features
        for missing_feature in required_features:
            incomplete_features = [f for f in required_features if f != missing_feature]
            with pytest.raises(ValidationError):
                DataConfigModel(features=incomplete_features)

class TestRuntimeConstraintValidation:
    """Test runtime constraint validation logic."""
    
    def create_test_config_file(self, config_dict: Dict[str, Any]) -> str:
        """Create a temporary config file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name
    
    def test_valid_runtime_constraints(self):
        """Test valid runtime constraints pass validation."""
        config_dict = {
            'trading': {
                'max_position_size': 0.10,
                'max_active_positions_pct': 0.30,
                'confidence_threshold': 0.45,
                'min_price_change_pct': 1.2,
                'max_interval_width_pct': 20.0
            },
            'model': {
                'sequence_length': 60,
                'validation_split': 0.2,
                'lstm_units': [128, 64, 32]
            },
            'data': {
                'historical_days': 180,
                'features': ['open', 'high', 'low', 'close', 'volume']
            },
            'database': {
                'db_path': 'test.db',
                'backup_path': 'test_backups/'
            },
            'logging': {
                'log_file': 'test.log'
            }
        }
        
        config_file = self.create_test_config_file(config_dict)
        try:
            config_manager = ConfigManager(config_file)
            assert config_manager.validate_config()
        finally:
            os.unlink(config_file)
    
    def test_position_size_exceeds_active_positions(self):
        """Test validation fails when position size exceeds active positions limit."""
        config_dict = {
            'trading': {
                'max_position_size': 0.40,  # 40%
                'max_active_positions_pct': 0.30,  # 30% - invalid!
                'confidence_threshold': 0.45
            },
            'model': {'sequence_length': 60, 'lstm_units': [32]},
            'data': {'historical_days': 30, 'features': ['open', 'high', 'low', 'close', 'volume']},
            'database': {'db_path': 'test.db', 'backup_path': 'test_backups/'},
            'logging': {'log_file': 'test.log'}
        }
        
        config_file = self.create_test_config_file(config_dict)
        try:
            config_manager = ConfigManager(config_file)
            assert not config_manager.validate_config()
        finally:
            os.unlink(config_file)
    
    def test_total_risk_exposure_limit(self):
        """Test total risk exposure validation."""
        config_dict = {
            'trading': {
                'max_position_size': 0.30,  # 30%
                'max_active_positions_pct': 0.80,  # 80% - results in 24% exposure, valid
                'confidence_threshold': 0.45
            },
            'model': {'sequence_length': 60, 'lstm_units': [32]},
            'data': {'historical_days': 30, 'features': ['open', 'high', 'low', 'close', 'volume']},
            'database': {'db_path': 'test.db', 'backup_path': 'test_backups/'},
            'logging': {'log_file': 'test.log'}
        }
        
        config_file = self.create_test_config_file(config_dict)
        try:
            config_manager = ConfigManager(config_file)
            assert config_manager.validate_config()  # Should pass (24% < 50%)
        finally:
            os.unlink(config_file)
        
        # Test excessive exposure
        config_dict['trading']['max_position_size'] = 0.70  # 70% * 80% = 56% exposure - invalid!
        config_file = self.create_test_config_file(config_dict)
        try:
            config_manager = ConfigManager(config_file)
            assert not config_manager.validate_config()
        finally:
            os.unlink(config_file)
    
    def test_confidence_threshold_warnings(self):
        """Test confidence threshold warning triggers."""
        with patch('utils.config.logger') as mock_logger:
            config_dict = {
                'trading': {
                    'confidence_threshold': 0.20,  # Very low - should trigger warning
                    'max_position_size': 0.10,
                    'max_active_positions_pct': 0.30
                },
                'model': {'sequence_length': 60, 'lstm_units': [32]},
                'data': {'historical_days': 30, 'features': ['open', 'high', 'low', 'close', 'volume']},
                'database': {'db_path': 'test.db', 'backup_path': 'test_backups/'},
                'logging': {'log_file': 'test.log'}
            }
            
            config_file = self.create_test_config_file(config_dict)
            try:
                config_manager = ConfigManager(config_file)
                config_manager.validate_config()
                
                # Check warning was logged
                warning_calls = [call for call in mock_logger.warning.call_args_list 
                               if 'Very low confidence threshold' in str(call)]
                assert len(warning_calls) > 0
            finally:
                os.unlink(config_file)
    
    def test_model_data_consistency(self):
        """Test model and data configuration consistency."""
        # Invalid: historical_days < sequence_length
        config_dict = {
            'trading': {'confidence_threshold': 0.45, 'max_position_size': 0.10, 'max_active_positions_pct': 0.30},
            'model': {'sequence_length': 100, 'lstm_units': [32]},  # Needs 100 days
            'data': {'historical_days': 60, 'features': ['open', 'high', 'low', 'close', 'volume']},  # Only 60 days
            'database': {'db_path': 'test.db', 'backup_path': 'test_backups/'},
            'logging': {'log_file': 'test.log'}
        }
        
        config_file = self.create_test_config_file(config_dict)
        try:
            config_manager = ConfigManager(config_file)
            assert not config_manager.validate_config()
        finally:
            os.unlink(config_file)

class TestEnvironmentVariableOverrides:
    """Test environment variable configuration overrides."""
    
    def test_trading_parameter_overrides(self):
        """Test trading parameter environment variable overrides."""
        config_dict = {
            'trading': {
                'confidence_threshold': 0.45,
                'max_position_size': 0.10,
                'paper_trading': False
            },
            'model': {'sequence_length': 60, 'lstm_units': [32]},
            'data': {'historical_days': 30, 'features': ['open', 'high', 'low', 'close', 'volume']},
            'database': {'db_path': 'test.db', 'backup_path': 'test_backups/'},
            'logging': {'log_file': 'test.log'}
        }
        
        config_file = self.create_test_config_file(config_dict)
        
        try:
            # Set environment variables
            with patch.dict(os.environ, {
                'CONFIDENCE_THRESHOLD': '0.55',
                'MAX_POSITION_SIZE': '0.15',
                'PAPER_TRADING': 'true'
            }):
                config_manager = ConfigManager(config_file)
                trading_config = config_manager.get_trading_config()
                
                # Check overrides were applied
                assert trading_config.confidence_threshold == 0.55
                assert trading_config.max_position_size == 0.15
                assert trading_config.paper_trading == True
                
        finally:
            os.unlink(config_file)
    
    def test_boolean_environment_conversions(self):
        """Test boolean environment variable conversions."""
        config_dict = {
            'trading': {'paper_trading': False, 'confidence_threshold': 0.45, 'max_position_size': 0.10, 'max_active_positions_pct': 0.30},
            'model': {'sequence_length': 60, 'lstm_units': [32]},
            'data': {'historical_days': 30, 'features': ['open', 'high', 'low', 'close', 'volume']},
            'database': {'db_path': 'test.db', 'backup_path': 'test_backups/'},
            'logging': {'log_file': 'test.log'}
        }
        
        config_file = self.create_test_config_file(config_dict)
        
        # Test various boolean representations
        boolean_tests = [
            ('true', True),
            ('1', True),
            ('yes', True),
            ('on', True),
            ('false', False),
            ('0', False),
            ('no', False),
            ('off', False),
        ]
        
        for env_value, expected_bool in boolean_tests:
            try:
                with patch.dict(os.environ, {'PAPER_TRADING': env_value}):
                    config_manager = ConfigManager(config_file)
                    trading_config = config_manager.get_trading_config()
                    assert trading_config.paper_trading == expected_bool
            finally:
                pass
        
        os.unlink(config_file)

class TestAdaptiveParameterConstraints:
    """Test adaptive parameter system constraints."""
    
    def test_adaptive_parameter_bounds_validation(self):
        """Test adaptive parameter bounds are enforced."""
        from trading.adaptive_parameters import AdaptiveParameterManager
        
        manager = AdaptiveParameterManager()
        
        # Test parameter validation
        valid_params = {
            'confidence_threshold': 0.45,
            'min_price_change_pct': 1.2,
            'max_interval_width_pct': 20.0,
            'max_position_size': 0.10
        }
        
        validation = manager.validate_parameters(valid_params)
        assert validation['valid'] == True
        assert len(validation['warnings']) == 0
        
        # Test invalid parameters get corrected
        invalid_params = {
            'confidence_threshold': 0.80,  # Above max (0.65)
            'min_price_change_pct': 0.5,   # Below min (0.8)
            'max_position_size': 0.20      # Above max (0.15)
        }
        
        validation = manager.validate_parameters(invalid_params)
        assert validation['valid'] == False
        assert len(validation['warnings']) > 0
        assert len(validation['corrections']) > 0
    
    def test_optimal_parameter_values(self):
        """Test optimal parameter values from backtesting."""
        from trading.adaptive_parameters import AdaptiveParameterManager
        
        manager = AdaptiveParameterManager()
        optimal_params = manager.reset_to_optimal()
        
        # Verify optimal values match backtesting insights
        assert optimal_params['confidence_threshold'] == 0.45
        assert optimal_params['min_price_change_pct'] == 1.2
        assert optimal_params['max_interval_width_pct'] == 20.0
        assert optimal_params['max_position_size'] == 0.10
        
        # Verify all optimal values are within bounds
        validation = manager.validate_parameters(optimal_params)
        assert validation['valid'] == True

class TestMonitoringConfigValidation:
    """Test monitoring configuration validation."""
    
    def test_alert_thresholds_validation(self):
        """Test monitoring alert thresholds validation."""
        valid_config = {
            'health_check_interval': 300,
            'performance_log_interval': 3600,
            'alert_thresholds': {
                'max_memory_mb': 2048,
                'max_cpu_percent': 80,
                'min_disk_space_gb': 5,
                'max_api_errors_per_hour': 10,
                'min_win_rate_pct': 40.0,
                'max_drawdown_pct': 15.0,
                'min_confidence_avg': 0.30,
                'max_uncertainty_avg': 25.0,
                'min_trades_per_day': 0.5,
                'max_trades_per_day': 10.0,
                'parameter_drift_threshold': 0.20
            }
        }
        
        config = MonitoringConfigModel(**valid_config)
        assert config.alert_thresholds['min_win_rate_pct'] == 40.0
        assert config.alert_thresholds['parameter_drift_threshold'] == 0.20

if __name__ == '__main__':
    # Run tests if script is executed directly
    pytest.main([__file__, '-v'])