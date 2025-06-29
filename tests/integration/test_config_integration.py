"""
Integration tests for configuration system with trading components.
Tests the complete configuration flow including adaptive parameters and trading engine integration.
"""

import pytest
import tempfile
import yaml
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.config import ConfigManager, TradingConfig, ModelConfig
from trading.adaptive_parameters import AdaptiveParameterManager, ParameterBounds
from trading.paper_trading import PaperTradingEngine, PaperTradingMode

class TestConfigTradingIntegration:
    """Test configuration integration with trading systems."""
    
    def create_complete_config(self) -> Dict[str, Any]:
        """Create a complete, valid configuration for testing."""
        return {
            'trading': {
                'initial_capital': 10000.0,
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
                'adaptive_parameters': True
            },
            'model': {
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
            },
            'data': {
                'historical_days': 180,
                'update_frequency_minutes': 60,
                'backup_retention_days': 30,
                'price_data_sources': {'primary': 'coingecko', 'backup': 'binance'},
                'features': ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd'],
                'data_validation': {'max_price_change_pct': 20.0, 'min_volume_threshold': 1000}
            },
            'telegram': {
                'notification_level': 'normal',
                'chart_resolution': '1h',
                'max_message_length': 4000,
                'notification_types': {
                    'trade_entry': True,
                    'trade_exit': True,
                    'parameter_adjustment': True,
                    'performance_alert': True
                },
                'chart_indicators': ['price', 'predictions', 'confidence_bands']
            },
            'database': {
                'db_path': 'data/test_trading_bot.db',
                'backup_path': 'data/test_backups/',
                'connection_timeout': 30,
                'max_retries': 3
            },
            'api': {
                'coingecko': {
                    'name': 'CoinGecko',
                    'base_url': 'https://api.coingecko.com/api/v3',
                    'rate_limit_per_minute': 50,
                    'timeout_seconds': 30
                },
                'binance': {
                    'name': 'Binance',
                    'base_url': 'https://api.binance.com/api/v3',
                    'rate_limit_per_minute': 1200,
                    'timeout_seconds': 10
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_file': 'logs/test_trading_bot.log',
                'max_file_size': 10485760,
                'backup_count': 5,
                'console_output': True
            },
            'security': {
                'encrypt_database': False,
                'api_key_encryption': True,
                'session_timeout_minutes': 60,
                'max_login_attempts': 3
            },
            'monitoring': {
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
        }
    
    def create_test_config_file(self, config_dict: Dict[str, Any]) -> str:
        """Create a temporary config file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name
    
    def test_complete_config_validation(self):
        """Test complete configuration loads and validates successfully."""
        config_dict = self.create_complete_config()
        config_file = self.create_test_config_file(config_dict)
        
        try:
            config_manager = ConfigManager(config_file)
            
            # Test all configuration sections load
            trading_config = config_manager.get_trading_config()
            model_config = config_manager.get_model_config()
            data_config = config_manager.get_data_config()
            telegram_config = config_manager.get_telegram_config()
            database_config = config_manager.get_database_config()
            
            # Validate key parameters
            assert trading_config.confidence_threshold == 0.45
            assert trading_config.adaptive_parameters == True
            assert model_config.sequence_length == 60
            assert data_config.historical_days == 180
            
            # Test runtime validation passes
            assert config_manager.validate_config() == True
            
        finally:
            os.unlink(config_file)
    
    def test_adaptive_parameter_config_integration(self):
        """Test adaptive parameter system integrates with configuration."""
        config_dict = self.create_complete_config()
        config_file = self.create_test_config_file(config_dict)
        
        try:
            config_manager = ConfigManager(config_file)
            trading_config = config_manager.get_trading_config()
            
            # Create adaptive parameter manager
            adaptive_manager = AdaptiveParameterManager()
            
            # Test current parameters are within bounds
            current_params = {
                'confidence_threshold': trading_config.confidence_threshold,
                'min_price_change_pct': trading_config.min_price_change_pct,
                'max_interval_width_pct': trading_config.max_interval_width_pct,
                'max_position_size': trading_config.max_position_size
            }
            
            validation = adaptive_manager.validate_parameters(current_params)
            assert validation['valid'] == True
            
            # Test optimal parameters are compatible with config structure
            optimal_params = adaptive_manager.reset_to_optimal()
            
            # Verify we can update trading config with optimal parameters
            for param_name, value in optimal_params.items():
                if hasattr(trading_config, param_name):
                    setattr(trading_config, param_name, value)
            
            # Config should still be valid after optimization
            assert config_manager.validate_config() == True
            
        finally:
            os.unlink(config_file)
    
    def test_paper_trading_config_integration(self):
        """Test paper trading system uses configuration correctly."""
        config_dict = self.create_complete_config()
        config_file = self.create_test_config_file(config_dict)
        
        try:
            config_manager = ConfigManager(config_file)
            trading_config = config_manager.get_trading_config()
            
            # Create paper trading engine
            paper_engine = PaperTradingEngine(
                initial_capital=trading_config.initial_capital,
                mode=PaperTradingMode.ADAPTATION_TEST
            )
            
            # Test parameter set creation from config
            config_params = {
                'confidence_threshold': trading_config.confidence_threshold,
                'min_price_change_pct': trading_config.min_price_change_pct,
                'max_interval_width_pct': trading_config.max_interval_width_pct,
                'max_position_size': trading_config.max_position_size,
                'stop_loss_multiplier': trading_config.stop_loss_multiplier,
                'take_profit_ratio': trading_config.take_profit_ratio,
                'position_timeout_hours': trading_config.position_timeout_hours,
                'risk_grading_enabled': trading_config.risk_grading_enabled
            }
            
            success = paper_engine.add_parameter_set('config_params', config_params)
            assert success == True
            
            # Test paper trading respects configuration
            assert paper_engine.initial_capital == trading_config.initial_capital
            
        finally:
            os.unlink(config_file)
    
    def test_configuration_parameter_boundaries(self):
        """Test configuration parameters respect boundaries from backtesting."""
        config_dict = self.create_complete_config()
        
        # Test optimal configuration (should pass all validations)
        optimal_trading = {
            'confidence_threshold': 0.45,  # Optimal from backtesting
            'min_price_change_pct': 1.2,   # Optimal from backtesting
            'max_interval_width_pct': 20.0, # Optimal from backtesting
            'max_position_size': 0.10,     # Optimal from backtesting
            'max_drawdown': 0.15,          # Conservative limit
            'max_active_positions_pct': 0.30,
            'paper_trading': True,
            'risk_grading_enabled': True,
            'adaptive_parameters': True
        }
        
        config_dict['trading'].update(optimal_trading)
        config_file = self.create_test_config_file(config_dict)
        
        try:
            config_manager = ConfigManager(config_file)
            assert config_manager.validate_config() == True
            
            # Test adaptive parameter manager accepts these values
            adaptive_manager = AdaptiveParameterManager()
            validation = adaptive_manager.validate_parameters(optimal_trading)
            assert validation['valid'] == True
            
        finally:
            os.unlink(config_file)
    
    def test_configuration_edge_cases(self):
        """Test configuration validation at boundary conditions."""
        base_config = self.create_complete_config()
        
        # Test cases at parameter boundaries
        boundary_tests = [
            # (parameter_path, test_value, should_pass)
            (['trading', 'confidence_threshold'], 0.30, True),   # Min boundary
            (['trading', 'confidence_threshold'], 0.65, True),   # Max boundary  
            (['trading', 'confidence_threshold'], 0.29, False),  # Below min
            (['trading', 'confidence_threshold'], 0.66, False),  # Above max
            
            (['trading', 'min_price_change_pct'], 0.8, True),    # Min boundary
            (['trading', 'min_price_change_pct'], 2.5, True),    # Max boundary
            (['trading', 'min_price_change_pct'], 0.7, False),   # Below min
            (['trading', 'min_price_change_pct'], 2.6, False),   # Above max
            
            (['trading', 'max_interval_width_pct'], 15.0, True), # Min boundary
            (['trading', 'max_interval_width_pct'], 30.0, True), # Max boundary
            (['trading', 'max_interval_width_pct'], 14.9, False), # Below min
            (['trading', 'max_interval_width_pct'], 30.1, False), # Above max
            
            (['trading', 'max_position_size'], 0.05, True),      # Min boundary
            (['trading', 'max_position_size'], 0.15, True),      # Max boundary
            (['trading', 'max_position_size'], 0.04, False),     # Below min
            (['trading', 'max_position_size'], 0.16, False),     # Above max
        ]
        
        for param_path, test_value, should_pass in boundary_tests:
            config_dict = base_config.copy()
            
            # Set the test value at the specified path
            current = config_dict
            for key in param_path[:-1]:
                current = current[key]
            current[param_path[-1]] = test_value
            
            config_file = self.create_test_config_file(config_dict)
            
            try:
                config_manager = ConfigManager(config_file)
                validation_result = config_manager.validate_config()
                
                if should_pass:
                    assert validation_result == True, f"Expected {param_path} = {test_value} to pass validation"
                else:
                    assert validation_result == False, f"Expected {param_path} = {test_value} to fail validation"
                    
            finally:
                os.unlink(config_file)
    
    def test_environment_override_integration(self):
        """Test environment variable overrides work with trading systems."""
        config_dict = self.create_complete_config()
        config_file = self.create_test_config_file(config_dict)
        
        try:
            # Test with environment overrides
            with patch.dict(os.environ, {
                'CONFIDENCE_THRESHOLD': '0.40',  # More aggressive
                'MAX_POSITION_SIZE': '0.12',     # Slightly larger
                'PAPER_TRADING': 'false'         # Live trading mode
            }):
                config_manager = ConfigManager(config_file)
                trading_config = config_manager.get_trading_config()
                
                # Verify overrides were applied
                assert trading_config.confidence_threshold == 0.40
                assert trading_config.max_position_size == 0.12
                assert trading_config.paper_trading == False
                
                # Test adaptive parameter manager accepts overridden values
                adaptive_manager = AdaptiveParameterManager()
                current_params = {
                    'confidence_threshold': trading_config.confidence_threshold,
                    'max_position_size': trading_config.max_position_size
                }
                
                validation = adaptive_manager.validate_parameters(current_params)
                assert validation['valid'] == True
                
        finally:
            os.unlink(config_file)
    
    def test_configuration_monitoring_integration(self):
        """Test configuration integrates with monitoring system."""
        config_dict = self.create_complete_config()
        config_file = self.create_test_config_file(config_dict)
        
        try:
            config_manager = ConfigManager(config_file)
            monitoring_config = config_manager.get_monitoring_config()
            
            # Test alert thresholds are properly configured
            alert_thresholds = monitoring_config.alert_thresholds
            
            # Verify trading performance thresholds
            assert alert_thresholds['min_win_rate_pct'] == 40.0
            assert alert_thresholds['max_drawdown_pct'] == 15.0
            assert alert_thresholds['min_confidence_avg'] == 0.30
            assert alert_thresholds['max_uncertainty_avg'] == 25.0
            assert alert_thresholds['parameter_drift_threshold'] == 0.20
            
            # Test thresholds align with trading configuration
            trading_config = config_manager.get_trading_config()
            
            # Max drawdown threshold should not exceed trading config limit
            assert alert_thresholds['max_drawdown_pct'] <= trading_config.max_drawdown * 100
            
            # Confidence threshold should be reasonable relative to trading threshold
            assert alert_thresholds['min_confidence_avg'] < trading_config.confidence_threshold
            
        finally:
            os.unlink(config_file)

class TestConfigurationErrorHandling:
    """Test configuration error handling and recovery."""
    
    def test_missing_required_sections(self):
        """Test handling of missing required configuration sections."""
        incomplete_configs = [
            {},  # Completely empty
            {'trading': {}},  # Missing other sections
            {'model': {}, 'data': {}},  # Missing trading section
        ]
        
        for config_dict in incomplete_configs:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(config_dict, temp_file)
            temp_file.close()
            
            try:
                # Should handle missing sections gracefully
                config_manager = ConfigManager(temp_file.name)
                
                # Should be able to get configs with defaults
                trading_config = config_manager.get_trading_config()
                assert trading_config is not None
                
            except Exception as e:
                # Expected for completely invalid configs
                assert "not found" in str(e).lower() or "missing" in str(e).lower()
                
            finally:
                os.unlink(temp_file.name)
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML files."""
        invalid_yaml = "invalid: yaml: content: [unclosed"
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        temp_file.write(invalid_yaml)
        temp_file.close()
        
        try:
            with pytest.raises(Exception):  # Should raise YAML parsing error
                ConfigManager(temp_file.name)
        finally:
            os.unlink(temp_file.name)
    
    def test_nonexistent_config_file(self):
        """Test handling of nonexistent configuration files."""
        with pytest.raises(FileNotFoundError):
            ConfigManager('nonexistent_config.yaml')
    
    def test_configuration_recovery(self):
        """Test configuration can recover from partial failures."""
        # Config with some invalid values but valid structure
        mixed_config = {
            'trading': {
                'confidence_threshold': 0.45,  # Valid
                'max_position_size': 2.0,      # Invalid (>100%)
                'paper_trading': True
            },
            'model': {
                'sequence_length': 60,
                'lstm_units': [128, 64, 32],
                'validation_split': 1.5  # Invalid (>1.0)
            },
            'data': {
                'historical_days': 30,
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
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(mixed_config, temp_file)
        temp_file.close()
        
        try:
            config_manager = ConfigManager(temp_file.name)
            
            # Should fail validation due to invalid values
            assert config_manager.validate_config() == False
            
            # But should still be able to get valid sections
            data_config = config_manager.get_data_config()
            assert data_config.historical_days == 30
            
        finally:
            os.unlink(temp_file.name)

if __name__ == '__main__':
    # Run tests if script is executed directly
    pytest.main([__file__, '-v'])