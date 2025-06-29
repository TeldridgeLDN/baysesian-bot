"""
Configuration management for the crypto trading bot.
Handles loading and validation of configuration parameters.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, ValidationError
from pathlib import Path

logger = logging.getLogger(__name__)

# Pydantic models for configuration validation
class TradingConfigModel(BaseModel):
    """Trading configuration validation model - Updated with backtesting insights."""
    max_position_size: float = Field(default=0.10, gt=0, le=1, description="Maximum position size as fraction of capital")
    confidence_threshold: float = Field(default=0.45, ge=0.25, le=1, description="Minimum confidence for trades")
    stop_loss_multiplier: float = Field(default=2.0, gt=0, description="Stop loss multiplier")
    take_profit_ratio: float = Field(default=1.5, gt=0, description="Risk-reward ratio")
    max_drawdown: float = Field(default=0.15, gt=0, le=0.25, description="Maximum drawdown threshold")
    position_timeout_hours: int = Field(default=24, gt=0, description="Position timeout in hours")
    min_price_change_pct: float = Field(default=1.2, gt=0.5, le=5.0, description="Minimum price change percentage")
    max_interval_width_pct: float = Field(default=20.0, gt=0, le=30.0, description="Maximum uncertainty width percentage")
    max_active_positions_pct: float = Field(default=0.30, gt=0, le=1, description="Maximum active positions as fraction")
    paper_trading: bool = Field(default=True, description="Paper trading mode flag")
    historical_days: int = Field(default=180, gt=0, description="Historical data window")
    trading_fee: float = Field(default=0.001, ge=0, le=0.01, description="Trading fee percentage")
    slippage: float = Field(default=0.0002, ge=0, le=0.01, description="Slippage percentage")
    risk_grading_enabled: bool = Field(default=True, description="Enable risk-graded position sizing")
    adaptive_parameters: bool = Field(default=False, description="Enable adaptive parameter adjustment")

    @validator('max_position_size')
    def validate_max_position_size(cls, v):
        if v > 0.15:  # Updated: 15% safety limit based on backtesting
            logger.warning(f"Position size {v} exceeds recommended 15% limit for crypto volatility")
        return v

class ModelConfigModel(BaseModel):
    """Model configuration validation model."""
    sequence_length: int = Field(default=60, gt=0, description="Input sequence length")
    retrain_frequency_hours: int = Field(default=24, gt=0, description="Retraining frequency")
    monte_carlo_samples: int = Field(default=100, gt=0, le=1000, description="Monte Carlo samples")
    confidence_interval: float = Field(default=0.95, gt=0, lt=1, description="Confidence interval")
    lstm_units: List[int] = Field(default=[128, 64, 32], description="LSTM layer units")
    dropout_rate: float = Field(default=0.2, ge=0, le=1, description="Dropout rate")
    recurrent_dropout_rate: float = Field(default=0.2, ge=0, le=1, description="Recurrent dropout rate")
    dense_dropout_rate: float = Field(default=0.3, ge=0, le=1, description="Dense layer dropout rate")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    batch_size: int = Field(default=32, gt=0, description="Training batch size")
    epochs: int = Field(default=100, gt=0, description="Maximum epochs")
    early_stopping_patience: int = Field(default=10, gt=0, description="Early stopping patience")
    validation_split: float = Field(default=0.2, gt=0, lt=1, description="Validation split")

    @validator('lstm_units')
    def validate_lstm_units(cls, v):
        if len(v) < 1:
            raise ValueError("At least one LSTM layer required")
        if any(units <= 0 for units in v):
            raise ValueError("All LSTM units must be positive")
        return v

class DataConfigModel(BaseModel):
    """Data configuration validation model."""
    historical_days: int = Field(default=180, gt=0, description="Historical data window")
    update_frequency_minutes: int = Field(default=60, gt=0, description="Update frequency")
    backup_retention_days: int = Field(default=30, gt=0, description="Backup retention period")
    price_data_sources: Dict[str, str] = Field(default={'primary': 'coingecko', 'backup': 'binance'}, description="Data sources")
    features: List[str] = Field(default=['open', 'high', 'low', 'close', 'volume'], description="Feature list")
    data_validation: Dict[str, float] = Field(default={'max_price_change_pct': 20.0, 'min_volume_threshold': 1000}, description="Data validation parameters")

    @validator('historical_days')
    def validate_historical_days(cls, v):
        if v < 30:
            raise ValueError("Historical days must be at least 30")
        return v

    @validator('features')
    def validate_features(cls, v):
        required_features = ['open', 'high', 'low', 'close', 'volume']
        for feature in required_features:
            if feature not in v:
                raise ValueError(f"Required feature '{feature}' missing")
        return v

class TelegramConfigModel(BaseModel):
    """Telegram configuration validation model."""
    notification_level: str = Field(description="Notification level")
    chart_resolution: str = Field(description="Chart resolution")
    max_message_length: int = Field(gt=0, le=4096, description="Max message length")
    notification_types: Dict[str, bool] = Field(description="Notification types")
    chart_indicators: List[str] = Field(description="Chart indicators")

    @validator('notification_level')
    def validate_notification_level(cls, v):
        valid_levels = ['minimal', 'normal', 'verbose']
        if v not in valid_levels:
            raise ValueError(f"Notification level must be one of {valid_levels}")
        return v

class DatabaseConfigModel(BaseModel):
    """Database configuration validation model."""
    db_path: str = Field(description="Database file path")
    backup_path: str = Field(description="Backup directory path")
    connection_timeout: int = Field(gt=0, description="Connection timeout")
    max_retries: int = Field(ge=0, description="Maximum retry attempts")

class CoinGeckoConfig(BaseModel):
    name: str
    base_url: str
    rate_limit_per_minute: int
    timeout_seconds: int

class BinanceConfig(BaseModel):
    name: str
    base_url: str
    rate_limit_per_minute: int
    timeout_seconds: int

class APIConfigModel(BaseModel):
    """API configuration validation model."""
    coingecko: CoinGeckoConfig
    binance: BinanceConfig

class LoggingConfigModel(BaseModel):
    """Logging configuration validation model."""
    level: str = Field(description="Logging level")
    format: str = Field(description="Log format")
    log_file: str = Field(description="Log file path")
    max_file_size: int = Field(gt=0, description="Max log file size")
    backup_count: int = Field(ge=0, description="Log backup count")
    console_output: bool = Field(description="Console output flag")

    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

class SecurityConfigModel(BaseModel):
    """Security configuration validation model."""
    encrypt_database: bool = Field(description="Database encryption flag")
    api_key_encryption: bool = Field(description="API key encryption flag")
    session_timeout_minutes: int = Field(gt=0, description="Session timeout")
    max_login_attempts: int = Field(gt=0, description="Max login attempts")

class MonitoringConfigModel(BaseModel):
    """Monitoring configuration validation model."""
    health_check_interval: int = Field(gt=0, description="Health check interval")
    performance_log_interval: int = Field(gt=0, description="Performance log interval")
    alert_thresholds: Dict[str, Any] = Field(description="Alert thresholds")

# Configuration dataclasses for application use
@dataclass
class TradingConfig:
    """Trading configuration parameters - Updated with optimal backtesting results."""
    # UPDATED: Optimal parameters from backtesting
    max_position_size: float = 0.10  # Keep conservative for safety
    confidence_threshold: float = 0.40  # CHANGED: From 0.60 to 0.40 (key finding)
    stop_loss_multiplier: float = 2.0
    take_profit_ratio: float = 1.5
    max_drawdown: float = 0.15  # CHANGED: From 0.10 to 0.15 (more realistic for crypto)
    position_timeout_hours: int = 24
    min_price_change_pct: float = 1.0  # CHANGED: From 2.0 to 1.0 (more aggressive)
    max_interval_width_pct: float = 22.0  # CHANGED: From 3.0 to 22.0 (crypto-realistic)
    max_active_positions_pct: float = 0.30
    paper_trading: bool = True
    historical_days: int = 180
    # NEW: Trading cost parameters
    trading_fee: float = 0.001  # 0.1% trading fee
    slippage: float = 0.0002    # 0.02% slippage
    # NEW: Risk management enhancements
    risk_grading_enabled: bool = True  # Enable risk-graded position sizing
    adaptive_parameters: bool = False  # Future: market regime adaptation

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    sequence_length: int = 60
    retrain_frequency_hours: int = 24
    monte_carlo_samples: int = 100
    confidence_interval: float = 0.95
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    recurrent_dropout_rate: float = 0.2
    dense_dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2

@dataclass
class DataConfig:
    """Data configuration parameters."""
    historical_days: int = 180
    update_frequency_minutes: int = 60
    backup_retention_days: int = 30
    price_data_sources: Dict[str, str] = field(default_factory=lambda: {"primary": "coingecko", "backup": "binance"})
    features: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume", "rsi", "macd", "bollinger_bands"])
    data_validation: Dict[str, float] = field(default_factory=lambda: {"max_price_change_pct": 20.0, "min_volume_threshold": 1000})

@dataclass
class TelegramConfig:
    """Telegram bot configuration parameters."""
    notification_level: str = "normal"
    chart_resolution: str = "1h"
    max_message_length: int = 4000
    notification_types: Dict[str, bool] = field(default_factory=lambda: {
        "trade_entry": True,
        "trade_exit": True,
        "high_confidence": True,
        "model_retrained": True,
        "risk_alert": True,
        "daily_summary": True
    })
    chart_indicators: List[str] = field(default_factory=lambda: ["price", "predictions", "confidence_bands", "volume"])

@dataclass
class DatabaseConfig:
    """Database configuration parameters."""
    db_path: str = "data/trading_bot.db"
    backup_path: str = "data/backups/"
    connection_timeout: int = 30
    max_retries: int = 3

@dataclass
class APIConfig:
    """API configuration parameters."""
    coingecko: CoinGeckoConfig
    binance: BinanceConfig

@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trading_bot.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = True

@dataclass
class SecurityConfig:
    """Security configuration parameters."""
    encrypt_database: bool = False
    api_key_encryption: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 3

@dataclass
class MonitoringConfig:
    """Monitoring configuration parameters - Enhanced with trading performance tracking."""
    health_check_interval: int = 300
    performance_log_interval: int = 3600
    alert_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": 2048,
        "max_cpu_percent": 80,
        "min_disk_space_gb": 5,
        "max_api_errors_per_hour": 10,
        # Trading performance thresholds based on backtesting
        "min_win_rate_pct": 40.0,  # Alert if win rate drops below 40%
        "max_drawdown_pct": 15.0,  # Alert if drawdown exceeds 15%
        "min_confidence_avg": 0.30,  # Alert if average confidence drops
        "max_uncertainty_avg": 25.0,  # Alert if uncertainty gets too high
        "min_trades_per_day": 0.5,  # Alert if trading frequency too low
        "max_trades_per_day": 10.0,  # Alert if overtrading
        "parameter_drift_threshold": 0.20  # Alert if parameters drift >20% from optimal
    })

class ConfigManager:
    """Manages application configuration with validation and environment variable support."""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = config_path
        self.raw_config = {}
        self.validated_config = {}
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration from YAML file with environment variable substitution."""
        try:
            # Load YAML configuration
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(config_file, 'r') as file:
                self.raw_config = yaml.safe_load(file)
            
            # Apply environment variable overrides
            self.raw_config = self._apply_env_overrides(self.raw_config)
            
            # Validate configuration
            self.validated_config = self._validate_config(self.raw_config)
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.validated_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Environment variable mapping
        env_mappings = {
            'TELEGRAM_BOT_TOKEN': ['telegram', 'bot_token'],
            'COINGECKO_API_KEY': ['api', 'coingecko', 'api_key'],
            'BINANCE_API_KEY': ['api', 'binance', 'api_key'],
            'BINANCE_SECRET_KEY': ['api', 'binance', 'secret_key'],
            'DATABASE_PATH': ['database', 'db_path'],
            'LOG_LEVEL': ['logging', 'level'],
            'PAPER_TRADING': ['trading', 'paper_trading'],
            'MAX_POSITION_SIZE': ['trading', 'max_position_size'],
            'CONFIDENCE_THRESHOLD': ['trading', 'confidence_threshold']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the nested config location
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert environment variable to appropriate type
                final_key = config_path[-1]
                if env_var in ['PAPER_TRADING']:
                    current[final_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif env_var in ['MAX_POSITION_SIZE', 'CONFIDENCE_THRESHOLD']:
                    current[final_key] = float(env_value)
                else:
                    current[final_key] = env_value
                
                logger.info(f"Applied environment override: {env_var} -> {'.'.join(config_path)}")
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration using pydantic models."""
        validated = {}
        
        try:
            # Validate each configuration section
            if 'trading' in config:
                validated['trading'] = TradingConfigModel(**config['trading']).dict()
            
            if 'model' in config:
                validated['model'] = ModelConfigModel(**config['model']).dict()
            
            if 'data' in config:
                validated['data'] = DataConfigModel(**config['data']).dict()
            
            if 'telegram' in config:
                validated['telegram'] = TelegramConfigModel(**config['telegram']).dict()
            
            if 'database' in config:
                validated['database'] = DatabaseConfigModel(**config['database']).dict()
            
            if 'api' in config:
                validated['api'] = APIConfigModel(**config['api']).dict()
            
            if 'logging' in config:
                validated['logging'] = LoggingConfigModel(**config['logging']).dict()
            
            if 'security' in config:
                validated['security'] = SecurityConfigModel(**config['security']).dict()
            
            if 'monitoring' in config:
                validated['monitoring'] = MonitoringConfigModel(**config['monitoring']).dict()
            
            logger.info("Configuration validation completed successfully")
            return validated
            
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during configuration validation: {str(e)}")
            raise
    
    def get_trading_config(self) -> TradingConfig:
        """Get validated trading configuration."""
        config_data = self.config.get('trading', {})
        return TradingConfig(**config_data)
        
    def get_model_config(self) -> ModelConfig:
        """Get validated model configuration."""
        config_data = self.config.get('model', {})
        return ModelConfig(**config_data)
        
    def get_data_config(self) -> DataConfig:
        """Get validated data configuration."""
        config_data = self.config.get('data', {})
        return DataConfig(**config_data)
        
    def get_telegram_config(self) -> TelegramConfig:
        """Get validated Telegram configuration."""
        config_data = self.config.get('telegram', {})
        return TelegramConfig(**config_data)
    
    def get_database_config(self) -> DatabaseConfig:
        """Get validated database configuration."""
        config_data = self.config.get('database', {})
        return DatabaseConfig(**config_data)
    
    def get_api_config(self) -> APIConfig:
        """Get validated API configuration."""
        config_data = self.config.get('api', {})
        return APIConfig(**config_data)
    
    def get_logging_config(self) -> LoggingConfig:
        """Get validated logging configuration."""
        config_data = self.config.get('logging', {})
        return LoggingConfig(**config_data)
    
    def get_security_config(self) -> SecurityConfig:
        """Get validated security configuration."""
        config_data = self.config.get('security', {})
        return SecurityConfig(**config_data)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get validated monitoring configuration."""
        config_data = self.config.get('monitoring', {})
        return MonitoringConfig(**config_data)
        
    def validate_config(self) -> bool:
        """Validate current configuration and return success status."""
        try:
            # Re-validate the current configuration
            self._validate_config(self.raw_config)
            
            # Additional runtime validations
            self._validate_runtime_constraints()
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def _validate_runtime_constraints(self):
        """Perform additional runtime validation checks."""
        trading_config = self.get_trading_config()
        
        # Validate trading constraints
        if trading_config.max_position_size > trading_config.max_active_positions_pct:
            raise ValueError("max_position_size cannot exceed max_active_positions_pct")
        
        if trading_config.confidence_threshold >= 1.0:
            raise ValueError("confidence_threshold must be less than 1.0")
        
        # Enhanced risk validation based on backtesting insights
        if trading_config.confidence_threshold < 0.25:
            logger.warning("Very low confidence threshold - high risk of poor trades")
            
        if trading_config.max_interval_width_pct > 25.0:
            logger.warning("Very high uncertainty tolerance - trades may be unreliable")
            
        if trading_config.min_price_change_pct < 0.8:
            logger.warning("Very low price change threshold - may generate excessive trades")
            
        # Portfolio risk checks
        total_risk_exposure = trading_config.max_position_size * trading_config.max_active_positions_pct
        if total_risk_exposure > 0.50:  # 50% maximum total exposure
            raise ValueError(f"Total risk exposure {total_risk_exposure:.1%} exceeds 50% safety limit")
        
        # Validate model constraints
        model_config = self.get_model_config()
        if model_config.validation_split >= 1.0:
            raise ValueError("validation_split must be less than 1.0")
        
        # Validate data constraints
        data_config = self.get_data_config()
        if data_config.historical_days < model_config.sequence_length:
            raise ValueError("historical_days must be >= sequence_length")
        
        # Validate file paths exist or can be created
        database_config = self.get_database_config()
        db_dir = Path(database_config.db_path).parent
        backup_dir = Path(database_config.backup_path)
        
        for directory in [db_dir, backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate log directory
        logging_config = self.get_logging_config()
        log_dir = Path(logging_config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def reload_config(self) -> bool:
        """Reload configuration from file."""
        try:
            self.config = self.load_config()
            logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload configuration: {str(e)}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration for logging/debugging."""
        summary = {
            'config_file': self.config_path,
            'trading': {
                'paper_trading': self.get_trading_config().paper_trading,
                'max_position_size': self.get_trading_config().max_position_size,
                'confidence_threshold': self.get_trading_config().confidence_threshold
            },
            'model': {
                'sequence_length': self.get_model_config().sequence_length,
                'monte_carlo_samples': self.get_model_config().monte_carlo_samples
            },
            'data': {
                'historical_days': self.get_data_config().historical_days,
                'primary_source': self.get_data_config().price_data_sources.get('primary')
            }
        }
        return summary 