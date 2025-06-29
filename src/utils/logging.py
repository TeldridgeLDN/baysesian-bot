"""
Logging configuration and setup for the crypto trading bot.
Handles log formatting, file rotation, and different log levels.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Dict

def setup_logging(config: Dict = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': 'logs/trading_bot.log',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5
        }
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.get('log_file', 'logs/trading_bot.log'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(config.get('format'))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        config.get('log_file'),
        maxBytes=config.get('max_file_size'),
        backupCount=config.get('backup_count')
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)

class TradingLogger:
    """Specialized logger for trading operations."""
    
    def __init__(self, name: str = "trading"):
        self.logger = logging.getLogger(name)
        
    def log_trade_entry(self, trade_data: Dict):
        """Log trade entry."""
        # Implementation placeholder
        pass
        
    def log_trade_exit(self, trade_data: Dict):
        """Log trade exit."""
        # Implementation placeholder
        pass
        
    def log_prediction(self, prediction_data: Dict):
        """Log model prediction."""
        # Implementation placeholder
        pass
        
    def log_error(self, error_message: str, exception: Exception = None):
        """Log error with optional exception details."""
        # Implementation placeholder
        pass 