"""
Model factory for creating and managing prediction models in the trading system.
Provides easy integration between Bayesian LSTM models and the trading engine.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

from .prediction_service import PredictionService
from .bayesian_lstm import BayesianLSTM, ModelManager
from ..data.storage import DatabaseManager
from ..utils.config import ModelConfig
from ..data.collectors import PriceDataCollector

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating and managing prediction models.
    
    Provides unified interface for:
    - Model creation and loading
    - Integration with trading engine
    - Performance monitoring
    - Model lifecycle management
    """
    
    def __init__(self, 
                 base_model_path: str = "models/",
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize model factory.
        
        Args:
            base_model_path: Base directory for model storage
            db_manager: Database manager for persistence
        """
        self.base_model_path = Path(base_model_path)
        self.base_model_path.mkdir(parents=True, exist_ok=True)
        
        self.db_manager = db_manager
        self.active_services = {}
        
        logger.info(f"ModelFactory initialized with base path: {base_model_path}")
    
    def create_prediction_service(self, 
                                 model_config: ModelConfig,
                                 model_name: str = "primary",
                                 load_existing: bool = True) -> PredictionService:
        """
        Create a prediction service with Bayesian LSTM model.
        
        Args:
            model_config: Model configuration
            model_name: Name for the model (for saving/loading)
            load_existing: Whether to try loading existing model
            
        Returns:
            Configured PredictionService
        """
        logger.info(f"Creating prediction service: {model_name}")
        
        # Determine model path
        model_path = self.base_model_path / f"bayesian_lstm_{model_name}"
        model_path_str = str(model_path) if load_existing else None
        
        # Create prediction service
        service = PredictionService(
            model_config=model_config,
            db_manager=self.db_manager,
            model_path=model_path_str,
            enable_caching=True
        )
        
        # Initialize the service
        if service.initialize():
            self.active_services[model_name] = service
            logger.info(f"Prediction service '{model_name}' created successfully")
            return service
        else:
            raise RuntimeError(f"Failed to initialize prediction service '{model_name}'")
    
    def get_service(self, model_name: str = "primary") -> Optional[PredictionService]:
        """Get an active prediction service by name."""
        return self.active_services.get(model_name)
    
    def create_trading_ready_predictor(self, 
                                     model_config: ModelConfig,
                                     data_collector: Optional[PriceDataCollector] = None) -> 'TradingPredictor':
        """
        Create a trading-ready predictor that interfaces with the trading engine.
        
        Args:
            model_config: Model configuration
            data_collector: Data collector for getting price history
            
        Returns:
            TradingPredictor instance
        """
        service = self.create_prediction_service(model_config)
        
        return TradingPredictor(
            prediction_service=service,
            data_collector=data_collector,
            db_manager=self.db_manager
        )
    
    def train_new_model(self,
                       model_config: ModelConfig,
                       training_data: Dict[str, Any],
                       model_name: str = "trained_model") -> Optional[PredictionService]:
        """
        Train a new model with provided data.
        
        Args:
            model_config: Model configuration
            training_data: Training data dictionary with X_train, y_train, X_val, y_val
            model_name: Name for the trained model
            
        Returns:
            Trained PredictionService or None if training failed
        """
        try:
            logger.info(f"Training new model: {model_name}")
            
            # Create fresh service (no existing model)
            service = PredictionService(
                model_config=model_config,
                db_manager=self.db_manager,
                model_path=None,  # No existing model
                enable_caching=True
            )
            
            if not service.initialize():
                logger.error("Failed to initialize service for training")
                return None
            
            # Train the model
            training_stats = service.model.train(
                X_train=training_data['X_train'],
                y_train=training_data['y_train'],
                X_val=training_data['X_val'],
                y_val=training_data['y_val'],
                epochs=model_config.epochs,
                batch_size=model_config.batch_size,
                early_stopping_patience=model_config.early_stopping_patience
            )
            
            # Save the trained model
            model_path = self.base_model_path / f"bayesian_lstm_{model_name}"
            service.model.save_model(str(model_path))
            
            # Store training metrics in database
            if self.db_manager:
                from data.storage import ModelMetrics
                metrics = ModelMetrics(
                    timestamp=int(datetime.now().timestamp()),
                    mse=training_stats['final_val_loss'],
                    mae=training_stats['final_val_mae'],
                    directional_accuracy=0.0,  # Would need to be calculated
                    model_version=service.model.model_version
                )
                self.db_manager.store_model_metrics(metrics)
            
            self.active_services[model_name] = service
            
            logger.info(f"Model training completed: {model_name}")
            logger.info(f"Training stats: {training_stats}")
            
            return service
            
        except Exception as e:
            logger.error(f"Failed to train model {model_name}: {str(e)}")
            return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in the base path."""
        models = []
        
        # Check for saved models
        for model_file in self.base_model_path.glob("*.keras"):
            model_name = model_file.stem
            metadata_file = model_file.with_suffix('').with_suffix('_metadata.json')
            
            model_info = {
                'name': model_name,
                'path': str(model_file),
                'size_mb': model_file.stat().st_size / 1024 / 1024,
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime),
                'has_metadata': metadata_file.exists()
            }
            
            # Load metadata if available
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    model_info['metadata'] = metadata
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_name}: {e}")
            
            models.append(model_info)
        
        # Add active services
        for name, service in self.active_services.items():
            model_info = next((m for m in models if name in m['name']), None)
            if model_info:
                model_info['active'] = True
                model_info['service_stats'] = service.get_service_stats()
        
        return models
    
    def cleanup_old_models(self, keep_days: int = 30) -> int:
        """Remove old model files."""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        removed_count = 0
        
        for model_file in self.base_model_path.glob("*.keras"):
            if datetime.fromtimestamp(model_file.stat().st_mtime) < cutoff_date:
                # Remove model file and metadata
                model_file.unlink()
                metadata_file = model_file.with_suffix('').with_suffix('_metadata.json')
                if metadata_file.exists():
                    metadata_file.unlink()
                removed_count += 1
                logger.info(f"Removed old model: {model_file.name}")
        
        return removed_count

class TradingPredictor:
    """
    Trading-specific predictor that provides the interface expected by the trading engine.
    
    This class bridges the PredictionService with the trading engine's expected interface,
    providing the data points that the trading engine needs for decision making.
    """
    
    def __init__(self,
                 prediction_service: PredictionService,
                 data_collector: Optional[PriceDataCollector] = None,
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize trading predictor.
        
        Args:
            prediction_service: The underlying prediction service
            data_collector: For getting price history
            db_manager: Database manager
        """
        self.prediction_service = prediction_service
        self.data_collector = data_collector
        self.db_manager = db_manager
        
        self.prediction_history = []
        self.last_prediction_time = None
        
        logger.info("TradingPredictor initialized")
    
    def get_prediction_for_trading(self, 
                                  current_price_data: Dict[str, Any],
                                  historical_data: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
        """
        Get prediction data formatted for the trading engine.
        
        Args:
            current_price_data: Current price data point
            historical_data: Historical price data (if not using data_collector)
            
        Returns:
            Dictionary with prediction data for trading engine:
            {
                'timestamp': datetime,
                'actual_price': float,
                'predicted_price': float,
                'interval_width_pct': float,
                'confidence': float (0-1),
                'uncertainty_pct': float
            }
        """
        try:
            # Get historical data
            if historical_data is None and self.data_collector:
                # Use data collector to get recent price history
                historical_data = self.data_collector.get_recent_data(
                    hours=24  # Get last 24 hours of data
                )
            
            if not historical_data:
                logger.error("No historical data available for prediction")
                return None
            
            # Add current price to historical data
            price_data = historical_data + [current_price_data]
            
            # Get prediction
            prediction = self.prediction_service.predict(price_data, store_prediction=True)
            
            if not prediction:
                logger.error("Failed to get prediction from service")
                return None
            
            # Format for trading engine
            trading_data = {
                'timestamp': prediction['timestamp'],
                'actual_price': prediction['current_price'],
                'predicted_price': prediction['predicted_price'],
                'interval_width_pct': prediction['interval_width_pct'],
                'confidence': self._calculate_trading_confidence(prediction),
                'uncertainty_pct': prediction['interval_width_pct'],
                'price_change_pct': prediction['price_change_pct'],
                'model_version': prediction['model_version']
            }
            
            # Store in prediction history
            self.prediction_history.append(trading_data)
            if len(self.prediction_history) > 1000:  # Keep last 1000 predictions
                self.prediction_history.pop(0)
            
            self.last_prediction_time = datetime.now()
            
            return trading_data
            
        except Exception as e:
            logger.error(f"Error generating trading prediction: {str(e)}")
            return None
    
    def _calculate_trading_confidence(self, prediction: Dict[str, Any]) -> float:
        """
        Calculate trading confidence score from prediction uncertainty.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            Confidence score between 0 and 1
        """
        # Use interval width to determine confidence
        # Smaller intervals = higher confidence
        interval_width_pct = prediction['interval_width_pct']
        
        # Map interval width to confidence (inverse relationship)
        # Typical crypto intervals: 10-30%, map to confidence 0.9-0.3
        if interval_width_pct <= 10:
            confidence = 0.9
        elif interval_width_pct <= 15:
            confidence = 0.8
        elif interval_width_pct <= 20:
            confidence = 0.6
        elif interval_width_pct <= 25:
            confidence = 0.4
        else:
            confidence = 0.2
        
        return confidence
    
    def evaluate_prediction_accuracy(self, lookback_hours: int = 24) -> Optional[Dict[str, float]]:
        """Evaluate recent prediction accuracy."""
        if not self.db_manager:
            return None
        
        # Get actual price data for evaluation
        recent_prices = self.db_manager.get_recent_prices(hours=lookback_hours)
        if not recent_prices:
            return None
        
        return self.prediction_service.evaluate_recent_performance(
            actual_prices=recent_prices,
            lookback_hours=lookback_hours
        )
    
    def get_predictor_stats(self) -> Dict[str, Any]:
        """Get comprehensive predictor statistics."""
        stats = {
            'trading_predictor': {
                'predictions_made': len(self.prediction_history),
                'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                'data_collector_available': self.data_collector is not None,
                'database_available': self.db_manager is not None
            },
            'prediction_service': self.prediction_service.get_service_stats()
        }
        
        # Add recent prediction accuracy if available
        recent_accuracy = self.evaluate_prediction_accuracy(lookback_hours=6)
        if recent_accuracy:
            stats['recent_accuracy'] = recent_accuracy
        
        return stats