"""
Mock ML predictor for testing and demonstration purposes.
Provides the same interface as the Bayesian LSTM but without TensorFlow dependencies.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MockBayesianPredictor:
    """
    Mock predictor that simulates Bayesian LSTM predictions with uncertainty quantification.
    
    This class provides realistic-looking predictions with proper uncertainty modeling
    for testing and demonstration purposes without requiring TensorFlow.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 base_uncertainty: float = 0.15,
                 prediction_skill: float = 0.65):
        """
        Initialize mock predictor.
        
        Args:
            sequence_length: Required sequence length for predictions
            base_uncertainty: Base uncertainty level (0.1 = 10%)
            prediction_skill: Prediction skill level (0.5 = random, 1.0 = perfect)
        """
        self.sequence_length = sequence_length
        self.base_uncertainty = base_uncertainty
        self.prediction_skill = prediction_skill
        self.model_version = f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulation parameters
        self.trend_memory = 0.0  # Remembers recent trends
        self.volatility_memory = 0.02  # Remembers recent volatility
        self.prediction_count = 0
        
        logger.info(f"MockBayesianPredictor initialized: skill={prediction_skill:.2f}, "
                   f"uncertainty={base_uncertainty:.2f}")
    
    def predict_with_uncertainty(self, price_data: List[Dict]) -> Dict[str, Any]:
        """
        Generate mock prediction with realistic uncertainty quantification.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Prediction dictionary with uncertainty measures
        """
        if len(price_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
        
        self.prediction_count += 1
        
        # Extract recent prices
        recent_prices = [d['price'] for d in price_data[-self.sequence_length:]]
        current_price = recent_prices[-1]
        
        # Calculate market characteristics
        recent_returns = np.diff(recent_prices) / recent_prices[:-1]
        current_volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.02
        recent_trend = np.mean(recent_returns[-5:]) if len(recent_returns) >= 5 else 0.0
        
        # Update memory with exponential decay
        self.trend_memory = 0.7 * self.trend_memory + 0.3 * recent_trend
        self.volatility_memory = 0.8 * self.volatility_memory + 0.2 * current_volatility
        
        # Generate prediction with skill
        # Skillful component: follow trend with some insight
        skillful_prediction = current_price * (1 + self.trend_memory * 2.0)
        
        # Random component
        random_prediction = current_price * (1 + np.random.normal(0, current_volatility))
        
        # Combine based on skill level
        predicted_price = (self.prediction_skill * skillful_prediction + 
                          (1 - self.prediction_skill) * random_prediction)
        
        # Add some momentum and mean reversion
        price_momentum = (current_price - recent_prices[-10]) / recent_prices[-10] if len(recent_prices) >= 10 else 0
        momentum_adjustment = predicted_price * price_momentum * 0.1
        predicted_price += momentum_adjustment
        
        # Calculate uncertainty based on market conditions
        base_uncertainty = self.base_uncertainty
        volatility_uncertainty = current_volatility * 2.0  # Higher vol = higher uncertainty
        trend_uncertainty = abs(self.trend_memory) * 0.5   # Strong trends = higher uncertainty
        
        total_uncertainty = base_uncertainty + volatility_uncertainty + trend_uncertainty
        total_uncertainty = np.clip(total_uncertainty, 0.05, 0.40)  # 5-40% range
        
        # Generate confidence interval
        uncertainty_dollars = total_uncertainty * current_price
        confidence_lower = predicted_price - 1.96 * uncertainty_dollars
        confidence_upper = predicted_price + 1.96 * uncertainty_dollars
        
        # Calculate interval width percentage
        interval_width_pct = ((confidence_upper - confidence_lower) / current_price) * 100
        
        # Additional realism: occasionally generate wider intervals
        if np.random.random() < 0.1:  # 10% chance of high uncertainty
            interval_width_pct *= 1.5
            confidence_lower = predicted_price - (interval_width_pct/100) * current_price / 2
            confidence_upper = predicted_price + (interval_width_pct/100) * current_price / 2
        
        prediction = {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'predicted_price': float(predicted_price),
            'confidence_lower': float(confidence_lower),
            'confidence_upper': float(confidence_upper),
            'uncertainty': float(total_uncertainty),
            'interval_width_pct': float(interval_width_pct),
            'price_change_pct': float((predicted_price - current_price) / current_price * 100),
            'model_version': self.model_version,
            'prediction_count': self.prediction_count,
            'market_volatility': float(current_volatility),
            'trend_memory': float(self.trend_memory),
            'monte_carlo_samples': 100,  # Simulated
            'confidence_interval': 0.95
        }
        
        logger.debug(f"Mock prediction #{self.prediction_count}: "
                    f"${predicted_price:,.0f} ({prediction['price_change_pct']:+.2f}%) "
                    f"CI: {interval_width_pct:.1f}%")
        
        return prediction
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'MockBayesianPredictor',
            'model_version': self.model_version,
            'sequence_length': self.sequence_length,
            'base_uncertainty': self.base_uncertainty,
            'prediction_skill': self.prediction_skill,
            'predictions_made': self.prediction_count,
            'is_trained': True,  # Mock model is always "trained"
            'parameters': 50000,  # Fake parameter count
            'architecture': 'Mock LSTM (32->16->8) + Monte Carlo Dropout'
        }

class MockPredictionService:
    """
    Mock prediction service that provides the same interface as PredictionService.
    """
    
    def __init__(self, model_config: Any, enable_caching: bool = True):
        """Initialize mock prediction service."""
        self.config = model_config
        self.enable_caching = enable_caching
        self.predictor = MockBayesianPredictor(
            sequence_length=getattr(model_config, 'sequence_length', 60),
            base_uncertainty=0.15,
            prediction_skill=0.65
        )
        self.cache = {}
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0
        }
        self.is_initialized = True
        
        logger.info("MockPredictionService initialized")
    
    def initialize(self) -> bool:
        """Initialize service (always succeeds for mock)."""
        return True
    
    def predict(self, price_data: List[Dict], store_prediction: bool = True) -> Optional[Dict[str, Any]]:
        """Generate prediction using mock predictor."""
        try:
            self.stats['total_predictions'] += 1
            
            # Check cache
            cache_key = str(hash(str(price_data[-5:])))  # Simple cache key
            if self.enable_caching and cache_key in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[cache_key]
            else:
                self.stats['cache_misses'] += 1
            
            # Generate prediction
            prediction = self.predictor.predict_with_uncertainty(price_data)
            
            # Cache result
            if self.enable_caching:
                self.cache[cache_key] = prediction
                # Simple cache cleanup
                if len(self.cache) > 100:
                    self.cache.pop(next(iter(self.cache)))
            
            return prediction
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Mock prediction error: {e}")
            return None
    
    def check_model_health(self) -> Dict[str, Any]:
        """Check service health."""
        return {
            'is_initialized': self.is_initialized,
            'model_available': True,
            'model_version': self.predictor.model_version,
            'prediction_stats': self.stats.copy(),
            'cache_enabled': self.enable_caching,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            'service_info': {
                'type': 'MockPredictionService',
                'initialized': self.is_initialized,
                'model_version': self.predictor.model_version
            },
            'prediction_stats': self.stats.copy(),
            'model_info': self.predictor.get_model_info(),
            'performance_config': {
                'sequence_length': self.predictor.sequence_length,
                'base_uncertainty': self.predictor.base_uncertainty,
                'prediction_skill': self.predictor.prediction_skill,
                'caching_enabled': self.enable_caching
            }
        }

class MockTradingPredictor:
    """
    Mock trading predictor that interfaces with the trading engine.
    """
    
    def __init__(self, prediction_service: MockPredictionService):
        """Initialize mock trading predictor."""
        self.prediction_service = prediction_service
        self.prediction_history = []
        
    def get_prediction_for_trading(self, 
                                  current_price_data: Dict[str, Any],
                                  historical_data: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
        """Get prediction formatted for trading engine."""
        try:
            # Combine historical and current data
            if historical_data is None:
                historical_data = []
            
            price_data = historical_data + [current_price_data]
            
            # Get prediction
            prediction = self.prediction_service.predict(price_data)
            
            if not prediction:
                return None
            
            # Calculate trading confidence from uncertainty
            interval_width_pct = prediction['interval_width_pct']
            if interval_width_pct <= 15:
                confidence = 0.85
            elif interval_width_pct <= 20:
                confidence = 0.65
            elif interval_width_pct <= 25:
                confidence = 0.45
            else:
                confidence = 0.25
            
            # Format for trading engine
            trading_data = {
                'timestamp': prediction['timestamp'],
                'actual_price': prediction['current_price'],
                'predicted_price': prediction['predicted_price'],
                'interval_width_pct': prediction['interval_width_pct'],
                'confidence': confidence,
                'uncertainty_pct': prediction['interval_width_pct'],
                'price_change_pct': prediction['price_change_pct'],
                'model_version': prediction['model_version']
            }
            
            self.prediction_history.append(trading_data)
            return trading_data
            
        except Exception as e:
            logger.error(f"Error in mock trading predictor: {e}")
            return None