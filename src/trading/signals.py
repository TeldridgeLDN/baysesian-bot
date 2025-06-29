"""
Trading signal generation logic based on Bayesian LSTM predictions.
Handles signal strength calculation and trade decision making.
"""

import logging
from typing import Tuple, Dict, Any, Literal
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SignalType = Literal['LONG', 'SHORT', 'HOLD']

class TradingConfig(BaseModel):
    initial_capital: float = 10000.0
    position_size: float = 0.1
    stop_loss: float = 0.05
    take_profit: float = 0.1
    min_price_change_pct: float = 2.0
    max_interval_width_pct: float = 3.0
    confidence_threshold: float = 0.60
    paper_trading: bool = True
    trading_fee: float = 0.001  # 0.1% trading fee
    slippage: float = 0.0002   # 0.02% slippage

class TradingSignalGenerator:
    """Generates trading signals based on model predictions and confidence."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.min_price_change = self.config.min_price_change_pct / 100.0  # Convert to decimal
        
    def generate_signal(self, current_price: float, predicted_price: float) -> SignalType:
        """Generate trading signal based on price prediction and confidence."""
        # Calculate price change percentage
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Generate signal based on predicted price movement
        if price_change_pct > self.min_price_change:
            return 'LONG'
        elif price_change_pct < -self.min_price_change:
            return 'SHORT'
        
        return 'HOLD'
        
    def calculate_confidence_score(self, price_change_pct: float, 
                                 interval_width_pct: float) -> float:
        """Calculate confidence score using risk-graded uncertainty scaling."""
        # Normalize price change relative to minimum required change
        price_change_abs = abs(price_change_pct * 100)  # Convert to percentage
        price_score = min(1.0, price_change_abs / self.config.min_price_change_pct)
        
        # Risk-graded uncertainty scaling instead of binary cutoff
        uncertainty_penalty = self._calculate_uncertainty_penalty(interval_width_pct)
        
        # Weight price signal and uncertainty penalty
        confidence_score = price_score * uncertainty_penalty
        
        # Apply additional penalty for smaller moves
        if price_change_abs < self.config.min_price_change_pct * 1.2:
            confidence_score *= 0.7  # Reduce confidence for marginal signals
            
        return confidence_score
    
    def _calculate_uncertainty_penalty(self, interval_width_pct: float) -> float:
        """Calculate risk-graded uncertainty penalty for position sizing.
        
        Returns a scaling factor (0.0 to 1.0) based on prediction uncertainty:
        - Low uncertainty (5-10%): 100% position scaling
        - Moderate uncertainty (10-15%): 75% position scaling  
        - High uncertainty (15-20%): 50% position scaling
        - Very high uncertainty (20-25%): 25% position scaling
        - Extreme uncertainty (>25%): 0% (no trading)
        """
        if interval_width_pct <= 5.0:
            return 1.0  # Perfect confidence
        elif interval_width_pct <= 10.0:
            # Linear scale from 1.0 to 0.9 (5%-10%)
            return 1.0 - 0.1 * (interval_width_pct - 5.0) / 5.0
        elif interval_width_pct <= 15.0:
            # Linear scale from 0.9 to 0.75 (10%-15%)  
            return 0.9 - 0.15 * (interval_width_pct - 10.0) / 5.0
        elif interval_width_pct <= 20.0:
            # Linear scale from 0.75 to 0.5 (15%-20%)
            return 0.75 - 0.25 * (interval_width_pct - 15.0) / 5.0
        elif interval_width_pct <= 25.0:
            # Linear scale from 0.5 to 0.25 (20%-25%)
            return 0.5 - 0.25 * (interval_width_pct - 20.0) / 5.0
        else:
            # Extreme uncertainty - no trading
            return 0.0
        
    def get_position_size_ratio(self, confidence_score: float, 
                              current_drawdown: float = 0.0) -> float:
        """Calculate position size ratio based on confidence and risk."""
        # Only take positions if confidence is above threshold
        if confidence_score < self.config.confidence_threshold:
            return 0.0
            
        # Base position size on confidence
        position_ratio = confidence_score
        
        # Reduce position size when in drawdown
        if current_drawdown > 0:
            drawdown_factor = 1.0 - (current_drawdown / self.config.stop_loss)
            drawdown_factor = max(0.2, drawdown_factor)  # Never go below 20% of normal size
            position_ratio *= drawdown_factor
            
        return position_ratio 