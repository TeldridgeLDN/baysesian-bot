"""
Adaptive Parameter Adjustment System for Bayesian Crypto Trading Bot.

This module implements intelligent parameter adjustment based on:
- Performance feedback from backtesting insights
- Real-time trading performance metrics
- Market volatility and regime changes
- Risk management constraints
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class AdjustmentReason(Enum):
    """Reasons for parameter adjustments."""
    LOW_WIN_RATE = "low_win_rate"
    HIGH_UNCERTAINTY = "high_uncertainty"
    LOW_FREQUENCY = "low_frequency"
    HIGH_DRAWDOWN = "high_drawdown"
    OVERTRADING = "overtrading"
    POOR_SHARPE = "poor_sharpe"
    MARKET_REGIME_CHANGE = "market_regime_change"
    MANUAL_OVERRIDE = "manual_override"

@dataclass
class ParameterBounds:
    """Safe bounds for parameter adjustments based on backtesting insights."""
    # Confidence threshold bounds (key insight: 0.40-0.50 is optimal)
    confidence_threshold_min: float = 0.30
    confidence_threshold_max: float = 0.65
    confidence_threshold_optimal: float = 0.45
    
    # Price change bounds (insight: 1.0-1.5% is optimal)
    min_price_change_pct_min: float = 0.8
    min_price_change_pct_max: float = 2.5
    min_price_change_pct_optimal: float = 1.2
    
    # Uncertainty tolerance bounds (insight: 18-22% is optimal)
    max_interval_width_pct_min: float = 15.0
    max_interval_width_pct_max: float = 30.0
    max_interval_width_pct_optimal: float = 20.0
    
    # Position size bounds
    max_position_size_min: float = 0.05
    max_position_size_max: float = 0.15
    max_position_size_optimal: float = 0.10
    
    # Risk management bounds
    max_drawdown_min: float = 0.08
    max_drawdown_max: float = 0.20
    max_drawdown_optimal: float = 0.15

@dataclass
class ParameterAdjustment:
    """Record of a parameter adjustment."""
    timestamp: datetime
    parameter_name: str
    old_value: float
    new_value: float
    reason: AdjustmentReason
    confidence_score: float  # 0-1, how confident we are in this adjustment
    expected_impact: str
    auto_applied: bool = False

class AdaptiveParameterManager:
    """
    Manages adaptive parameter adjustments for the trading engine.
    
    Key principles:
    1. Conservative adjustments based on proven backtesting insights
    2. Safety bounds to prevent extreme parameter values
    3. Gradual adjustments with cooling periods
    4. Performance validation before applying changes
    """
    
    def __init__(self, bounds: Optional[ParameterBounds] = None):
        """Initialize adaptive parameter manager."""
        self.bounds = bounds or ParameterBounds()
        self.adjustment_history: List[ParameterAdjustment] = []
        self.cooling_periods: Dict[str, datetime] = {}
        self.performance_baseline = None
        self.adjustment_sensitivity = 0.7  # How aggressive adjustments are (0-1)
        
        # Minimum time between adjustments for each parameter (hours)
        self.cooling_period_hours = {
            'confidence_threshold': 24,
            'min_price_change_pct': 12,
            'max_interval_width_pct': 12,
            'max_position_size': 48,  # More conservative for position sizing
            'max_drawdown': 72  # Very conservative for risk limits
        }
        
        logger.info("AdaptiveParameterManager initialized with backtesting-derived bounds")
    
    def suggest_adjustments(self, performance_metrics: Dict[str, Any], 
                           current_params: Dict[str, Any]) -> List[ParameterAdjustment]:
        """
        Suggest parameter adjustments based on performance metrics.
        
        Args:
            performance_metrics: Current trading performance data
            current_params: Current trading parameter values
            
        Returns:
            List of suggested parameter adjustments
        """
        suggestions = []
        
        if not performance_metrics:
            return suggestions
        
        logger.info(f"Analyzing performance for parameter adjustments: {performance_metrics}")
        
        # 1. Win Rate Analysis (Target: 50-60% based on backtesting)
        win_rate = performance_metrics.get('win_rate_pct', 0)
        if win_rate < 45.0:
            suggestion = self._suggest_confidence_adjustment(
                current_params.get('confidence_threshold', 0.45),
                target_change=-0.05,
                reason=AdjustmentReason.LOW_WIN_RATE,
                confidence=0.8
            )
            if suggestion:
                suggestions.append(suggestion)
        
        # 2. Trading Frequency Analysis (Target: 1-3 trades per day)
        trades_per_day = performance_metrics.get('trades_per_day', 0)
        if trades_per_day < 0.5:
            # Too few trades - make parameters more aggressive
            suggestion = self._suggest_price_change_adjustment(
                current_params.get('min_price_change_pct', 1.2),
                target_change=-0.2,
                reason=AdjustmentReason.LOW_FREQUENCY,
                confidence=0.7
            )
            if suggestion:
                suggestions.append(suggestion)
        elif trades_per_day > 5.0:
            # Too many trades - make parameters more conservative
            suggestion = self._suggest_price_change_adjustment(
                current_params.get('min_price_change_pct', 1.2),
                target_change=+0.3,
                reason=AdjustmentReason.OVERTRADING,
                confidence=0.8
            )
            if suggestion:
                suggestions.append(suggestion)
        
        # 3. Uncertainty Analysis (Target: 18-22% interval width)
        avg_uncertainty = performance_metrics.get('avg_uncertainty_pct', 0)
        if avg_uncertainty > 25.0:
            suggestion = self._suggest_uncertainty_adjustment(
                current_params.get('max_interval_width_pct', 20.0),
                target_change=+2.0,
                reason=AdjustmentReason.HIGH_UNCERTAINTY,
                confidence=0.6
            )
            if suggestion:
                suggestions.append(suggestion)
        
        # 4. Drawdown Analysis (Target: <15%)
        max_drawdown = performance_metrics.get('max_drawdown_pct', 0)
        if max_drawdown > 12.0:
            suggestion = self._suggest_position_size_adjustment(
                current_params.get('max_position_size', 0.10),
                target_change=-0.02,
                reason=AdjustmentReason.HIGH_DRAWDOWN,
                confidence=0.9
            )
            if suggestion:
                suggestions.append(suggestion)
        
        # 5. Sharpe Ratio Analysis (Target: >1.0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        if sharpe_ratio < 0.8:
            # Poor risk-adjusted returns - be more selective
            suggestion = self._suggest_confidence_adjustment(
                current_params.get('confidence_threshold', 0.45),
                target_change=+0.05,
                reason=AdjustmentReason.POOR_SHARPE,
                confidence=0.7
            )
            if suggestion:
                suggestions.append(suggestion)
        
        logger.info(f"Generated {len(suggestions)} parameter adjustment suggestions")
        return suggestions
    
    def _suggest_confidence_adjustment(self, current_value: float, target_change: float,
                                     reason: AdjustmentReason, confidence: float) -> Optional[ParameterAdjustment]:
        """Suggest confidence threshold adjustment."""
        if not self._can_adjust_parameter('confidence_threshold'):
            return None
        
        new_value = current_value + (target_change * self.adjustment_sensitivity)
        new_value = max(self.bounds.confidence_threshold_min, 
                       min(self.bounds.confidence_threshold_max, new_value))
        
        if abs(new_value - current_value) < 0.01:  # Minimum meaningful change
            return None
        
        return ParameterAdjustment(
            timestamp=datetime.now(),
            parameter_name='confidence_threshold',
            old_value=current_value,
            new_value=new_value,
            reason=reason,
            confidence_score=confidence,
            expected_impact=self._describe_confidence_impact(new_value - current_value)
        )
    
    def _suggest_price_change_adjustment(self, current_value: float, target_change: float,
                                       reason: AdjustmentReason, confidence: float) -> Optional[ParameterAdjustment]:
        """Suggest minimum price change adjustment."""
        if not self._can_adjust_parameter('min_price_change_pct'):
            return None
        
        new_value = current_value + (target_change * self.adjustment_sensitivity)
        new_value = max(self.bounds.min_price_change_pct_min,
                       min(self.bounds.min_price_change_pct_max, new_value))
        
        if abs(new_value - current_value) < 0.1:  # Minimum meaningful change
            return None
        
        return ParameterAdjustment(
            timestamp=datetime.now(),
            parameter_name='min_price_change_pct',
            old_value=current_value,
            new_value=new_value,
            reason=reason,
            confidence_score=confidence,
            expected_impact=self._describe_price_change_impact(new_value - current_value)
        )
    
    def _suggest_uncertainty_adjustment(self, current_value: float, target_change: float,
                                      reason: AdjustmentReason, confidence: float) -> Optional[ParameterAdjustment]:
        """Suggest uncertainty tolerance adjustment."""
        if not self._can_adjust_parameter('max_interval_width_pct'):
            return None
        
        new_value = current_value + (target_change * self.adjustment_sensitivity)
        new_value = max(self.bounds.max_interval_width_pct_min,
                       min(self.bounds.max_interval_width_pct_max, new_value))
        
        if abs(new_value - current_value) < 0.5:  # Minimum meaningful change
            return None
        
        return ParameterAdjustment(
            timestamp=datetime.now(),
            parameter_name='max_interval_width_pct',
            old_value=current_value,
            new_value=new_value,
            reason=reason,
            confidence_score=confidence,
            expected_impact=self._describe_uncertainty_impact(new_value - current_value)
        )
    
    def _suggest_position_size_adjustment(self, current_value: float, target_change: float,
                                        reason: AdjustmentReason, confidence: float) -> Optional[ParameterAdjustment]:
        """Suggest position size adjustment."""
        if not self._can_adjust_parameter('max_position_size'):
            return None
        
        new_value = current_value + (target_change * self.adjustment_sensitivity)
        new_value = max(self.bounds.max_position_size_min,
                       min(self.bounds.max_position_size_max, new_value))
        
        if abs(new_value - current_value) < 0.005:  # Minimum meaningful change
            return None
        
        return ParameterAdjustment(
            timestamp=datetime.now(),
            parameter_name='max_position_size',
            old_value=current_value,
            new_value=new_value,
            reason=reason,
            confidence_score=confidence,
            expected_impact=self._describe_position_size_impact(new_value - current_value)
        )
    
    def _can_adjust_parameter(self, param_name: str) -> bool:
        """Check if parameter is within cooling period."""
        last_adjustment = self.cooling_periods.get(param_name)
        if last_adjustment is None:
            return True
        
        cooling_hours = self.cooling_period_hours.get(param_name, 24)
        time_since_last = datetime.now() - last_adjustment
        return time_since_last > timedelta(hours=cooling_hours)
    
    def _describe_confidence_impact(self, change: float) -> str:
        """Describe expected impact of confidence threshold change."""
        if change > 0:
            return f"Higher confidence requirement (+{change:.3f}) → fewer but higher quality trades"
        else:
            return f"Lower confidence requirement ({change:.3f}) → more trades with slightly lower quality"
    
    def _describe_price_change_impact(self, change: float) -> str:
        """Describe expected impact of price change threshold change."""
        if change > 0:
            return f"Higher price change requirement (+{change:.1f}%) → fewer trades, focus on larger moves"
        else:
            return f"Lower price change requirement ({change:.1f}%) → more trades including smaller moves"
    
    def _describe_uncertainty_impact(self, change: float) -> str:
        """Describe expected impact of uncertainty tolerance change."""
        if change > 0:
            return f"Higher uncertainty tolerance (+{change:.1f}%) → more trades despite higher uncertainty"
        else:
            return f"Lower uncertainty tolerance ({change:.1f}%) → fewer trades, only high certainty"
    
    def _describe_position_size_impact(self, change: float) -> str:
        """Describe expected impact of position size change."""
        if change > 0:
            return f"Larger position sizes (+{change:.1f}%) → higher risk/reward per trade"
        else:
            return f"Smaller position sizes ({change:.1f}%) → lower risk, more conservative approach"
    
    def apply_adjustment(self, adjustment: ParameterAdjustment) -> bool:
        """
        Apply a parameter adjustment and record it.
        
        Args:
            adjustment: The adjustment to apply
            
        Returns:
            True if applied successfully
        """
        try:
            # Record the adjustment
            adjustment.auto_applied = True
            self.adjustment_history.append(adjustment)
            
            # Set cooling period
            self.cooling_periods[adjustment.parameter_name] = adjustment.timestamp
            
            logger.info(f"Applied parameter adjustment: {adjustment.parameter_name} "
                       f"{adjustment.old_value:.3f} → {adjustment.new_value:.3f} "
                       f"(reason: {adjustment.reason.value})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply parameter adjustment: {e}")
            return False
    
    def get_adjustment_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent parameter adjustment history."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_adjustments = [
            adj for adj in self.adjustment_history 
            if adj.timestamp > cutoff
        ]
        
        return [asdict(adj) for adj in recent_adjustments]
    
    def get_current_bounds(self) -> Dict[str, Any]:
        """Get current parameter bounds for validation."""
        return asdict(self.bounds)
    
    def reset_to_optimal(self) -> Dict[str, float]:
        """Reset all parameters to optimal values from backtesting."""
        optimal_params = {
            'confidence_threshold': self.bounds.confidence_threshold_optimal,
            'min_price_change_pct': self.bounds.min_price_change_pct_optimal,
            'max_interval_width_pct': self.bounds.max_interval_width_pct_optimal,
            'max_position_size': self.bounds.max_position_size_optimal,
            'max_drawdown': self.bounds.max_drawdown_optimal
        }
        
        logger.info(f"Reset parameters to optimal values: {optimal_params}")
        return optimal_params
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against safety bounds.
        
        Args:
            params: Parameter dictionary to validate
            
        Returns:
            Dictionary with validation results and corrected values
        """
        validation_results = {
            'valid': True,
            'corrections': {},
            'warnings': []
        }
        
        # Validate confidence threshold
        confidence = params.get('confidence_threshold')
        if confidence is not None:
            if confidence < self.bounds.confidence_threshold_min:
                validation_results['corrections']['confidence_threshold'] = self.bounds.confidence_threshold_min
                validation_results['warnings'].append(f"Confidence threshold {confidence:.3f} below minimum {self.bounds.confidence_threshold_min:.3f}")
                validation_results['valid'] = False
            elif confidence > self.bounds.confidence_threshold_max:
                validation_results['corrections']['confidence_threshold'] = self.bounds.confidence_threshold_max
                validation_results['warnings'].append(f"Confidence threshold {confidence:.3f} above maximum {self.bounds.confidence_threshold_max:.3f}")
                validation_results['valid'] = False
        
        # Validate other parameters similarly...
        price_change = params.get('min_price_change_pct')
        if price_change is not None:
            if price_change < self.bounds.min_price_change_pct_min:
                validation_results['corrections']['min_price_change_pct'] = self.bounds.min_price_change_pct_min
                validation_results['warnings'].append(f"Price change threshold {price_change:.1f}% below minimum {self.bounds.min_price_change_pct_min:.1f}%")
                validation_results['valid'] = False
            elif price_change > self.bounds.min_price_change_pct_max:
                validation_results['corrections']['min_price_change_pct'] = self.bounds.min_price_change_pct_max
                validation_results['warnings'].append(f"Price change threshold {price_change:.1f}% above maximum {self.bounds.min_price_change_pct_max:.1f}%")
                validation_results['valid'] = False
        
        return validation_results