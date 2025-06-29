"""
Paper Trading System with Live Parameter Adaptation.

This module implements a comprehensive paper trading environment that:
- Simulates real trading with live market data
- Enables safe parameter adaptation and strategy testing
- Provides detailed performance analytics
- Supports A/B testing of different parameter sets
- Tracks adaptation success rates
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)

class PaperTradingMode(Enum):
    """Different paper trading operation modes."""
    SIMULATION = "simulation"  # Pure simulation with historical data
    LIVE_PAPER = "live_paper"  # Live data, paper trades
    ADAPTATION_TEST = "adaptation_test"  # Testing parameter adaptations
    A_B_TEST = "a_b_test"  # Comparing multiple parameter sets

@dataclass
class PaperPosition:
    """Represents a paper trading position."""
    id: str
    timestamp: datetime
    entry_price: float
    position_size: float
    direction: str  # 'long' or 'short'
    confidence: float
    uncertainty_pct: float
    parameters_used: Dict[str, Any]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    success: Optional[bool] = None

@dataclass
class ParameterTestResult:
    """Results from testing a parameter set."""
    parameter_set_id: str
    parameters: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    total_trades: int
    winning_trades: int
    win_rate_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_duration_hours: float
    avg_confidence: float
    avg_uncertainty_pct: float
    trades_per_day: float
    parameter_score: float  # Overall parameter effectiveness score

@dataclass
class AdaptationEvent:
    """Records a parameter adaptation event."""
    timestamp: datetime
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    trigger_reason: str
    performance_before: Dict[str, Any]
    adaptation_confidence: float
    expected_improvement: str

class PaperTradingEngine:
    """
    Paper trading engine with live parameter adaptation capabilities.
    
    Features:
    - Real-time simulation with live market data
    - Safe parameter adaptation testing
    - Performance comparison across parameter sets
    - Automatic adaptation based on performance feedback
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 mode: PaperTradingMode = PaperTradingMode.LIVE_PAPER):
        """Initialize paper trading engine."""
        self.mode = mode
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Trading state
        self.positions: List[PaperPosition] = []
        self.closed_positions: List[PaperPosition] = []
        self.current_position: Optional[PaperPosition] = None
        
        # Parameter testing
        self.parameter_sets: Dict[str, Dict[str, Any]] = {}
        self.parameter_test_results: List[ParameterTestResult] = []
        self.adaptation_history: List[AdaptationEvent] = []
        
        # Performance tracking
        self.performance_snapshots: List[Dict[str, Any]] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Adaptation settings
        self.adaptation_enabled = True
        self.adaptation_frequency_hours = 6
        self.last_adaptation_check = datetime.now()
        self.min_trades_for_adaptation = 10
        
        # A/B testing
        self.ab_test_active = False
        self.ab_test_sets: Dict[str, Dict[str, Any]] = {}
        self.ab_test_allocation: Dict[str, float] = {}
        
        logger.info(f"PaperTradingEngine initialized: mode={mode.value}, capital=${initial_capital:,.0f}")
    
    def add_parameter_set(self, set_id: str, parameters: Dict[str, Any]) -> bool:
        """Add a parameter set for testing."""
        try:
            self.parameter_sets[set_id] = deepcopy(parameters)
            logger.info(f"Added parameter set '{set_id}': {parameters}")
            return True
        except Exception as e:
            logger.error(f"Error adding parameter set: {e}")
            return False
    
    def start_ab_test(self, parameter_sets: Dict[str, Dict[str, Any]], 
                     allocation: Optional[Dict[str, float]] = None) -> bool:
        """
        Start A/B testing multiple parameter sets.
        
        Args:
            parameter_sets: Dictionary of parameter set ID to parameters
            allocation: Traffic allocation per set (must sum to 1.0)
        """
        try:
            if allocation is None:
                # Equal allocation
                n_sets = len(parameter_sets)
                allocation = {set_id: 1.0/n_sets for set_id in parameter_sets.keys()}
            
            # Validate allocation
            if abs(sum(allocation.values()) - 1.0) > 0.001:
                raise ValueError("Allocation must sum to 1.0")
            
            self.ab_test_sets = deepcopy(parameter_sets)
            self.ab_test_allocation = allocation
            self.ab_test_active = True
            self.mode = PaperTradingMode.A_B_TEST
            
            logger.info(f"Started A/B test with {len(parameter_sets)} parameter sets")
            for set_id, params in parameter_sets.items():
                logger.info(f"  {set_id} ({allocation[set_id]*100:.1f}%): {params}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting A/B test: {e}")
            return False
    
    def execute_paper_trade(self, signal: str, current_price: float, 
                           prediction_data: Dict[str, Any],
                           parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a paper trade based on trading signal.
        
        Args:
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
            current_price: Current market price
            prediction_data: ML prediction data with confidence/uncertainty
            parameters: Current trading parameters
            
        Returns:
            Trade result dictionary or None
        """
        try:
            timestamp = datetime.now()
            
            # Close existing position if selling
            if signal == 'SELL' and self.current_position:
                return self._close_position(current_price, "SELL_SIGNAL", timestamp)
            
            # Open new position if buying
            if signal == 'BUY' and not self.current_position:
                return self._open_position(current_price, prediction_data, parameters, timestamp)
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return None
    
    def _open_position(self, entry_price: float, prediction_data: Dict[str, Any],
                      parameters: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Open a new paper trading position."""
        try:
            # Calculate position size based on parameters
            max_position_size = parameters.get('max_position_size', 0.10)
            confidence = prediction_data.get('confidence', 0.5)
            
            # Risk-adjusted position sizing
            if parameters.get('risk_grading_enabled', True):
                # Higher confidence = larger position (within limits)
                confidence_multiplier = min(confidence / 0.5, 1.5)  # Cap at 1.5x
                position_size_pct = max_position_size * confidence_multiplier
            else:
                position_size_pct = max_position_size
            
            # Calculate position value
            position_value = self.current_capital * position_size_pct
            position_size_btc = position_value / entry_price
            
            # Create position
            position = PaperPosition(
                id=f"paper_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                timestamp=timestamp,
                entry_price=entry_price,
                position_size=position_size_btc,
                direction='long',
                confidence=confidence,
                uncertainty_pct=prediction_data.get('interval_width_pct', 0),
                parameters_used=deepcopy(parameters)
            )
            
            # Set stop loss and take profit
            stop_loss_pct = parameters.get('stop_loss_multiplier', 2.0) * prediction_data.get('interval_width_pct', 5.0) / 100
            take_profit_ratio = parameters.get('take_profit_ratio', 1.5)
            
            position.stop_loss = entry_price * (1 - stop_loss_pct)
            position.take_profit = entry_price * (1 + stop_loss_pct * take_profit_ratio)
            
            self.current_position = position
            self.positions.append(position)
            
            # Update capital (reserve for position)
            self.current_capital -= position_value
            
            result = {
                'action': 'OPEN_POSITION',
                'position_id': position.id,
                'entry_price': entry_price,
                'position_size': position_size_btc,
                'position_value': position_value,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'confidence': confidence,
                'timestamp': timestamp
            }
            
            logger.info(f"Opened paper position: {position.id} @ ${entry_price:,.0f}, "
                       f"size: {position_size_btc:.6f} BTC (${position_value:,.0f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error opening paper position: {e}")
            return None
    
    def _close_position(self, exit_price: float, exit_reason: str, 
                       timestamp: datetime) -> Dict[str, Any]:
        """Close the current paper trading position."""
        try:
            if not self.current_position:
                return None
            
            position = self.current_position
            
            # Calculate P&L
            position_value = position.position_size * exit_price
            original_value = position.position_size * position.entry_price
            pnl = position_value - original_value
            pnl_pct = (pnl / original_value) * 100
            
            # Update position
            position.exit_timestamp = timestamp
            position.exit_price = exit_price
            position.exit_reason = exit_reason
            position.pnl = pnl
            position.pnl_pct = pnl_pct
            position.success = pnl > 0
            
            # Update capital
            self.current_capital += position_value
            
            # Move to closed positions
            self.closed_positions.append(position)
            self.current_position = None
            
            # Record equity point
            total_equity = self.get_total_equity(exit_price)
            self.equity_curve.append((timestamp, total_equity))
            
            result = {
                'action': 'CLOSE_POSITION',
                'position_id': position.id,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'success': position.success,
                'duration_hours': (timestamp - position.timestamp).total_seconds() / 3600,
                'total_equity': total_equity,
                'timestamp': timestamp
            }
            
            logger.info(f"Closed paper position: {position.id} @ ${exit_price:,.0f}, "
                       f"P&L: ${pnl:,.0f} ({pnl_pct:+.2f}%), reason: {exit_reason}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error closing paper position: {e}")
            return None
    
    def check_position_exits(self, current_price: float) -> Optional[Dict[str, Any]]:
        """Check if current position should be closed due to stop loss/take profit."""
        if not self.current_position:
            return None
        
        position = self.current_position
        timestamp = datetime.now()
        
        # Check stop loss
        if position.stop_loss and current_price <= position.stop_loss:
            return self._close_position(current_price, "STOP_LOSS", timestamp)
        
        # Check take profit
        if position.take_profit and current_price >= position.take_profit:
            return self._close_position(current_price, "TAKE_PROFIT", timestamp)
        
        # Check position timeout
        position_age_hours = (timestamp - position.timestamp).total_seconds() / 3600
        timeout_hours = position.parameters_used.get('position_timeout_hours', 24)
        
        if position_age_hours >= timeout_hours:
            return self._close_position(current_price, "TIMEOUT", timestamp)
        
        return None
    
    def get_total_equity(self, current_price: float) -> float:
        """Calculate total equity including open positions."""
        equity = self.current_capital
        
        if self.current_position:
            position_value = self.current_position.position_size * current_price
            equity += position_value
        
        return equity
    
    def get_performance_summary(self, current_price: float) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            total_trades = len(self.closed_positions)
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'equity': self.get_total_equity(current_price),
                    'total_return_pct': 0.0,
                    'message': 'No completed trades yet'
                }
            
            # Calculate metrics
            winning_trades = sum(1 for p in self.closed_positions if p.success)
            win_rate = (winning_trades / total_trades) * 100
            
            total_pnl = sum(p.pnl for p in self.closed_positions)
            total_return_pct = (total_pnl / self.initial_capital) * 100
            
            # Calculate drawdown
            equity_values = [eq for _, eq in self.equity_curve]
            if equity_values:
                peak = self.initial_capital
                max_drawdown = 0
                for equity in equity_values:
                    if equity > peak:
                        peak = equity
                    drawdown = (peak - equity) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                max_drawdown = 0
            
            # Calculate Sharpe ratio (simplified)
            if len(self.closed_positions) > 1:
                returns = [p.pnl_pct for p in self.closed_positions]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Trading frequency
            if self.closed_positions:
                first_trade = min(p.timestamp for p in self.closed_positions)
                days_trading = (datetime.now() - first_trade).days or 1
                trades_per_day = total_trades / days_trading
            else:
                trades_per_day = 0
            
            # Average metrics
            avg_confidence = np.mean([p.confidence for p in self.closed_positions])
            avg_uncertainty = np.mean([p.uncertainty_pct for p in self.closed_positions])
            
            current_equity = self.get_total_equity(current_price)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate_pct': win_rate,
                'total_return_pct': total_return_pct,
                'total_pnl': total_pnl,
                'current_equity': current_equity,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'trades_per_day': trades_per_day,
                'avg_confidence': avg_confidence,
                'avg_uncertainty_pct': avg_uncertainty,
                'equity_curve_points': len(self.equity_curve),
                'current_position': self.current_position is not None
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {'error': str(e)}
    
    def check_adaptation_trigger(self, current_performance: Dict[str, Any],
                                parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if parameter adaptation should be triggered.
        
        Args:
            current_performance: Current trading performance metrics
            parameters: Current parameter set
            
        Returns:
            Adaptation suggestion or None
        """
        if not self.adaptation_enabled:
            return None
        
        # Check timing
        time_since_last = datetime.now() - self.last_adaptation_check
        if time_since_last.total_seconds() < self.adaptation_frequency_hours * 3600:
            return None
        
        # Need minimum trades for reliable adaptation
        if current_performance.get('total_trades', 0) < self.min_trades_for_adaptation:
            return None
        
        try:
            # Analyze performance for adaptation triggers
            win_rate = current_performance.get('win_rate_pct', 0)
            trades_per_day = current_performance.get('trades_per_day', 0)
            sharpe_ratio = current_performance.get('sharpe_ratio', 0)
            max_drawdown = current_performance.get('max_drawdown_pct', 0)
            
            adaptation_suggestions = []
            
            # Low win rate - reduce confidence threshold
            if win_rate < 45.0:
                new_confidence = max(0.35, parameters.get('confidence_threshold', 0.45) - 0.05)
                adaptation_suggestions.append({
                    'parameter': 'confidence_threshold',
                    'current_value': parameters.get('confidence_threshold', 0.45),
                    'suggested_value': new_confidence,
                    'reason': 'low_win_rate',
                    'confidence': 0.8
                })
            
            # Low trading frequency - more aggressive parameters
            if trades_per_day < 0.5:
                new_price_change = max(0.8, parameters.get('min_price_change_pct', 1.2) - 0.2)
                adaptation_suggestions.append({
                    'parameter': 'min_price_change_pct',
                    'current_value': parameters.get('min_price_change_pct', 1.2),
                    'suggested_value': new_price_change,
                    'reason': 'low_frequency',
                    'confidence': 0.7
                })
            
            # High drawdown - reduce position size
            if max_drawdown > 15.0:
                new_position_size = max(0.05, parameters.get('max_position_size', 0.10) - 0.02)
                adaptation_suggestions.append({
                    'parameter': 'max_position_size',
                    'current_value': parameters.get('max_position_size', 0.10),
                    'suggested_value': new_position_size,
                    'reason': 'high_drawdown',
                    'confidence': 0.9
                })
            
            if adaptation_suggestions:
                self.last_adaptation_check = datetime.now()
                return {
                    'timestamp': datetime.now(),
                    'trigger_performance': current_performance,
                    'suggestions': adaptation_suggestions,
                    'total_suggestions': len(adaptation_suggestions)
                }
            
            self.last_adaptation_check = datetime.now()
            return None
            
        except Exception as e:
            logger.error(f"Error checking adaptation trigger: {e}")
            return None
    
    def apply_parameter_adaptation(self, adaptation_data: Dict[str, Any],
                                  current_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter adaptation and record the event.
        
        Args:
            adaptation_data: Adaptation suggestions from check_adaptation_trigger
            current_parameters: Current parameter set
            
        Returns:
            New parameter set
        """
        try:
            new_parameters = deepcopy(current_parameters)
            applied_changes = []
            
            for suggestion in adaptation_data['suggestions']:
                param_name = suggestion['parameter']
                new_value = suggestion['suggested_value']
                
                # Apply high-confidence suggestions automatically
                if suggestion['confidence'] >= 0.8:
                    old_value = new_parameters.get(param_name, suggestion['current_value'])
                    new_parameters[param_name] = new_value
                    
                    applied_changes.append({
                        'parameter': param_name,
                        'old_value': old_value,
                        'new_value': new_value,
                        'reason': suggestion['reason'],
                        'confidence': suggestion['confidence']
                    })
            
            if applied_changes:
                # Record adaptation event
                adaptation_event = AdaptationEvent(
                    timestamp=datetime.now(),
                    old_parameters=deepcopy(current_parameters),
                    new_parameters=deepcopy(new_parameters),
                    trigger_reason=f"Performance-based adaptation: {len(applied_changes)} changes",
                    performance_before=adaptation_data['trigger_performance'],
                    adaptation_confidence=np.mean([c['confidence'] for c in applied_changes]),
                    expected_improvement=f"Applied {len(applied_changes)} parameter adjustments"
                )
                
                self.adaptation_history.append(adaptation_event)
                
                logger.info(f"Applied parameter adaptation: {len(applied_changes)} changes")
                for change in applied_changes:
                    logger.info(f"  {change['parameter']}: {change['old_value']:.3f} → {change['new_value']:.3f} "
                               f"({change['reason']}, confidence: {change['confidence']:.2f})")
                
                return new_parameters
            
            return current_parameters
            
        except Exception as e:
            logger.error(f"Error applying parameter adaptation: {e}")
            return current_parameters
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of parameter adaptation history."""
        try:
            if not self.adaptation_history:
                return {
                    'total_adaptations': 0,
                    'adaptation_enabled': self.adaptation_enabled,
                    'message': 'No adaptations performed yet'
                }
            
            recent_adaptations = [
                a for a in self.adaptation_history
                if a.timestamp > datetime.now() - timedelta(days=7)
            ]
            
            # Calculate adaptation effectiveness (simplified)
            adaptation_scores = []
            for i, adaptation in enumerate(self.adaptation_history[:-1]):
                # Compare performance before and after (would need more sophisticated tracking)
                adaptation_scores.append(adaptation.adaptation_confidence)
            
            avg_effectiveness = np.mean(adaptation_scores) if adaptation_scores else 0
            
            return {
                'total_adaptations': len(self.adaptation_history),
                'recent_adaptations': len(recent_adaptations),
                'adaptation_enabled': self.adaptation_enabled,
                'avg_effectiveness': avg_effectiveness,
                'last_adaptation': self.adaptation_history[-1].timestamp.isoformat(),
                'adaptation_frequency_hours': self.adaptation_frequency_hours,
                'min_trades_for_adaptation': self.min_trades_for_adaptation,
                'recent_adaptation_details': [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'reason': a.trigger_reason,
                        'confidence': a.adaptation_confidence,
                        'parameter_changes': len([k for k in a.new_parameters.keys() 
                                                if a.new_parameters[k] != a.old_parameters.get(k)])
                    }
                    for a in recent_adaptations[-5:]  # Last 5
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting adaptation summary: {e}")
            return {'error': str(e)}
    
    def export_trading_session(self) -> Dict[str, Any]:
        """Export complete trading session data for analysis."""
        try:
            return {
                'session_info': {
                    'mode': self.mode.value,
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'session_start': min(p.timestamp for p in self.closed_positions).isoformat() if self.closed_positions else None,
                    'session_end': datetime.now().isoformat()
                },
                'positions': [asdict(p) for p in self.closed_positions],
                'current_position': asdict(self.current_position) if self.current_position else None,
                'adaptations': [asdict(a) for a in self.adaptation_history],
                'equity_curve': [(ts.isoformat(), equity) for ts, equity in self.equity_curve],
                'parameter_sets_tested': self.parameter_sets,
                'performance_snapshots': self.performance_snapshots
            }
        except Exception as e:
            logger.error(f"Error exporting trading session: {e}")
            return {'error': str(e)}