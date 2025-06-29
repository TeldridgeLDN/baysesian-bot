from typing import Dict, Any, List, Literal, Optional
from datetime import datetime
import logging
from pydantic import BaseModel
from .portfolio import PortfolioManager
from .signals import TradingSignalGenerator
from .execution import TradeExecutor
from .adaptive_parameters import AdaptiveParameterManager, ParameterAdjustment, AdjustmentReason
from ..monitoring.performance_tracker import PerformanceTracker
from ..data.storage import DatabaseManager
from ..utils.config import MonitoringConfig

logger = logging.getLogger(__name__)

class TradingConfig(BaseModel):
    initial_capital: float = 10000.0
    position_size: float = 0.1
    stop_loss: float = 0.05
    take_profit: float = 0.1
    min_price_change_pct: float = 2.0
    max_interval_width_pct: float = 3.0
    confidence_threshold: float = 0.60
    paper_trading: bool = True
    stop_loss_type: Literal['none', 'fixed', 'trailing'] = 'fixed'
    trailing_stop_pct: float = 0.02
    trading_fee: float = 0.001  # 0.1% trading fee
    slippage: float = 0.0002   # 0.02% slippage

class TradingEngine:
    def __init__(self, config: TradingConfig, enable_performance_tracking: bool = True, 
                 db_manager: Optional[DatabaseManager] = None, 
                 monitoring_config: Optional[MonitoringConfig] = None,
                 enable_adaptive_parameters: bool = True):
        self.config = config
        self.portfolio = PortfolioManager(config)
        self.signal_generator = TradingSignalGenerator(config)
        self.executor = TradeExecutor(config)
        
        # Performance tracking setup
        self.enable_performance_tracking = enable_performance_tracking
        self.performance_tracker = None
        self.db_manager = db_manager
        
        # Adaptive parameter management
        self.enable_adaptive_parameters = enable_adaptive_parameters
        self.adaptive_manager = AdaptiveParameterManager() if enable_adaptive_parameters else None
        self.pending_adjustments: List[ParameterAdjustment] = []
        self.last_adjustment_check = datetime.now()
        
        if enable_performance_tracking:
            if monitoring_config is None:
                # Create default monitoring config
                monitoring_config = MonitoringConfig()
            
            self.performance_tracker = PerformanceTracker(monitoring_config)
            
            if db_manager is None:
                # Create default database manager
                self.db_manager = DatabaseManager()
            else:
                self.db_manager = db_manager
            
            logger.info("Performance tracking enabled for trading engine")
        else:
            logger.info("Performance tracking disabled for trading engine")
        
    def process_data_point(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data point and execute trades if necessary."""
        try:
            timestamp = data_point['timestamp']
            current_price = data_point['actual_price']
            predicted_price = data_point['predicted_price']

            # Update performance metrics before any action
            self.portfolio.update_performance_metrics(current_price)

            # Initialize result data
            result = {
                'timestamp': timestamp, 
                'price': current_price, 
                'signal': 'HOLD',
                'confidence': 0.0, 
                'portfolio_value': self.portfolio.get_total_value(current_price),
                'trade_executed': False,
                'trade_data': None
            }

            # 1. Check for stop-loss
            if self.portfolio.check_stop_loss(current_price):
                trade_result = self.executor.execute_trade(signal='CLOSE', price=current_price, portfolio=self.portfolio)
                if trade_result:
                    self.portfolio.update(trade_result)
                    result.update({
                        'signal': 'STOP_LOSS',
                        'confidence': 0.0,
                        'portfolio_value': self.portfolio.get_total_value(current_price),
                        'trade_executed': True,
                        'trade_data': trade_result
                    })
                    
                    # Record trade with performance tracker
                    if self.enable_performance_tracking and self.performance_tracker:
                        self._record_trade_performance(trade_result, current_price, data_point)
                    
                    return result

            # 2. If no stop-loss, proceed with signal generation
            price_change_pct = (predicted_price - current_price) / current_price
            interval_width_pct = data_point.get('interval_width_pct', 2.0)
            
            confidence_score = self.signal_generator.calculate_confidence_score(
                price_change_pct=price_change_pct, interval_width_pct=interval_width_pct
            )
            
            signal = self.signal_generator.generate_signal(
                current_price=current_price, predicted_price=predicted_price
            )
            
            position_ratio = self.signal_generator.get_position_size_ratio(
                confidence_score=confidence_score,
                current_drawdown=self.portfolio.max_drawdown
            )
            
            # Update result with signal data
            result.update({
                'signal': signal,
                'confidence': confidence_score,
                'portfolio_value': self.portfolio.get_total_value(current_price)
            })
            
            # 3. Execute trade if signal is not HOLD
            if signal != 'HOLD' and position_ratio > 0:
                trade_result = self.executor.execute_trade(
                    signal=signal, price=current_price, portfolio=self.portfolio, position_ratio=position_ratio
                )
                if trade_result:
                    self.portfolio.update(trade_result)
                    result.update({
                        'portfolio_value': self.portfolio.get_total_value(current_price),
                        'trade_executed': True,
                        'trade_data': trade_result
                    })
                    
                    # Record trade with performance tracker
                    if self.enable_performance_tracking and self.performance_tracker:
                        self._record_trade_performance(trade_result, current_price, data_point)
            
            # 4. Update performance snapshot periodically
            if self.enable_performance_tracking and self.performance_tracker:
                self._update_performance_snapshot(current_price, data_point)
            
            return result
            
        except KeyError as e:
            logger.error(f"Error processing data point: Missing key {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing data point: {str(e)}")
            return None
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        return self.portfolio.get_state()
    
    def reset(self):
        """Reset the trading engine state."""
        self.portfolio = PortfolioManager(self.config)
        self.signal_generator = TradingSignalGenerator(self.config)
        self.executor = TradeExecutor(self.config)
        
        # Reset performance tracking
        if self.enable_performance_tracking and self.performance_tracker:
            # Clear trade and performance history for fresh start
            self.performance_tracker.trade_history.clear()
            self.performance_tracker.performance_history.clear()
            logger.info("Performance tracking reset with trading engine")
    
    def _record_trade_performance(self, trade_result: Dict[str, Any], current_price: float, data_point: Dict[str, Any]):
        """Record trade data with performance tracker."""
        try:
            # Calculate trade return percentage
            if 'pnl' in trade_result and trade_result['pnl'] is not None:
                position_value = abs(trade_result.get('position', 0.0) * current_price)
                return_pct = (trade_result['pnl'] / position_value * 100) if position_value > 0 else 0.0
            else:
                return_pct = 0.0
            
            # Create trade record for performance tracker
            trade_data = {
                'return_pct': return_pct,
                'confidence': data_point.get('confidence', 0.0),
                'uncertainty_pct': data_point.get('interval_width_pct', 0.0),
                'position_size': abs(trade_result.get('position', 0.0)),
                'success': trade_result.get('pnl', 0.0) > 0
            }
            
            # Record with performance tracker
            self.performance_tracker.record_trade(trade_data)
            
            # Store in database if available
            if self.db_manager:
                from data.storage import TradeData
                # Ensure entry_price is valid (> 0)
                entry_price = trade_result.get('entry_price', current_price)
                if entry_price <= 0:
                    entry_price = current_price
                
                # Ensure position_size is valid
                position_size = abs(trade_result.get('position', 0.0))
                if position_size <= 0:
                    position_size = 0.001  # Minimum position size
                
                trade_record = TradeData(
                    entry_timestamp=int(data_point.get('timestamp', datetime.now()).timestamp()) 
                        if isinstance(data_point.get('timestamp'), datetime) 
                        else data_point.get('timestamp', int(datetime.now().timestamp())),
                    entry_price=entry_price,
                    position_size=position_size,
                    position_type='long' if trade_result.get('position', 0.0) > 0 else 'short',
                    confidence_score=data_point.get('confidence', 0.0),
                    exit_timestamp=int(datetime.now().timestamp()) if trade_result.get('is_closed', False) else None,
                    exit_price=current_price if trade_result.get('is_closed', False) else None,
                    status='closed' if trade_result.get('is_closed', False) else 'open',
                    pnl=trade_result.get('pnl')
                )
                
                self.db_manager.store_trade(trade_record)
            
        except Exception as e:
            logger.error(f"Error recording trade performance: {str(e)}")
    
    def _update_performance_snapshot(self, current_price: float, data_point: Dict[str, Any]):
        """Update performance snapshot periodically."""
        try:
            # Update performance snapshot every 10 data points to avoid excessive database writes
            if not hasattr(self, '_snapshot_counter'):
                self._snapshot_counter = 0
            
            self._snapshot_counter += 1
            
            if self._snapshot_counter >= 10:  # Update every 10 data points
                portfolio_value = self.portfolio.get_total_value(current_price)
                self.performance_tracker.update_performance_snapshot(
                    portfolio_value=portfolio_value,
                    initial_capital=self.config.initial_capital
                )
                
                # Store performance snapshot in database
                if self.db_manager and self.performance_tracker.performance_history:
                    latest_metrics = self.performance_tracker.performance_history[-1]
                    
                    from data.storage import PerformanceSnapshot
                    snapshot = PerformanceSnapshot(
                        timestamp=int(latest_metrics.timestamp.timestamp()),
                        total_return_pct=latest_metrics.total_return_pct,
                        win_rate_pct=latest_metrics.win_rate_pct,
                        avg_confidence=latest_metrics.avg_confidence,
                        avg_uncertainty_pct=latest_metrics.avg_uncertainty_pct,
                        trades_per_day=latest_metrics.trades_per_day,
                        max_drawdown_pct=latest_metrics.max_drawdown_pct,
                        portfolio_value=portfolio_value,
                        active_positions=1 if self.portfolio.position != 0 else 0,
                        sharpe_ratio=latest_metrics.sharpe_ratio
                    )
                    
                    self.db_manager.store_performance_snapshot(snapshot)
                
                self._snapshot_counter = 0  # Reset counter
                
        except Exception as e:
            logger.error(f"Error updating performance snapshot: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.enable_performance_tracking or not self.performance_tracker:
            return {"error": "Performance tracking not enabled"}
        
        try:
            return self.performance_tracker.get_performance_summary()
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {"error": str(e)}
    
    def get_parameter_adjustment_suggestions(self) -> List[ParameterAdjustment]:
        """Get intelligent parameter adjustment suggestions from adaptive manager."""
        if not self.enable_adaptive_parameters or not self.adaptive_manager:
            return []
        
        if not self.enable_performance_tracking or not self.performance_tracker:
            logger.warning("Performance tracking required for parameter adjustments")
            return []
        
        try:
            # Get current performance metrics
            performance_metrics = self.get_performance_summary()
            
            # Get current configuration as dict
            current_params = {
                'confidence_threshold': getattr(self.config, 'confidence_threshold', 0.45),
                'min_price_change_pct': getattr(self.config, 'min_price_change_pct', 1.2),
                'max_interval_width_pct': getattr(self.config, 'max_interval_width_pct', 20.0),
                'max_position_size': getattr(self.config, 'max_position_size', 0.10),
                'max_drawdown': getattr(self.config, 'max_drawdown', 0.15)
            }
            
            # Get suggestions from adaptive manager
            suggestions = self.adaptive_manager.suggest_adjustments(
                performance_metrics.get('current_performance', {}),
                current_params
            )
            
            logger.info(f"Generated {len(suggestions)} adaptive parameter suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting adaptive parameter suggestions: {str(e)}")
            return []
    
    def apply_parameter_adjustments(self, adjustments: List[ParameterAdjustment]) -> bool:
        """Apply adaptive parameter adjustments to trading configuration."""
        if not adjustments:
            return False
            
        if not self.enable_adaptive_parameters or not self.adaptive_manager:
            logger.warning("Adaptive parameters not enabled")
            return False
            
        try:
            applied_count = 0
            
            for adjustment in adjustments:
                if hasattr(self.config, adjustment.parameter_name):
                    # Validate the adjustment
                    validation = self.adaptive_manager.validate_parameters({
                        adjustment.parameter_name: adjustment.new_value
                    })
                    
                    if not validation['valid']:
                        logger.warning(f"Parameter adjustment validation failed: {validation['warnings']}")
                        # Use corrected value if available
                        corrected_value = validation['corrections'].get(adjustment.parameter_name)
                        if corrected_value is not None:
                            adjustment.new_value = corrected_value
                            logger.info(f"Using corrected value: {corrected_value}")
                        else:
                            continue
                    
                    # Update the configuration
                    setattr(self.config, adjustment.parameter_name, adjustment.new_value)
                    
                    # Apply through adaptive manager to record history
                    self.adaptive_manager.apply_adjustment(adjustment)
                    
                    # Store adjustment record in database
                    if self.db_manager:
                        from ..data.storage import ParameterAdjustment as DBParameterAdjustment
                        adjustment_record = DBParameterAdjustment(
                            timestamp=int(adjustment.timestamp.timestamp()),
                            parameter_name=adjustment.parameter_name,
                            old_value=float(adjustment.old_value),
                            new_value=float(adjustment.new_value),
                            adjustment_reason=adjustment.reason.value,
                            trigger_metric=f"confidence_{adjustment.confidence_score:.2f}",
                            trigger_value=adjustment.confidence_score,
                            performance_before=str(self.get_performance_summary())
                        )
                        
                        self.db_manager.store_parameter_adjustment(adjustment_record)
                    
                    applied_count += 1
                    logger.info(f"Applied adaptive parameter adjustment: {adjustment.parameter_name} "
                               f"{adjustment.old_value:.3f} → {adjustment.new_value:.3f} "
                               f"(reason: {adjustment.reason.value}, confidence: {adjustment.confidence_score:.2f})")
                else:
                    logger.warning(f"Parameter {adjustment.parameter_name} not found in configuration")
            
            # Update signal generator and executor with new config
            self.signal_generator = TradingSignalGenerator(self.config)
            self.executor = TradeExecutor(self.config)
            
            logger.info(f"Successfully applied {applied_count} adaptive parameter adjustments")
            return applied_count > 0
            
        except Exception as e:
            logger.error(f"Error applying parameter adjustments: {str(e)}")
            return False
    
    def check_and_apply_adaptive_parameters(self, force_check: bool = False) -> Dict[str, Any]:
        """
        Check if adaptive parameter adjustments are needed and apply them.
        
        Args:
            force_check: Force parameter check regardless of timing
            
        Returns:
            Dictionary with adjustment results
        """
        if not self.enable_adaptive_parameters or not self.adaptive_manager:
            return {"status": "adaptive_parameters_disabled"}
        
        # Check timing - only run every 6 hours unless forced
        time_since_last = datetime.now() - self.last_adjustment_check
        if not force_check and time_since_last.total_seconds() < 6 * 3600:  # 6 hours
            return {"status": "too_soon", "next_check_in_hours": 6 - time_since_last.total_seconds() / 3600}
        
        try:
            # Get suggestions
            suggestions = self.get_parameter_adjustment_suggestions()
            
            if not suggestions:
                self.last_adjustment_check = datetime.now()
                return {"status": "no_adjustments_needed", "suggestions_count": 0}
            
            # Filter high-confidence suggestions for auto-application
            auto_apply_threshold = 0.8
            high_confidence_suggestions = [
                s for s in suggestions 
                if s.confidence_score >= auto_apply_threshold
            ]
            
            results = {
                "status": "adjustments_available",
                "total_suggestions": len(suggestions),
                "high_confidence_suggestions": len(high_confidence_suggestions),
                "suggestions": [
                    {
                        "parameter": s.parameter_name,
                        "current_value": s.old_value,
                        "suggested_value": s.new_value,
                        "reason": s.reason.value,
                        "confidence": s.confidence_score,
                        "expected_impact": s.expected_impact
                    }
                    for s in suggestions
                ],
                "applied_adjustments": []
            }
            
            # Auto-apply high confidence adjustments
            if high_confidence_suggestions:
                logger.info(f"Auto-applying {len(high_confidence_suggestions)} high-confidence parameter adjustments")
                success = self.apply_parameter_adjustments(high_confidence_suggestions)
                
                if success:
                    results["applied_adjustments"] = [
                        {
                            "parameter": s.parameter_name,
                            "old_value": s.old_value,
                            "new_value": s.new_value,
                            "reason": s.reason.value
                        }
                        for s in high_confidence_suggestions
                    ]
                    results["status"] = "adjustments_applied"
            
            self.last_adjustment_check = datetime.now()
            return results
            
        except Exception as e:
            logger.error(f"Error in adaptive parameter check: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_adaptive_parameter_status(self) -> Dict[str, Any]:
        """Get current adaptive parameter system status."""
        if not self.enable_adaptive_parameters or not self.adaptive_manager:
            return {"enabled": False}
        
        try:
            current_params = {
                'confidence_threshold': getattr(self.config, 'confidence_threshold', 0.45),
                'min_price_change_pct': getattr(self.config, 'min_price_change_pct', 1.2),
                'max_interval_width_pct': getattr(self.config, 'max_interval_width_pct', 20.0),
                'max_position_size': getattr(self.config, 'max_position_size', 0.10),
                'max_drawdown': getattr(self.config, 'max_drawdown', 0.15)
            }
            
            bounds = self.adaptive_manager.get_current_bounds()
            recent_adjustments = self.adaptive_manager.get_adjustment_history(days=7)
            
            # Check if parameters are at optimal values
            optimal_params = self.adaptive_manager.reset_to_optimal()
            parameter_drift = {}
            for param, current_val in current_params.items():
                optimal_val = optimal_params.get(param, current_val)
                drift_pct = abs(current_val - optimal_val) / optimal_val * 100
                parameter_drift[param] = {
                    "current": current_val,
                    "optimal": optimal_val,
                    "drift_pct": drift_pct
                }
            
            return {
                "enabled": True,
                "current_parameters": current_params,
                "parameter_bounds": bounds,
                "parameter_drift": parameter_drift,
                "recent_adjustments": recent_adjustments,
                "last_check": self.last_adjustment_check.isoformat(),
                "pending_suggestions": len(self.pending_adjustments)
            }
            
        except Exception as e:
            logger.error(f"Error getting adaptive parameter status: {e}")
            return {"enabled": True, "error": str(e)}
    
    def reset_parameters_to_optimal(self) -> bool:
        """Reset all parameters to optimal values from backtesting."""
        if not self.enable_adaptive_parameters or not self.adaptive_manager:
            return False
        
        try:
            optimal_params = self.adaptive_manager.reset_to_optimal()
            
            # Create adjustment records for reset
            adjustments = []
            for param_name, new_value in optimal_params.items():
                if hasattr(self.config, param_name):
                    old_value = getattr(self.config, param_name)
                    
                    adjustment = ParameterAdjustment(
                        timestamp=datetime.now(),
                        parameter_name=param_name,
                        old_value=old_value,
                        new_value=new_value,
                        reason=AdjustmentReason.MANUAL_OVERRIDE,
                        confidence_score=1.0,
                        expected_impact="Reset to optimal values from backtesting"
                    )
                    adjustments.append(adjustment)
            
            # Apply the adjustments
            success = self.apply_parameter_adjustments(adjustments)
            
            if success:
                logger.info("Successfully reset all parameters to optimal values")
                
            return success
            
        except Exception as e:
            logger.error(f"Error resetting parameters to optimal: {e}")
            return False 