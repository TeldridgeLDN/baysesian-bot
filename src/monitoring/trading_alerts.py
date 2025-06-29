"""
Trading-specific alert integration for the Bayesian Crypto Trading Bot.

This module integrates the general alerts system with trading-specific monitoring,
including performance tracker integration, parameter drift detection, and
trading anomaly detection.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from .alerts import (
    PerformanceAlertManager, Alert, AlertRule, AlertSeverity, AlertCategory,
    log_alert_handler, database_alert_handler
)
from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class TradingAlertManager:
    """
    Trading-specific alert manager that integrates with the performance tracker
    and provides specialized trading performance monitoring.
    """
    
    def __init__(self, performance_tracker: PerformanceTracker, 
                 config: Optional[Dict[str, Any]] = None):
        """Initialize trading alert manager."""
        self.performance_tracker = performance_tracker
        self.config = config or {}
        
        # Initialize base alert manager
        self.alert_manager = PerformanceAlertManager(config)
        
        # Add default handlers
        self.alert_manager.add_alert_handler(log_alert_handler)
        if self.config.get('enable_database_alerts', True):
            self.alert_manager.add_alert_handler(
                lambda alert: database_alert_handler(alert, self.config.get('alert_db_path', 'data/trading_alerts.db'))
            )
        
        # Trading-specific state
        self.last_performance_check = datetime.now()
        self.parameter_baseline: Dict[str, float] = {}
        self.performance_trends: Dict[str, List[float]] = {}
        
        # Add trading-specific alert rules
        self._add_trading_alert_rules()
        
        logger.info("TradingAlertManager initialized")
    
    def _add_trading_alert_rules(self):
        """Add trading-specific alert rules."""
        trading_rules = [
            # Advanced Trading Performance Rules
            AlertRule(
                id="consecutive_losses",
                name="Consecutive Losses",
                category=AlertCategory.TRADING_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric_name="consecutive_losses",
                condition="greater_than",
                threshold=5.0,
                cooldown_minutes=120,
                description="More than 5 consecutive losing trades"
            ),
            AlertRule(
                id="rapid_drawdown",
                name="Rapid Drawdown",
                category=AlertCategory.RISK_MANAGEMENT,
                severity=AlertSeverity.ERROR,
                metric_name="drawdown_velocity",
                condition="greater_than",
                threshold=5.0,  # 5% drawdown in short period
                cooldown_minutes=30,
                description="Rapid drawdown detected"
            ),
            AlertRule(
                id="model_confidence_drop",
                name="Model Confidence Drop",
                category=AlertCategory.MODEL_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric_name="confidence_trend",
                condition="less_than",
                threshold=-0.1,  # 10% drop in confidence
                cooldown_minutes=60,
                description="Significant drop in model confidence"
            ),
            AlertRule(
                id="volatility_spike",
                name="Market Volatility Spike",
                category=AlertCategory.MARKET_ANOMALY,
                severity=AlertSeverity.INFO,
                metric_name="market_volatility",
                condition="greater_than",
                threshold=0.05,  # 5% volatility spike
                cooldown_minutes=15,
                description="Unusual market volatility detected"
            ),
            AlertRule(
                id="data_staleness",
                name="Stale Data",
                category=AlertCategory.DATA_QUALITY,
                severity=AlertSeverity.WARNING,
                metric_name="data_age_minutes",
                condition="greater_than",
                threshold=30.0,  # Data older than 30 minutes
                cooldown_minutes=60,
                description="Market data is stale"
            ),
            AlertRule(
                id="parameter_optimization_needed",
                name="Parameter Optimization Needed",
                category=AlertCategory.PARAMETER_DRIFT,
                severity=AlertSeverity.INFO,
                metric_name="optimization_score",
                condition="less_than",
                threshold=0.7,  # Optimization score below 70%
                cooldown_minutes=360,
                description="Parameters may need optimization"
            )
        ]
        
        for rule in trading_rules:
            self.alert_manager.add_alert_rule(rule)
        
        logger.info(f"Added {len(trading_rules)} trading-specific alert rules")
    
    def check_trading_performance(self, current_price: float, 
                                trade_result: Optional[Dict[str, Any]] = None) -> List[Alert]:
        """
        Check trading performance and generate alerts.
        
        Args:
            current_price: Current market price
            trade_result: Recent trade result (if any)
            
        Returns:
            List of triggered alerts
        """
        try:
            # Get performance summary from tracker
            performance_summary = self.performance_tracker.get_performance_summary()
            
            # Calculate additional trading metrics
            trading_metrics = self._calculate_trading_metrics(
                performance_summary, current_price, trade_result
            )
            
            # Combine with base metrics
            all_metrics = {**performance_summary, **trading_metrics}
            
            # Check metrics against alert rules
            alerts = self.alert_manager.check_metrics(all_metrics)
            
            # Update performance trends
            self._update_performance_trends(all_metrics)
            
            self.last_performance_check = datetime.now()
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking trading performance: {e}")
            return []
    
    def _calculate_trading_metrics(self, performance_summary: Dict[str, Any], 
                                 current_price: float, 
                                 trade_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate additional trading-specific metrics."""
        metrics = {}
        
        try:
            # Consecutive losses
            if hasattr(self.performance_tracker, 'trade_history'):
                recent_trades = self.performance_tracker.trade_history[-10:]  # Last 10 trades
                consecutive_losses = 0
                for trade in reversed(recent_trades):
                    if trade.get('success', False):
                        break
                    consecutive_losses += 1
                metrics['consecutive_losses'] = consecutive_losses
            
            # Drawdown velocity (rate of drawdown change)
            if 'max_drawdown_pct' in performance_summary:
                current_drawdown = performance_summary['max_drawdown_pct']
                if hasattr(self, '_last_drawdown'):
                    drawdown_change = current_drawdown - self._last_drawdown
                    time_delta_hours = (datetime.now() - self.last_performance_check).total_seconds() / 3600
                    metrics['drawdown_velocity'] = drawdown_change / max(time_delta_hours, 0.1)
                self._last_drawdown = current_drawdown
            
            # Confidence trend
            if 'avg_confidence' in performance_summary:
                current_confidence = performance_summary['avg_confidence']
                if 'avg_confidence' in self.performance_trends:
                    recent_confidences = self.performance_trends['avg_confidence'][-5:]
                    if len(recent_confidences) >= 2:
                        trend = (recent_confidences[-1] - recent_confidences[0]) / len(recent_confidences)
                        metrics['confidence_trend'] = trend
            
            # Market volatility (simplified calculation)
            if hasattr(self, '_price_history'):
                self._price_history.append(current_price)
                if len(self._price_history) > 20:
                    self._price_history = self._price_history[-20:]
                
                if len(self._price_history) > 5:
                    returns = np.diff(self._price_history) / self._price_history[:-1]
                    volatility = np.std(returns)
                    metrics['market_volatility'] = volatility
            else:
                self._price_history = [current_price]
            
            # Data staleness
            metrics['data_age_minutes'] = (datetime.now() - self.last_performance_check).total_seconds() / 60
            
            # Optimization score (based on parameter drift from optimal)
            optimization_score = self._calculate_optimization_score()
            metrics['optimization_score'] = optimization_score
            
            # Add system metrics
            import psutil
            process = psutil.Process()
            metrics.update({
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_usage_pct': process.cpu_percent(),
                'thread_count': process.num_threads()
            })
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
        
        return metrics
    
    def _calculate_optimization_score(self) -> float:
        """Calculate how close current parameters are to optimal values."""
        try:
            # This would integrate with the adaptive parameter manager
            # For now, return a default score
            base_score = 0.8
            
            # Adjust based on recent performance
            if hasattr(self.performance_tracker, 'performance_history'):
                recent_performance = self.performance_tracker.performance_history[-5:]
                if recent_performance:
                    avg_win_rate = np.mean([p.win_rate_pct for p in recent_performance])
                    if avg_win_rate > 50:
                        base_score += 0.1
                    elif avg_win_rate < 40:
                        base_score -= 0.2
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.5
    
    def _update_performance_trends(self, metrics: Dict[str, Any]):
        """Update performance trend tracking."""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric_name not in self.performance_trends:
                    self.performance_trends[metric_name] = []
                
                self.performance_trends[metric_name].append(value)
                
                # Keep only recent values (last 20 measurements)
                if len(self.performance_trends[metric_name]) > 20:
                    self.performance_trends[metric_name] = self.performance_trends[metric_name][-20:]
    
    def check_parameter_drift(self, current_parameters: Dict[str, float], 
                            optimal_parameters: Dict[str, float]) -> List[Alert]:
        """
        Check for parameter drift from optimal values.
        
        Args:
            current_parameters: Current trading parameters
            optimal_parameters: Optimal parameter values
            
        Returns:
            List of drift-related alerts
        """
        try:
            drift_metrics = {}
            total_drift = 0.0
            param_count = 0
            
            for param_name, current_value in current_parameters.items():
                if param_name in optimal_parameters:
                    optimal_value = optimal_parameters[param_name]
                    if optimal_value != 0:
                        drift_pct = abs(current_value - optimal_value) / optimal_value * 100
                        drift_metrics[f'{param_name}_drift_pct'] = drift_pct
                        total_drift += drift_pct
                        param_count += 1
            
            if param_count > 0:
                avg_drift_pct = total_drift / param_count
                drift_metrics['parameter_drift_pct'] = avg_drift_pct
            
            return self.alert_manager.check_metrics(drift_metrics)
            
        except Exception as e:
            logger.error(f"Error checking parameter drift: {e}")
            return []
    
    def check_market_anomalies(self, price_data: List[float], 
                             volume_data: List[float]) -> List[Alert]:
        """
        Check for market anomalies that might affect trading.
        
        Args:
            price_data: Recent price data
            volume_data: Recent volume data
            
        Returns:
            List of market anomaly alerts
        """
        try:
            anomaly_metrics = {}
            
            if len(price_data) > 5:
                # Price volatility analysis
                returns = np.diff(price_data) / price_data[:-1]
                volatility = np.std(returns)
                avg_volatility = np.mean(np.abs(returns))
                
                anomaly_metrics.update({
                    'price_volatility': volatility,
                    'avg_price_movement': avg_volatility,
                    'max_price_change': np.max(np.abs(returns)) * 100
                })
            
            if len(volume_data) > 5:
                # Volume analysis
                avg_volume = np.mean(volume_data)
                volume_spike = np.max(volume_data) / avg_volume if avg_volume > 0 else 0
                
                anomaly_metrics.update({
                    'volume_spike_ratio': volume_spike,
                    'volume_variance': np.var(volume_data)
                })
            
            return self.alert_manager.check_metrics(anomaly_metrics)
            
        except Exception as e:
            logger.error(f"Error checking market anomalies: {e}")
            return []
    
    def get_alert_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive alert dashboard data."""
        try:
            base_summary = self.alert_manager.get_alert_summary()
            
            # Add trading-specific information
            trading_dashboard = {
                **base_summary,
                'trading_specific': {
                    'last_performance_check': self.last_performance_check.isoformat(),
                    'performance_trends_tracked': len(self.performance_trends),
                    'consecutive_monitoring_hours': (datetime.now() - self.last_performance_check).total_seconds() / 3600,
                },
                'recent_trading_alerts': self._get_recent_trading_alerts(),
                'parameter_drift_status': self._get_parameter_drift_status(),
                'market_conditions': self._get_market_conditions_summary()
            }
            
            return trading_dashboard
            
        except Exception as e:
            logger.error(f"Error generating alert dashboard: {e}")
            return {'error': str(e)}
    
    def _get_recent_trading_alerts(self) -> Dict[str, Any]:
        """Get summary of recent trading-specific alerts."""
        recent_alerts = self.alert_manager.get_alert_history(hours=24)
        
        trading_alerts = [
            alert for alert in recent_alerts 
            if alert.category in [
                AlertCategory.TRADING_PERFORMANCE,
                AlertCategory.RISK_MANAGEMENT,
                AlertCategory.MODEL_PERFORMANCE
            ]
        ]
        
        return {
            'total_trading_alerts_24h': len(trading_alerts),
            'by_severity': {
                severity.value: len([a for a in trading_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            'most_recent': trading_alerts[0].title if trading_alerts else None
        }
    
    def _get_parameter_drift_status(self) -> Dict[str, Any]:
        """Get current parameter drift status."""
        # This would integrate with adaptive parameter manager
        return {
            'last_drift_check': self.last_performance_check.isoformat(),
            'parameters_monitored': len(self.parameter_baseline),
            'drift_alerts_24h': len([
                alert for alert in self.alert_manager.get_alert_history(hours=24)
                if alert.category == AlertCategory.PARAMETER_DRIFT
            ])
        }
    
    def _get_market_conditions_summary(self) -> Dict[str, Any]:
        """Get current market conditions summary."""
        return {
            'volatility_level': 'normal',  # Would be calculated from recent data
            'trend_direction': 'neutral',  # Would be calculated from price trends
            'data_quality': 'good',        # Would be based on data freshness
            'anomalies_detected_24h': len([
                alert for alert in self.alert_manager.get_alert_history(hours=24)
                if alert.category == AlertCategory.MARKET_ANOMALY
            ])
        }
    
    def start_monitoring(self, check_interval: int = 60):
        """Start continuous trading performance monitoring."""
        self.alert_manager.start_monitoring(check_interval)
        logger.info(f"Started trading alert monitoring (interval: {check_interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.alert_manager.stop_monitoring()
        logger.info("Stopped trading alert monitoring")
    
    def add_custom_handler(self, handler):
        """Add a custom alert handler."""
        self.alert_manager.add_alert_handler(handler)
    
    def export_trading_alerts(self) -> str:
        """Export trading alerts with additional trading context."""
        base_export = self.alert_manager.export_alerts()
        
        # Add trading-specific context
        import json
        export_data = json.loads(base_export)
        export_data['trading_context'] = {
            'performance_trends': {k: v[-5:] for k, v in self.performance_trends.items()},
            'parameter_baseline': self.parameter_baseline,
            'dashboard_summary': self.get_alert_dashboard()
        }
        
        return json.dumps(export_data, indent=2, default=str)