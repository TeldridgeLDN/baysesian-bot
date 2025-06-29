"""
Performance monitoring and adaptive parameter adjustment based on backtesting insights.
Tracks trading performance and suggests parameter adjustments when performance degrades.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring trading system."""
    timestamp: datetime
    total_return_pct: float
    win_rate_pct: float
    avg_confidence: float
    avg_uncertainty_pct: float
    trades_per_day: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float] = None
    
class PerformanceTracker:
    """
    Monitors trading performance and suggests parameter adjustments.
    Based on backtesting insights that showed optimal parameters vary by market conditions.
    """
    
    def __init__(self, monitoring_config):
        self.config = monitoring_config
        self.performance_history = deque(maxlen=100)  # Keep last 100 performance snapshots
        self.trade_history = deque(maxlen=1000)  # Keep last 1000 trades
        self.alert_history = deque(maxlen=50)  # Keep last 50 alerts
        
        # Optimal parameter ranges from backtesting
        self.optimal_ranges = {
            'confidence_threshold': (0.40, 0.50),  # Sweet spot for crypto
            'min_price_change_pct': (1.0, 1.5),   # Aggressive but not over-trading
            'max_interval_width_pct': (18.0, 22.0), # Crypto-appropriate uncertainty
            'position_size': (0.10, 0.12),        # Conservative but profitable
        }
        
        # Performance thresholds that trigger parameter adjustment
        self.adjustment_triggers = {
            'consecutive_losses': 5,      # 5 losses in a row
            'low_win_rate_days': 3,      # Win rate below 40% for 3 days
            'high_drawdown': 0.10,       # 10% drawdown
            'low_trading_frequency': 0.2  # Less than 0.2 trades per day for 5 days
        }
        
    def record_trade(self, trade_data: Dict):
        """Record a completed trade for performance tracking."""
        trade_record = {
            'timestamp': datetime.now(),
            'return_pct': trade_data.get('return_pct', 0.0),
            'confidence': trade_data.get('confidence', 0.0),
            'uncertainty_pct': trade_data.get('uncertainty_pct', 0.0),
            'position_size': trade_data.get('position_size', 0.0),
            'success': trade_data.get('return_pct', 0.0) > 0
        }
        
        self.trade_history.append(trade_record)
        logger.info(f"Recorded trade: {trade_record['return_pct']:+.2f}% return, "
                   f"{trade_record['confidence']:.3f} confidence")
        
    def update_performance_snapshot(self, portfolio_value: float, initial_capital: float):
        """Update current performance metrics."""
        if len(self.trade_history) < 5:  # Need minimum trades for meaningful metrics
            return
            
        recent_trades = list(self.trade_history)[-20:]  # Last 20 trades
        
        # Calculate metrics
        total_return_pct = ((portfolio_value - initial_capital) / initial_capital) * 100
        win_rate_pct = len([t for t in recent_trades if t['success']]) / len(recent_trades) * 100
        avg_confidence = np.mean([t['confidence'] for t in recent_trades])
        avg_uncertainty_pct = np.mean([t['uncertainty_pct'] for t in recent_trades])
        
        # Calculate trades per day (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_week_trades = [t for t in recent_trades if t['timestamp'] > week_ago]
        trades_per_day = len(recent_week_trades) / 7.0
        
        # Calculate max drawdown
        returns = [t['return_pct'] for t in recent_trades]
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown_pct = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Calculate Sharpe ratio (if we have enough data)
        sharpe_ratio = None
        if len(returns) >= 10:
            returns_array = np.array(returns)
            if np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(365)  # Annualized
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            total_return_pct=total_return_pct,
            win_rate_pct=win_rate_pct,
            avg_confidence=avg_confidence,
            avg_uncertainty_pct=avg_uncertainty_pct,
            trades_per_day=trades_per_day,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio
        )
        
        self.performance_history.append(metrics)
        self._check_performance_alerts(metrics)
        
        logger.info(f"Performance update: {total_return_pct:+.2f}% return, "
                   f"{win_rate_pct:.1f}% win rate, {trades_per_day:.1f} trades/day")
        
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check if performance metrics trigger alerts."""
        alerts = []
        thresholds = self.config.alert_thresholds
        
        # Check each threshold
        if metrics.win_rate_pct < thresholds.get('min_win_rate_pct', 40.0):
            alerts.append(f"Low win rate: {metrics.win_rate_pct:.1f}% < {thresholds['min_win_rate_pct']:.1f}%")
            
        if metrics.max_drawdown_pct > thresholds.get('max_drawdown_pct', 15.0):
            alerts.append(f"High drawdown: {metrics.max_drawdown_pct:.1f}% > {thresholds['max_drawdown_pct']:.1f}%")
            
        if metrics.avg_confidence < thresholds.get('min_confidence_avg', 0.30):
            alerts.append(f"Low confidence: {metrics.avg_confidence:.3f} < {thresholds['min_confidence_avg']:.3f}")
            
        if metrics.avg_uncertainty_pct > thresholds.get('max_uncertainty_avg', 25.0):
            alerts.append(f"High uncertainty: {metrics.avg_uncertainty_pct:.1f}% > {thresholds['max_uncertainty_avg']:.1f}%")
            
        if metrics.trades_per_day < thresholds.get('min_trades_per_day', 0.5):
            alerts.append(f"Low trading frequency: {metrics.trades_per_day:.2f} < {thresholds['min_trades_per_day']:.2f}")
            
        if metrics.trades_per_day > thresholds.get('max_trades_per_day', 10.0):
            alerts.append(f"Overtrading: {metrics.trades_per_day:.2f} > {thresholds['max_trades_per_day']:.2f}")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Performance Alert: {alert}")
            self.alert_history.append({
                'timestamp': datetime.now(),
                'alert': alert,
                'metrics': metrics
            })
            
    def should_adjust_parameters(self) -> bool:
        """Determine if parameters should be adjusted based on performance."""
        if len(self.performance_history) < 5:
            return False
            
        recent_performance = list(self.performance_history)[-5:]
        recent_trades = list(self.trade_history)[-10:]
        
        # Check consecutive losses
        if len(recent_trades) >= self.adjustment_triggers['consecutive_losses']:
            recent_results = [t['success'] for t in recent_trades[-self.adjustment_triggers['consecutive_losses']:]]
            if not any(recent_results):  # All False
                logger.warning("Consecutive losses detected - parameter adjustment recommended")
                return True
        
        # Check sustained low win rate
        low_win_rate_count = sum(1 for p in recent_performance 
                                if p.win_rate_pct < self.config.alert_thresholds.get('min_win_rate_pct', 40.0))
        if low_win_rate_count >= self.adjustment_triggers['low_win_rate_days']:
            logger.warning("Sustained low win rate - parameter adjustment recommended")
            return True
            
        # Check high drawdown
        latest_metrics = recent_performance[-1]
        if latest_metrics.max_drawdown_pct > self.adjustment_triggers['high_drawdown'] * 100:
            logger.warning("High drawdown detected - parameter adjustment recommended")
            return True
            
        # Check low trading frequency
        low_frequency_count = sum(1 for p in recent_performance 
                                 if p.trades_per_day < self.adjustment_triggers['low_trading_frequency'])
        if low_frequency_count >= 5:  # 5 days of low activity
            logger.warning("Low trading frequency - parameter adjustment recommended")
            return True
            
        return False
        
    def get_parameter_adjustment_suggestions(self) -> Dict[str, float]:
        """
        Suggest parameter adjustments based on current performance.
        Uses insights from backtesting about what parameters work in different conditions.
        """
        if not self.should_adjust_parameters():
            return {}
            
        suggestions = {}
        latest_metrics = self.performance_history[-1] if self.performance_history else None
        
        if latest_metrics is None:
            return suggestions
            
        # Analyze current issues and suggest fixes
        
        # Low win rate - reduce confidence threshold to get more trades
        if latest_metrics.win_rate_pct < 45.0:
            current_confidence = 0.50  # Would need current config
            new_confidence = max(0.35, current_confidence - 0.05)
            suggestions['confidence_threshold'] = new_confidence
            logger.info(f"Suggesting lower confidence threshold: {new_confidence:.3f}")
            
        # High uncertainty - increase uncertainty tolerance
        if latest_metrics.avg_uncertainty_pct > 20.0:
            suggestions['max_interval_width_pct'] = min(25.0, latest_metrics.avg_uncertainty_pct + 2.0)
            logger.info(f"Suggesting higher uncertainty tolerance: {suggestions['max_interval_width_pct']:.1f}%")
            
        # Low trading frequency - more aggressive parameters
        if latest_metrics.trades_per_day < 0.5:
            suggestions['min_price_change_pct'] = 0.8  # More aggressive
            suggestions['confidence_threshold'] = 0.35  # Lower bar
            logger.info("Suggesting more aggressive parameters for low trading frequency")
            
        # High drawdown - more conservative parameters
        if latest_metrics.max_drawdown_pct > 12.0:
            suggestions['position_size'] = 0.08  # Smaller positions
            suggestions['confidence_threshold'] = 0.55  # Higher confidence required
            logger.info("Suggesting more conservative parameters for high drawdown")
            
        return suggestions
        
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {"status": "No performance data available"}
            
        latest = self.performance_history[-1]
        recent_alerts = len([a for a in self.alert_history 
                           if a['timestamp'] > datetime.now() - timedelta(days=1)])
        
        summary = {
            "current_performance": {
                "total_return_pct": latest.total_return_pct,
                "win_rate_pct": latest.win_rate_pct,
                "avg_confidence": latest.avg_confidence,
                "trades_per_day": latest.trades_per_day,
                "max_drawdown_pct": latest.max_drawdown_pct,
                "sharpe_ratio": latest.sharpe_ratio
            },
            "alerts_24h": recent_alerts,
            "total_trades": len(self.trade_history),
            "adjustment_needed": self.should_adjust_parameters(),
            "suggested_adjustments": self.get_parameter_adjustment_suggestions()
        }
        
        return summary