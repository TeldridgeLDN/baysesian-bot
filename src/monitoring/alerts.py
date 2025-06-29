"""
Real-time Performance Alerts System for Bayesian Crypto Trading Bot.

This module implements a comprehensive alerting system that monitors:
- Trading performance metrics in real-time
- System health and resource usage
- Parameter drift and adaptation events
- Critical trading failures and anomalies
- Market condition changes
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
import time
import psutil
import sqlite3

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertCategory(Enum):
    """Alert categories for better organization."""
    TRADING_PERFORMANCE = "trading_performance"
    SYSTEM_HEALTH = "system_health"
    PARAMETER_DRIFT = "parameter_drift"
    MARKET_ANOMALY = "market_anomaly"
    DATA_QUALITY = "data_quality"
    MODEL_PERFORMANCE = "model_performance"
    RISK_MANAGEMENT = "risk_management"

@dataclass
class Alert:
    """Represents a performance alert."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    additional_data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class AlertRule:
    """Defines an alert rule with conditions and thresholds."""
    id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals'
    threshold: float
    enabled: bool = True
    cooldown_minutes: int = 30  # Minimum time between same alerts
    description: str = ""

class PerformanceAlertManager:
    """
    Manages real-time performance alerts for the trading system.
    
    Features:
    - Real-time monitoring of trading metrics
    - Configurable alert rules and thresholds
    - Alert aggregation and deduplication
    - Multi-channel alert delivery (logs, database, external systems)
    - Alert acknowledgment and resolution tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance alert manager."""
        self.config = config or {}
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_queue: queue.Queue = queue.Queue()
        
        # Performance metrics cache
        self.latest_metrics: Dict[str, Any] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
        logger.info("PerformanceAlertManager initialized")
    
    def _initialize_default_rules(self):
        """Initialize default alert rules based on backtesting insights."""
        default_rules = [
            # Trading Performance Alerts
            AlertRule(
                id="low_win_rate",
                name="Low Win Rate",
                category=AlertCategory.TRADING_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric_name="win_rate_pct",
                condition="less_than",
                threshold=40.0,
                cooldown_minutes=60,
                description="Win rate dropped below 40% threshold"
            ),
            AlertRule(
                id="high_drawdown",
                name="High Drawdown",
                category=AlertCategory.RISK_MANAGEMENT,
                severity=AlertSeverity.ERROR,
                metric_name="max_drawdown_pct",
                condition="greater_than",
                threshold=15.0,
                cooldown_minutes=30,
                description="Maximum drawdown exceeded 15% safety limit"
            ),
            AlertRule(
                id="low_confidence",
                name="Low Average Confidence",
                category=AlertCategory.MODEL_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric_name="avg_confidence",
                condition="less_than",
                threshold=0.30,
                cooldown_minutes=120,
                description="Average model confidence below 30%"
            ),
            AlertRule(
                id="high_uncertainty",
                name="High Uncertainty",
                category=AlertCategory.MODEL_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric_name="avg_uncertainty_pct",
                condition="greater_than",
                threshold=25.0,
                cooldown_minutes=90,
                description="Average uncertainty above 25%"
            ),
            AlertRule(
                id="trading_frequency_low",
                name="Low Trading Frequency",
                category=AlertCategory.TRADING_PERFORMANCE,
                severity=AlertSeverity.INFO,
                metric_name="trades_per_day",
                condition="less_than",
                threshold=0.5,
                cooldown_minutes=240,
                description="Trading frequency below 0.5 trades per day"
            ),
            AlertRule(
                id="trading_frequency_high",
                name="High Trading Frequency",
                category=AlertCategory.RISK_MANAGEMENT,
                severity=AlertSeverity.WARNING,
                metric_name="trades_per_day",
                condition="greater_than",
                threshold=10.0,
                cooldown_minutes=60,
                description="Trading frequency above 10 trades per day (possible overtrading)"
            ),
            AlertRule(
                id="poor_sharpe_ratio",
                name="Poor Sharpe Ratio",
                category=AlertCategory.TRADING_PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric_name="sharpe_ratio",
                condition="less_than",
                threshold=0.5,
                cooldown_minutes=180,
                description="Sharpe ratio below 0.5 (poor risk-adjusted returns)"
            ),
            
            # System Health Alerts
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                category=AlertCategory.SYSTEM_HEALTH,
                severity=AlertSeverity.WARNING,
                metric_name="memory_usage_mb",
                condition="greater_than",
                threshold=2048,
                cooldown_minutes=15,
                description="Memory usage above 2GB"
            ),
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                category=AlertCategory.SYSTEM_HEALTH,
                severity=AlertSeverity.WARNING,
                metric_name="cpu_usage_pct",
                condition="greater_than",
                threshold=80.0,
                cooldown_minutes=10,
                description="CPU usage above 80%"
            ),
            
            # Parameter Drift Alerts
            AlertRule(
                id="parameter_drift",
                name="Parameter Drift",
                category=AlertCategory.PARAMETER_DRIFT,
                severity=AlertSeverity.INFO,
                metric_name="parameter_drift_pct",
                condition="greater_than",
                threshold=20.0,
                cooldown_minutes=360,
                description="Parameters drifted >20% from optimal values"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
        
        logger.info(f"Initialized {len(default_rules)} default alert rules")
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a custom alert rule."""
        try:
            self.alert_rules[rule.id] = rule
            logger.info(f"Added alert rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding alert rule: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing alert rule: {e}")
            return False
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a handler function for alerts."""
        self.alert_handlers.append(handler)
        handler_name = getattr(handler, '__name__', str(type(handler).__name__))
        logger.info(f"Added alert handler: {handler_name}")
    
    def check_metrics(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        Check metrics against alert rules and generate alerts.
        
        Args:
            metrics: Dictionary of current performance metrics
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        current_time = datetime.now()
        
        # Update metrics cache
        self.latest_metrics = metrics
        self.metrics_history.append({
            'timestamp': current_time,
            'metrics': metrics.copy()
        })
        
        # Keep only recent history (last 100 measurements)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        # Check each alert rule
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule_id, current_time):
                continue
            
            # Get metric value
            metric_value = metrics.get(rule.metric_name)
            if metric_value is None:
                continue
            
            # Evaluate condition
            if self._evaluate_condition(metric_value, rule.condition, rule.threshold):
                alert = self._create_alert(rule, metric_value, current_time)
                triggered_alerts.append(alert)
                
                # Set cooldown
                self.alert_cooldowns[rule_id] = current_time
                
                # Store alert
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                
                # Trigger handlers
                self._trigger_alert_handlers(alert)
        
        return triggered_alerts
    
    def _is_in_cooldown(self, rule_id: str, current_time: datetime) -> bool:
        """Check if alert rule is in cooldown period."""
        if rule_id not in self.alert_cooldowns:
            return False
        
        last_alert_time = self.alert_cooldowns[rule_id]
        rule = self.alert_rules[rule_id]
        cooldown_duration = timedelta(minutes=rule.cooldown_minutes)
        
        return current_time - last_alert_time < cooldown_duration
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001
        elif condition == "not_equals":
            return abs(value - threshold) >= 0.001
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    def _create_alert(self, rule: AlertRule, current_value: float, timestamp: datetime) -> Alert:
        """Create an alert from a rule and current value."""
        alert_id = f"{rule.id}_{int(timestamp.timestamp())}"
        
        # Create descriptive message
        direction = "above" if rule.condition == "greater_than" else "below"
        message = f"{rule.description}. Current value: {current_value:.2f}, Threshold: {rule.threshold:.2f}"
        
        # Add context from recent metrics
        additional_data = {
            'rule_id': rule.id,
            'evaluation_time': timestamp.isoformat(),
            'recent_metrics': self.metrics_history[-5:] if self.metrics_history else []
        }
        
        return Alert(
            id=alert_id,
            timestamp=timestamp,
            severity=rule.severity,
            category=rule.category,
            title=rule.name,
            message=message,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold,
            additional_data=additional_data
        )
    
    def _trigger_alert_handlers(self, alert: Alert):
        """Trigger all registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__name__}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                self.active_alerts[alert_id].additional_data['acknowledged_by'] = acknowledged_by
                self.active_alerts[alert_id].additional_data['acknowledged_at'] = datetime.now().isoformat()
                
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_timestamp = datetime.now()
                alert.additional_data['resolved_by'] = resolved_by
                
                # Move to history and remove from active
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None, 
                         category: Optional[AlertCategory] = None) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts
    
    def get_alert_history(self, hours: int = 24, 
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = [a for a in self.alert_history if a.timestamp > cutoff_time]
        
        if severity:
            history = [a for a in history if a.severity == severity]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        return history
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary."""
        current_time = datetime.now()
        last_24h = current_time - timedelta(hours=24)
        
        # Count alerts by severity in last 24h
        recent_alerts = [a for a in self.alert_history if a.timestamp > last_24h]
        severity_counts = {}
        category_counts = {}
        
        for alert in recent_alerts:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            category_counts[alert.category.value] = category_counts.get(alert.category.value, 0) + 1
        
        # Most frequent alerts
        metric_counts = {}
        for alert in recent_alerts:
            metric_counts[alert.metric_name] = metric_counts.get(alert.metric_name, 0) + 1
        
        top_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'timestamp': current_time.isoformat(),
            'active_alerts_count': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
            'last_24h_summary': {
                'total_alerts': len(recent_alerts),
                'by_severity': severity_counts,
                'by_category': category_counts,
                'top_triggered_metrics': top_metrics
            },
            'system_status': self._get_system_status(),
            'alert_handlers_count': len(self.alert_handlers)
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status for health monitoring."""
        try:
            process = psutil.Process()
            
            return {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_usage_pct': process.cpu_percent(),
                'uptime_hours': (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds() / 3600,
                'thread_count': process.num_threads(),
                'monitoring_active': self.is_monitoring
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def start_monitoring(self, check_interval: int = 60):
        """Start continuous monitoring with specified check interval."""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        
        def monitoring_loop():
            logger.info(f"Started performance monitoring (interval: {check_interval}s)")
            
            while self.is_monitoring:
                try:
                    # Get system metrics
                    system_metrics = self._get_system_status()
                    
                    # Check system metrics against rules
                    self.check_metrics(system_metrics)
                    
                    # Process any queued external metrics
                    self._process_queued_metrics()
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)  # Brief pause on error
            
            logger.info("Performance monitoring stopped")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            self.monitoring_thread = None
        
        logger.info("Performance monitoring stopped")
    
    def queue_metrics_check(self, metrics: Dict[str, Any]):
        """Queue metrics for checking (thread-safe)."""
        self.alert_queue.put(metrics)
    
    def _process_queued_metrics(self):
        """Process all queued metrics."""
        while not self.alert_queue.empty():
            try:
                metrics = self.alert_queue.get_nowait()
                self.check_metrics(metrics)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing queued metrics: {e}")
    
    def export_alerts(self, format: str = "json") -> str:
        """Export alerts in specified format."""
        try:
            alert_data = {
                'export_timestamp': datetime.now().isoformat(),
                'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
                'alert_history': [asdict(alert) for alert in self.alert_history],
                'alert_rules': [asdict(rule) for rule in self.alert_rules.values()],
                'summary': self.get_alert_summary()
            }
            
            if format.lower() == "json":
                return json.dumps(alert_data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting alerts: {e}")
            return json.dumps({'error': str(e)})

# Built-in alert handlers
def log_alert_handler(alert: Alert):
    """Default alert handler that logs alerts."""
    log_level = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.ERROR: logging.ERROR,
        AlertSeverity.CRITICAL: logging.CRITICAL
    }.get(alert.severity, logging.INFO)
    
    logger.log(log_level, f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")

def database_alert_handler(alert: Alert, db_path: str = "data/alerts.db"):
    """Alert handler that stores alerts in SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                severity TEXT,
                category TEXT,
                title TEXT,
                message TEXT,
                metric_name TEXT,
                current_value REAL,
                threshold_value REAL,
                additional_data TEXT,
                acknowledged INTEGER,
                resolved INTEGER
            )
        ''')
        
        # Insert alert
        cursor.execute('''
            INSERT OR REPLACE INTO alerts 
            (id, timestamp, severity, category, title, message, metric_name, 
             current_value, threshold_value, additional_data, acknowledged, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.id,
            alert.timestamp.isoformat(),
            alert.severity.value,
            alert.category.value,
            alert.title,
            alert.message,
            alert.metric_name,
            alert.current_value,
            alert.threshold_value,
            json.dumps(alert.additional_data),
            1 if alert.acknowledged else 0,
            1 if alert.resolved else 0
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error storing alert in database: {e}")