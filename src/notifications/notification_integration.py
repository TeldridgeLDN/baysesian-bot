"""
Integration module for connecting Telegram notifications with 
trading alerts and parameter adjustment systems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

from notifications.telegram_alerts import TelegramNotificationManager, TelegramAlertHandler, NotificationLevel
from monitoring.alerts import PerformanceAlertManager
from monitoring.trading_alerts import TradingAlertManager
from trading.adaptive_parameters import AdaptiveParameterManager

logger = logging.getLogger(__name__)

class NotificationIntegrationManager:
    """
    Manages integration between trading systems and Telegram notifications.
    
    Features:
    - Automatic alert forwarding to Telegram
    - Parameter adjustment announcements
    - Daily summary generation and delivery
    - System health monitoring notifications
    - Performance milestone alerts
    """
    
    def __init__(self, 
                 telegram_config: Dict[str, Any],
                 alert_manager: PerformanceAlertManager,
                 trading_alert_manager: TradingAlertManager,
                 adaptive_manager: Optional[AdaptiveParameterManager] = None):
        """Initialize notification integration."""
        
        # Initialize Telegram manager
        self.telegram_manager = TelegramNotificationManager(
            bot_token=telegram_config.get('bot_token', ''),
            chat_id=telegram_config.get('chat_id', ''),
            notification_level=NotificationLevel(telegram_config.get('notification_level', 'normal')),
            config=telegram_config
        )
        
        # Store manager references
        self.alert_manager = alert_manager
        self.trading_alert_manager = trading_alert_manager
        self.adaptive_manager = adaptive_manager
        
        # Integration state
        self.is_running = False
        self.last_daily_summary = None
        self.parameter_adjustment_history = []
        
        # Configuration
        self.config = telegram_config
        self.daily_summary_hour = self.config.get('daily_summary_hour', 18)  # 6 PM
        self.system_status_interval_hours = self.config.get('system_status_interval_hours', 6)
        
        # Setup alert handler integration
        self._setup_alert_integration()
        
        logger.info("NotificationIntegrationManager initialized")
    
    def _setup_alert_integration(self):
        """Setup automatic alert forwarding to Telegram."""
        telegram_handler = TelegramAlertHandler(self.telegram_manager)
        self.alert_manager.add_alert_handler(telegram_handler)
        logger.info("Telegram alert handler added to alert manager")
    
    async def start_integration(self):
        """Start the notification integration service."""
        if self.is_running:
            logger.warning("Notification integration already running")
            return
        
        self.is_running = True
        
        # Test Telegram connection
        connection_success = await self.telegram_manager.test_connection()
        if not connection_success:
            logger.error("Failed to connect to Telegram - notifications may not work")
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Notification integration started")
    
    def stop_integration(self):
        """Stop the notification integration service."""
        self.is_running = False
        logger.info("Notification integration stopped")
    
    def _start_background_tasks(self):
        """Start background tasks for scheduled notifications."""
        def background_loop():
            """Background task runner."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                while self.is_running:
                    # Check for daily summary
                    loop.run_until_complete(self._check_daily_summary())
                    
                    # Check for system status updates
                    loop.run_until_complete(self._check_system_status())
                    
                    # Sleep for a minute before next check
                    if self.is_running:
                        loop.run_until_complete(asyncio.sleep(60))
                        
            except Exception as e:
                logger.error(f"Error in background notification task: {e}")
            finally:
                loop.close()
        
        # Start background thread
        background_thread = threading.Thread(target=background_loop, daemon=True)
        background_thread.start()
        logger.info("Background notification tasks started")
    
    async def _check_daily_summary(self):
        """Check if daily summary should be sent."""
        now = datetime.now()
        
        # Check if it's time for daily summary
        if (now.hour == self.daily_summary_hour and 
            (self.last_daily_summary is None or 
             self.last_daily_summary.date() < now.date())):
            
            await self.send_daily_summary()
            self.last_daily_summary = now
    
    async def _check_system_status(self):
        """Check if system status should be sent."""
        # Only send system status in verbose mode
        if self.telegram_manager.notification_level != NotificationLevel.VERBOSE:
            return
        
        # Send system status every few hours
        now = datetime.now()
        if hasattr(self, '_last_system_status'):
            hours_since_last = (now - self._last_system_status).total_seconds() / 3600
            if hours_since_last < self.system_status_interval_hours:
                return
        
        await self.send_system_status()
        self._last_system_status = now
    
    async def notify_parameter_adjustment(self, 
                                        adjustment_type: str,
                                        old_parameters: Dict[str, float],
                                        new_parameters: Dict[str, float],
                                        performance_context: Dict[str, Any],
                                        reasoning: str = ""):
        """Send parameter adjustment notification."""
        try:
            # Store adjustment in history
            adjustment_record = {
                'timestamp': datetime.now(),
                'type': adjustment_type,
                'old_parameters': old_parameters.copy(),
                'new_parameters': new_parameters.copy(),
                'performance_context': performance_context.copy(),
                'reasoning': reasoning
            }
            self.parameter_adjustment_history.append(adjustment_record)
            
            # Keep only recent history
            if len(self.parameter_adjustment_history) > 50:
                self.parameter_adjustment_history = self.parameter_adjustment_history[-50:]
            
            # Send notification
            success = await self.telegram_manager.send_parameter_adjustment_notification(
                adjustment_type=adjustment_type,
                old_parameters=old_parameters,
                new_parameters=new_parameters,
                performance_context=performance_context,
                reasoning=reasoning
            )
            
            if success:
                logger.info(f"Sent parameter adjustment notification: {adjustment_type}")
            else:
                logger.warning(f"Failed to send parameter adjustment notification: {adjustment_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending parameter adjustment notification: {e}")
            return False
    
    async def notify_trade_execution(self,
                                   action: str,
                                   symbol: str,
                                   amount: float,
                                   price: float,
                                   confidence: float,
                                   prediction_data: Dict[str, Any]):
        """Send trade execution notification."""
        try:
            success = await self.telegram_manager.send_trade_notification(
                action=action,
                symbol=symbol,
                amount=amount,
                price=price,
                confidence=confidence,
                prediction_data=prediction_data
            )
            
            if success:
                logger.info(f"Sent trade notification: {action} {symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {e}")
            return False
    
    async def send_daily_summary(self):
        """Generate and send daily performance summary."""
        try:
            # Get performance data from trading alert manager
            dashboard = self.trading_alert_manager.get_alert_dashboard()
            
            # Get recent performance metrics
            performance_summary = self.trading_alert_manager.performance_tracker.get_performance_summary()
            
            # Calculate daily metrics
            summary_data = {
                'total_return_pct': performance_summary.get('total_return_pct', 0),
                'win_rate': performance_summary.get('win_rate_pct', 0),
                'total_trades': performance_summary.get('total_trades', 0),
                'best_trade_pct': performance_summary.get('best_return_pct', 0),
                'avg_confidence': performance_summary.get('avg_confidence', 0),
                'avg_uncertainty_pct': performance_summary.get('avg_uncertainty_pct', 0),
                'confidence_threshold': 0.4,  # Would get from current parameters
                'position_size': 0.1  # Would get from current parameters
            }
            
            # Add recent parameter adjustments
            recent_adjustments = [
                adj for adj in self.parameter_adjustment_history
                if (datetime.now() - adj['timestamp']).days == 0
            ]
            
            if recent_adjustments:
                summary_data['parameter_adjustments_today'] = len(recent_adjustments)
            
            success = await self.telegram_manager.send_daily_summary(summary_data)
            
            if success:
                logger.info("Sent daily summary notification")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False
    
    async def send_system_status(self):
        """Send system health status notification."""
        try:
            # Get system metrics from alert manager
            alert_summary = self.alert_manager.get_alert_summary()
            system_status = alert_summary.get('system_status', {})
            
            # Get trading status
            dashboard = self.trading_alert_manager.get_alert_dashboard()
            
            status_data = {
                'status': 'healthy' if alert_summary['active_alerts_count'] == 0 else 'warning',
                'memory_usage_mb': system_status.get('memory_usage_mb', 0),
                'cpu_usage_pct': system_status.get('cpu_usage_pct', 0),
                'uptime_hours': system_status.get('uptime_hours', 0),
                'trading_mode': 'paper',  # Would get from configuration
                'active_positions': 0,    # Would get from trading engine
                'active_alerts': alert_summary['active_alerts_count']
            }
            
            success = await self.telegram_manager.send_system_status(status_data)
            
            if success:
                logger.info("Sent system status notification")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending system status: {e}")
            return False
    
    async def send_performance_milestone(self, milestone_type: str, data: Dict[str, Any]):
        """Send performance milestone notification."""
        try:
            milestone_messages = {
                'profit_target': "🎯 *Profit Target Reached!*\nDaily return target achieved: {return_pct:+.2%}",
                'win_streak': "🔥 *Win Streak!*\n{streak} consecutive winning trades",
                'new_high': "📈 *New Performance High!*\nPortfolio reached new all-time high",
                'recovery': "🔄 *Recovery Complete!*\nDrawdown recovered to positive territory"
            }
            
            template = milestone_messages.get(milestone_type, "🎉 *Performance Milestone*\n{description}")
            message = template.format(**data)
            
            # Send as system status type message
            success = await self.telegram_manager.send_system_status({
                'status': 'milestone',
                'milestone_type': milestone_type,
                'milestone_data': data,
                'timestamp': datetime.now().isoformat(),
                **data
            })
            
            if success:
                logger.info(f"Sent performance milestone notification: {milestone_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending performance milestone: {e}")
            return False
    
    def update_notification_settings(self, settings: Dict[str, Any]):
        """Update notification settings."""
        try:
            # Update notification level
            if 'notification_level' in settings:
                new_level = NotificationLevel(settings['notification_level'])
                self.telegram_manager.update_notification_level(new_level)
            
            # Update timing settings
            if 'daily_summary_hour' in settings:
                self.daily_summary_hour = settings['daily_summary_hour']
            
            if 'system_status_interval_hours' in settings:
                self.system_status_interval_hours = settings['system_status_interval_hours']
            
            logger.info(f"Updated notification settings: {settings}")
            
        except Exception as e:
            logger.error(f"Error updating notification settings: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and statistics."""
        telegram_status = self.telegram_manager.get_status()
        
        return {
            'integration_running': self.is_running,
            'telegram_status': telegram_status,
            'last_daily_summary': self.last_daily_summary.isoformat() if self.last_daily_summary else None,
            'parameter_adjustments_count': len(self.parameter_adjustment_history),
            'daily_summary_hour': self.daily_summary_hour,
            'system_status_interval_hours': self.system_status_interval_hours
        }
    
    async def test_all_notifications(self):
        """Send test notifications for all message types."""
        test_results = {}
        
        try:
            # Test connection
            test_results['connection'] = await self.telegram_manager.test_connection()
            
            # Test parameter adjustment
            test_results['parameter_adjustment'] = await self.notify_parameter_adjustment(
                adjustment_type="Test Adjustment",
                old_parameters={'confidence_threshold': 0.40, 'position_size': 0.10},
                new_parameters={'confidence_threshold': 0.45, 'position_size': 0.12},
                performance_context={'win_rate_pct': 45, 'max_drawdown_pct': 8, 'avg_confidence': 0.42},
                reasoning="This is a test parameter adjustment notification."
            )
            
            # Test trade notification
            test_results['trade_notification'] = await self.notify_trade_execution(
                action="BUY",
                symbol="BTC/USD",
                amount=1000.0,
                price=50000.0,
                confidence=0.75,
                prediction_data={
                    'target_price': 52000.0,
                    'stop_loss': 48000.0,
                    'direction': 1,
                    'uncertainty_pct': 15.0
                }
            )
            
            # Test daily summary
            test_results['daily_summary'] = await self.send_daily_summary()
            
            # Test system status
            test_results['system_status'] = await self.send_system_status()
            
            logger.info(f"Test notifications completed: {test_results}")
            return test_results
            
        except Exception as e:
            logger.error(f"Error testing notifications: {e}")
            return {'error': str(e)}