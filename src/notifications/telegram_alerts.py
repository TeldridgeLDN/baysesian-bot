"""
Telegram notification system for Bayesian Crypto Trading Bot.
Sends real-time alerts and parameter adjustment notifications via Telegram.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import os

# Import telegram bot library
try:
    from telegram import Bot, ParseMode
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    Bot = None
    TelegramError = Exception
    TELEGRAM_AVAILABLE = False
    logging.warning("Telegram library not available. Install with: pip install python-telegram-bot")

from monitoring.alerts import Alert, AlertSeverity, AlertCategory
from monitoring.trading_alerts import TradingAlertManager

logger = logging.getLogger(__name__)

class NotificationLevel(Enum):
    """Notification levels for filtering messages."""
    MINIMAL = "minimal"      # Only critical alerts
    NORMAL = "normal"        # Important trading events
    VERBOSE = "verbose"      # All alerts and detailed info

class MessageType(Enum):
    """Types of Telegram messages."""
    ALERT = "alert"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    TRADE_NOTIFICATION = "trade_notification"
    DAILY_SUMMARY = "daily_summary"
    SYSTEM_STATUS = "system_status"

@dataclass
class TelegramMessage:
    """Represents a Telegram message to be sent."""
    chat_id: str
    message_type: MessageType
    title: str
    content: str
    priority: AlertSeverity
    timestamp: datetime
    parse_mode: str = "Markdown"
    additional_data: Dict[str, Any] = None

class TelegramNotificationManager:
    """
    Manages Telegram notifications for the trading bot.
    
    Features:
    - Real-time alert notifications
    - Parameter adjustment announcements
    - Trade notifications with context
    - Daily performance summaries
    - System health status updates
    - Configurable notification levels
    """
    
    def __init__(self, bot_token: str, chat_id: str, 
                 notification_level: NotificationLevel = NotificationLevel.NORMAL,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize Telegram notification manager."""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.notification_level = notification_level
        self.config = config or {}
        
        # Initialize Telegram bot
        if TELEGRAM_AVAILABLE and bot_token:
            self.bot = Bot(token=bot_token)
            self.enabled = True
        else:
            self.bot = None
            self.enabled = False
            logger.warning("Telegram notifications disabled - missing token or library")
        
        # Message queue and rate limiting
        self.message_queue: List[TelegramMessage] = []
        self.last_sent_time = datetime.now()
        self.rate_limit_seconds = self.config.get('rate_limit_seconds', 5)
        self.max_message_length = self.config.get('max_message_length', 4000)
        
        # Notification filters
        self.notification_filters = self._initialize_filters()
        
        # Message templates
        self.message_templates = self._initialize_templates()
        
        logger.info(f"TelegramNotificationManager initialized (enabled: {self.enabled})")
    
    def _initialize_filters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize notification filters based on level."""
        filters = {
            NotificationLevel.MINIMAL.value: {
                'alert_severities': [AlertSeverity.CRITICAL],
                'alert_categories': [AlertCategory.RISK_MANAGEMENT, AlertCategory.SYSTEM_HEALTH],
                'include_parameter_adjustments': False,
                'include_trade_notifications': False,
                'include_daily_summaries': False
            },
            NotificationLevel.NORMAL.value: {
                'alert_severities': [AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL],
                'alert_categories': list(AlertCategory),
                'include_parameter_adjustments': True,
                'include_trade_notifications': True,
                'include_daily_summaries': True
            },
            NotificationLevel.VERBOSE.value: {
                'alert_severities': list(AlertSeverity),
                'alert_categories': list(AlertCategory),
                'include_parameter_adjustments': True,
                'include_trade_notifications': True,
                'include_daily_summaries': True,
                'include_system_status': True
            }
        }
        
        return filters.get(self.notification_level.value, filters[NotificationLevel.NORMAL.value])
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize message templates."""
        return {
            'alert': """🚨 *{severity} Alert*
📊 *{title}*

{message}

⏰ Time: {timestamp}
📈 Metric: {metric_name}
🎯 Current: {current_value}
🔴 Threshold: {threshold_value}

{additional_info}""",
            
            'parameter_adjustment': """⚙️ *Parameter Adjustment*
🎛️ *{adjustment_type}*

📊 *Changes Made:*
{parameter_changes}

📈 *Performance Context:*
• Win Rate: {win_rate}%
• Drawdown: {drawdown}%
• Confidence: {confidence}%

⏰ Applied: {timestamp}

{reasoning}""",
            
            'trade_notification': """💰 *Trade {action}*
🪙 *{symbol}*

💵 Amount: ${amount:.2f}
📊 Confidence: {confidence:.1%}
🎯 Target: {target_price:.2f}
🛡️ Stop Loss: {stop_loss:.2f}

📈 *Model Prediction:*
• Direction: {direction}
• Uncertainty: ±{uncertainty:.1%}

⏰ {timestamp}""",
            
            'daily_summary': """📊 *Daily Trading Summary*
📅 {date}

💰 *Performance:*
• Total Return: {total_return:+.2%}
• Win Rate: {win_rate:.1%}
• Trades: {total_trades}
• Best Trade: {best_trade:+.2%}

📈 *Model Stats:*
• Avg Confidence: {avg_confidence:.1%}
• Avg Uncertainty: {avg_uncertainty:.1%}

⚙️ *Current Parameters:*
• Confidence Threshold: {conf_threshold:.1%}
• Position Size: {position_size:.1%}

{additional_insights}""",
            
            'system_status': """🖥️ *System Status*
🟢 Status: {status}

💾 *Resources:*
• Memory: {memory_mb:.0f}MB
• CPU: {cpu_pct:.1f}%
• Uptime: {uptime_hours:.1f}h

📊 *Trading:*
• Mode: {trading_mode}
• Active Positions: {active_positions}
• Alerts: {active_alerts}

⏰ {timestamp}"""
        }
    
    async def send_alert_notification(self, alert: Alert) -> bool:
        """Send alert notification via Telegram."""
        if not self._should_send_alert(alert):
            logger.debug(f"Alert filtered out: {alert.title} ({alert.severity.value})")
            return False
        
        try:
            # Format alert message
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️", 
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            additional_info = ""
            if alert.additional_data:
                if 'rule_id' in alert.additional_data:
                    additional_info += f"🔍 Rule: {alert.additional_data['rule_id']}\n"
                if 'evaluation_time' in alert.additional_data:
                    additional_info += f"⏱️ Evaluated: {alert.additional_data['evaluation_time']}\n"
            
            message_content = self.message_templates['alert'].format(
                severity=f"{severity_emoji.get(alert.severity, '⚠️')} {alert.severity.value.upper()}",
                title=alert.title,
                message=alert.message,
                timestamp=alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                metric_name=alert.metric_name,
                current_value=f"{alert.current_value:.2f}",
                threshold_value=f"{alert.threshold_value:.2f}",
                additional_info=additional_info.strip()
            )
            
            telegram_message = TelegramMessage(
                chat_id=self.chat_id,
                message_type=MessageType.ALERT,
                title=f"Alert: {alert.title}",
                content=message_content,
                priority=alert.severity,
                timestamp=datetime.now(),
                additional_data=asdict(alert)
            )
            
            return await self._send_message(telegram_message)
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
            return False
    
    async def send_parameter_adjustment_notification(self, 
                                                   adjustment_type: str,
                                                   old_parameters: Dict[str, float],
                                                   new_parameters: Dict[str, float],
                                                   performance_context: Dict[str, Any],
                                                   reasoning: str = "") -> bool:
        """Send parameter adjustment notification."""
        if not self.notification_filters.get('include_parameter_adjustments', True):
            return False
        
        try:
            # Format parameter changes
            changes = []
            for param, new_value in new_parameters.items():
                old_value = old_parameters.get(param, 0)
                change_pct = ((new_value - old_value) / old_value * 100) if old_value != 0 else 0
                direction = "↗️" if change_pct > 0 else "↘️" if change_pct < 0 else "➡️"
                changes.append(f"• {param}: {old_value:.3f} → {new_value:.3f} {direction} ({change_pct:+.1f}%)")
            
            message_content = self.message_templates['parameter_adjustment'].format(
                adjustment_type=adjustment_type,
                parameter_changes="\n".join(changes),
                win_rate=performance_context.get('win_rate_pct', 0),
                drawdown=performance_context.get('max_drawdown_pct', 0),
                confidence=performance_context.get('avg_confidence', 0) * 100,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                reasoning=reasoning or "Automated adjustment based on performance metrics."
            )
            
            telegram_message = TelegramMessage(
                chat_id=self.chat_id,
                message_type=MessageType.PARAMETER_ADJUSTMENT,
                title="Parameter Adjustment",
                content=message_content,
                priority=AlertSeverity.INFO,
                timestamp=datetime.now(),
                additional_data={
                    'old_parameters': old_parameters,
                    'new_parameters': new_parameters,
                    'performance_context': performance_context
                }
            )
            
            return await self._send_message(telegram_message)
            
        except Exception as e:
            logger.error(f"Error sending parameter adjustment notification: {e}")
            return False
    
    async def send_trade_notification(self, 
                                    action: str,
                                    symbol: str,
                                    amount: float,
                                    price: float,
                                    confidence: float,
                                    prediction_data: Dict[str, Any]) -> bool:
        """Send trade notification."""
        if not self.notification_filters.get('include_trade_notifications', True):
            return False
        
        try:
            target_price = prediction_data.get('target_price', price)
            stop_loss = prediction_data.get('stop_loss', price * 0.95)
            direction = "🟢 LONG" if prediction_data.get('direction', 1) > 0 else "🔴 SHORT"
            uncertainty = prediction_data.get('uncertainty_pct', 0) / 100
            
            message_content = self.message_templates['trade_notification'].format(
                action=action.upper(),
                symbol=symbol,
                amount=amount,
                confidence=confidence,
                target_price=target_price,
                stop_loss=stop_loss,
                direction=direction,
                uncertainty=uncertainty,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            telegram_message = TelegramMessage(
                chat_id=self.chat_id,
                message_type=MessageType.TRADE_NOTIFICATION,
                title=f"Trade {action}: {symbol}",
                content=message_content,
                priority=AlertSeverity.INFO,
                timestamp=datetime.now(),
                additional_data=prediction_data
            )
            
            return await self._send_message(telegram_message)
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {e}")
            return False
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily performance summary."""
        if not self.notification_filters.get('include_daily_summaries', True):
            return False
        
        try:
            # Generate insights based on performance
            insights = []
            win_rate = summary_data.get('win_rate', 0)
            total_return = summary_data.get('total_return_pct', 0)
            
            if win_rate > 60:
                insights.append("🎯 Excellent win rate!")
            elif win_rate < 40:
                insights.append("⚠️ Low win rate - consider parameter adjustment")
            
            if total_return > 2:
                insights.append("📈 Strong daily performance")
            elif total_return < -1:
                insights.append("📉 Negative day - monitor closely")
            
            additional_insights = "\n".join(insights) if insights else "📊 Standard trading day"
            
            message_content = self.message_templates['daily_summary'].format(
                date=datetime.now().strftime('%Y-%m-%d'),
                total_return=summary_data.get('total_return_pct', 0) / 100,
                win_rate=summary_data.get('win_rate', 0),
                total_trades=summary_data.get('total_trades', 0),
                best_trade=summary_data.get('best_trade_pct', 0) / 100,
                avg_confidence=summary_data.get('avg_confidence', 0) * 100,
                avg_uncertainty=summary_data.get('avg_uncertainty_pct', 0),
                conf_threshold=summary_data.get('confidence_threshold', 0.4) * 100,
                position_size=summary_data.get('position_size', 0.1) * 100,
                additional_insights=additional_insights
            )
            
            telegram_message = TelegramMessage(
                chat_id=self.chat_id,
                message_type=MessageType.DAILY_SUMMARY,
                title="Daily Summary",
                content=message_content,
                priority=AlertSeverity.INFO,
                timestamp=datetime.now(),
                additional_data=summary_data
            )
            
            return await self._send_message(telegram_message)
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False
    
    async def send_system_status(self, status_data: Dict[str, Any]) -> bool:
        """Send system status update."""
        if not self.notification_filters.get('include_system_status', False):
            return False
        
        try:
            status_emoji = "🟢" if status_data.get('status') == 'healthy' else "🟡"
            
            message_content = self.message_templates['system_status'].format(
                status=f"{status_emoji} {status_data.get('status', 'unknown').title()}",
                memory_mb=status_data.get('memory_usage_mb', 0),
                cpu_pct=status_data.get('cpu_usage_pct', 0),
                uptime_hours=status_data.get('uptime_hours', 0),
                trading_mode=status_data.get('trading_mode', 'paper'),
                active_positions=status_data.get('active_positions', 0),
                active_alerts=status_data.get('active_alerts', 0),
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            telegram_message = TelegramMessage(
                chat_id=self.chat_id,
                message_type=MessageType.SYSTEM_STATUS,
                title="System Status",
                content=message_content,
                priority=AlertSeverity.INFO,
                timestamp=datetime.now(),
                additional_data=status_data
            )
            
            return await self._send_message(telegram_message)
            
        except Exception as e:
            logger.error(f"Error sending system status: {e}")
            return False
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Determine if alert should be sent based on filters."""
        # Check severity filter
        if alert.severity not in self.notification_filters.get('alert_severities', []):
            return False
        
        # Check category filter
        if alert.category not in self.notification_filters.get('alert_categories', []):
            return False
        
        return True
    
    async def _send_message(self, message: TelegramMessage) -> bool:
        """Send message via Telegram with rate limiting."""
        if not self.enabled:
            logger.debug(f"Telegram disabled - would send: {message.title}")
            return True
        
        try:
            # Rate limiting
            time_since_last = (datetime.now() - self.last_sent_time).total_seconds()
            if time_since_last < self.rate_limit_seconds:
                await asyncio.sleep(self.rate_limit_seconds - time_since_last)
            
            # Truncate message if too long
            content = message.content
            if len(content) > self.max_message_length:
                content = content[:self.max_message_length-50] + "\n\n... (truncated)"
            
            # Send message
            await self.bot.send_message(
                chat_id=message.chat_id,
                text=content,
                parse_mode=message.parse_mode
            )
            
            self.last_sent_time = datetime.now()
            logger.info(f"Sent Telegram message: {message.title}")
            return True
            
        except TelegramError as e:
            logger.error(f"Telegram error sending message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        if not self.enabled:
            logger.warning("Telegram not enabled - cannot test connection")
            return False
        
        try:
            bot_info = await self.bot.get_me()
            logger.info(f"Telegram bot connected: {bot_info.first_name} (@{bot_info.username})")
            
            # Send test message
            test_message = f"🤖 *Bot Connected*\nTest message from Bayesian Crypto Trading Bot\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=test_message,
                parse_mode="Markdown"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
    
    def update_notification_level(self, level: NotificationLevel):
        """Update notification level and filters."""
        self.notification_level = level
        self.notification_filters = self._initialize_filters()
        logger.info(f"Updated notification level to: {level.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get notification manager status."""
        return {
            'enabled': self.enabled,
            'notification_level': self.notification_level.value,
            'chat_id': self.chat_id,
            'rate_limit_seconds': self.rate_limit_seconds,
            'last_sent': self.last_sent_time.isoformat() if self.last_sent_time else None,
            'filters': self.notification_filters
        }

class TelegramAlertHandler:
    """Alert handler that sends alerts via Telegram."""
    
    def __init__(self, telegram_manager: TelegramNotificationManager):
        self.telegram_manager = telegram_manager
    
    def __call__(self, alert: Alert) -> None:
        """Handle alert by sending Telegram notification."""
        try:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            success = loop.run_until_complete(
                self.telegram_manager.send_alert_notification(alert)
            )
            
            if not success:
                logger.warning(f"Failed to send Telegram alert: {alert.title}")
                
        except Exception as e:
            logger.error(f"Error in Telegram alert handler: {e}")
        finally:
            loop.close()