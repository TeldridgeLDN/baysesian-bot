"""
Notification system for Telegram bot.
Handles message formatting, sending, and notification scheduling.
"""

import logging
import asyncio
from datetime import datetime, timedelta
import telegram
from typing import Dict, List, Optional
from enum import Enum
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    HIGH_CONFIDENCE = "high_confidence"
    MODEL_RETRAINED = "model_retrained"
    RISK_ALERT = "risk_alert"
    DAILY_SUMMARY = "daily_summary"
    PERFORMANCE_UPDATE = "performance_update"
    SIGNAL_GENERATED = "signal_generated"

class NotificationManager:
    """Manages bot notifications and message formatting."""
    
    def __init__(self, bot: telegram.Bot, chat_id: str, config: ConfigManager):
        self.bot = bot
        self.chat_id = chat_id
        self.config = config
        self.notification_templates = self._load_templates()
        self.notification_queue = asyncio.Queue()
        self.is_running = False
        
    def _load_templates(self) -> Dict[str, str]:
        """Load notification message templates."""
        return {
            "trade_entry": """
🚀 **Trade Entered**

**Position:** {position_type} BTC
**Entry Price:** ${entry_price:,.2f}
**Size:** {size:.4f} BTC
**Capital:** ${capital:,.2f}
**Confidence:** {confidence:.1f}%
**Uncertainty:** {uncertainty:.1f}%

**Stop Loss:** ${stop_loss:,.2f}
**Take Profit:** ${take_profit:,.2f}

**Reasoning:** {reasoning}

⏰ {timestamp}
            """,
            
            "trade_exit": """
💰 **Trade Closed**

**Position:** {position_type} BTC
**Entry Price:** ${entry_price:,.2f}
**Exit Price:** ${exit_price:,.2f}
**Size:** {size:.4f} BTC

**PnL:** {pnl_symbol} ${pnl:,.2f} ({pnl_pct:+.2f}%)
**Duration:** {duration}
**Exit Reason:** {exit_reason}

**Portfolio PnL:** ${portfolio_pnl:,.2f} ({portfolio_pnl_pct:+.2f}%)

⏰ {timestamp}
            """,
            
            "high_confidence": """
⚡ **High Confidence Signal**

**Direction:** {direction}
**Current Price:** ${current_price:,.2f}
**Prediction:** ${prediction:,.2f}
**Confidence:** {confidence:.1f}%
**Uncertainty:** {uncertainty:.1f}%

**Price Change:** {price_change:+.2f}%
**Signal Strength:** {signal_strength}

**Reasoning:** {reasoning}

⏰ {timestamp}
            """,
            
            "model_retrained": """
🔄 **Model Retrained**

**New Accuracy:** {accuracy:.2f}%
**Previous Accuracy:** {previous_accuracy:.2f}%
**Training Time:** {training_time:.1f} seconds
**Data Points:** {data_points:,}

**Performance Change:** {accuracy_change:+.2f}%

**Next Retraining:** {next_retraining}

⏰ {timestamp}
            """,
            
            "risk_alert": """
⚠️ **Risk Alert**

**Alert Type:** {alert_type}
**Severity:** {severity}

**Current Drawdown:** {current_drawdown:.2f}%
**Max Allowed:** {max_drawdown:.2f}%

**Active Positions:** {active_positions}
**Total Risk:** {total_risk:.2f}%

**Action:** {action}

**Recommendation:** {recommendation}

⏰ {timestamp}
            """,
            
            "daily_summary": """
📊 **Daily Summary**

**Date:** {date}

**Trading Activity:**
• Total Trades: {trades_count}
• Winning Trades: {winning_trades}
• Win Rate: {win_rate:.1f}%

**Performance:**
• Daily PnL: {daily_pnl_symbol} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)
• Total PnL: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)
• Portfolio Value: ${portfolio_value:,.2f}

**Model Performance:**
• Predictions Made: {predictions_count}
• Average Confidence: {avg_confidence:.1f}%
• Average Uncertainty: {avg_uncertainty:.1f}%

**Risk Metrics:**
• Max Drawdown: {max_drawdown:.2f}%
• Volatility: {volatility:.2f}%
• Sharpe Ratio: {sharpe_ratio:.2f}

⏰ {timestamp}
            """,
            
            "performance_update": """
📈 **Performance Update**

**Period:** {period}

**Returns:**
• Period Return: {period_return:+.2f}%
• Annualized Return: {annualized_return:+.2f}%
• Total Return: {total_return:+.2f}%

**Risk Metrics:**
• Sharpe Ratio: {sharpe_ratio:.2f}
• Max Drawdown: {max_drawdown:.2f}%
• Volatility: {volatility:.2f}%
• VaR (95%): {var_95:.2f}%

**Trading Stats:**
• Total Trades: {total_trades}
• Win Rate: {win_rate:.1f}%
• Profit Factor: {profit_factor:.2f}
• Average Win: {avg_win:.2f}%
• Average Loss: {avg_loss:.2f}%

⏰ {timestamp}
            """,
            
            "signal_generated": """
📊 **Signal Generated**

**Signal Type:** {signal_type}
**Confidence:** {confidence:.1f}%
**Uncertainty:** {uncertainty:.1f}%

**Price Analysis:**
• Current Price: ${current_price:,.2f}
• Predicted Price: ${prediction:,.2f}
• Expected Change: {price_change:+.2f}%

**Risk Assessment:**
• Signal Strength: {signal_strength}
• Risk Level: {risk_level}
• Recommended Action: {recommended_action}

**Reasoning:** {reasoning}

⏰ {timestamp}
            """
        }
        
    async def send_notification(self, notification_type: NotificationType, data: Dict):
        """Send formatted notification to user."""
        try:
            template = self.notification_templates.get(notification_type.value)
            if not template:
                logger.error(f"No template found for notification type: {notification_type}")
                return
            
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Format the message
            message = template.format(**data)
            
            # Send the notification
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"Sent {notification_type.value} notification")
            
        except Exception as e:
            logger.error(f"Error sending notification {notification_type.value}: {e}")
    
    async def send_trade_entry_notification(self, trade_data: Dict):
        """Send trade entry notification."""
        try:
            # Add PnL symbol
            trade_data['pnl_symbol'] = "📈" if trade_data.get('pnl', 0) >= 0 else "📉"
            
            await self.send_notification(NotificationType.TRADE_ENTRY, trade_data)
            
        except Exception as e:
            logger.error(f"Error sending trade entry notification: {e}")
    
    async def send_trade_exit_notification(self, trade_data: Dict):
        """Send trade exit notification."""
        try:
            # Add PnL symbol
            trade_data['pnl_symbol'] = "📈" if trade_data.get('pnl', 0) >= 0 else "📉"
            
            await self.send_notification(NotificationType.TRADE_EXIT, trade_data)
            
        except Exception as e:
            logger.error(f"Error sending trade exit notification: {e}")
    
    async def send_high_confidence_notification(self, signal_data: Dict):
        """Send high confidence signal notification."""
        try:
            # Determine signal strength
            confidence = signal_data.get('confidence', 0)
            if confidence >= 90:
                signal_data['signal_strength'] = "Very Strong"
            elif confidence >= 80:
                signal_data['signal_strength'] = "Strong"
            elif confidence >= 70:
                signal_data['signal_strength'] = "Moderate"
            else:
                signal_data['signal_strength'] = "Weak"
            
            await self.send_notification(NotificationType.HIGH_CONFIDENCE, signal_data)
            
        except Exception as e:
            logger.error(f"Error sending high confidence notification: {e}")
    
    async def send_model_retrained_notification(self, model_data: Dict):
        """Send model retraining notification."""
        try:
            # Calculate accuracy change
            accuracy_change = model_data.get('accuracy', 0) - model_data.get('previous_accuracy', 0)
            model_data['accuracy_change'] = accuracy_change
            
            # Calculate next retraining time
            next_retraining = datetime.now() + timedelta(hours=self.config.model.retraining_interval_hours)
            model_data['next_retraining'] = next_retraining.strftime('%Y-%m-%d %H:%M:%S')
            
            await self.send_notification(NotificationType.MODEL_RETRAINED, model_data)
            
        except Exception as e:
            logger.error(f"Error sending model retrained notification: {e}")
    
    async def send_risk_alert_notification(self, risk_data: Dict):
        """Send risk alert notification."""
        try:
            # Determine severity
            drawdown = risk_data.get('current_drawdown', 0)
            if drawdown >= 15:
                risk_data['severity'] = "🔴 Critical"
            elif drawdown >= 10:
                risk_data['severity'] = "🟡 High"
            elif drawdown >= 5:
                risk_data['severity'] = "🟠 Medium"
            else:
                risk_data['severity'] = "🟢 Low"
            
            await self.send_notification(NotificationType.RISK_ALERT, risk_data)
            
        except Exception as e:
            logger.error(f"Error sending risk alert notification: {e}")
    
    async def send_daily_summary_notification(self, summary_data: Dict):
        """Send daily performance summary."""
        try:
            # Add PnL symbols
            summary_data['daily_pnl_symbol'] = "📈" if summary_data.get('daily_pnl', 0) >= 0 else "📉"
            
            await self.send_notification(NotificationType.DAILY_SUMMARY, summary_data)
            
        except Exception as e:
            logger.error(f"Error sending daily summary notification: {e}")
    
    async def send_performance_update_notification(self, performance_data: Dict):
        """Send performance update notification."""
        try:
            await self.send_notification(NotificationType.PERFORMANCE_UPDATE, performance_data)
            
        except Exception as e:
            logger.error(f"Error sending performance update notification: {e}")
    
    async def send_signal_generated_notification(self, signal_data: Dict):
        """Send signal generated notification."""
        try:
            # Determine risk level
            uncertainty = signal_data.get('uncertainty', 0)
            if uncertainty <= 1:
                signal_data['risk_level'] = "🟢 Low"
            elif uncertainty <= 2:
                signal_data['risk_level'] = "🟡 Medium"
            elif uncertainty <= 3:
                signal_data['risk_level'] = "🟠 High"
            else:
                signal_data['risk_level'] = "🔴 Very High"
            
            # Determine signal strength
            confidence = signal_data.get('confidence', 0)
            if confidence >= 85:
                signal_data['signal_strength'] = "Very Strong"
            elif confidence >= 75:
                signal_data['signal_strength'] = "Strong"
            elif confidence >= 65:
                signal_data['signal_strength'] = "Moderate"
            else:
                signal_data['signal_strength'] = "Weak"
            
            await self.send_notification(NotificationType.SIGNAL_GENERATED, signal_data)
            
        except Exception as e:
            logger.error(f"Error sending signal generated notification: {e}")
    
    async def send_custom_notification(self, message: str, parse_mode: str = 'Markdown'):
        """Send a custom notification message."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            
            logger.info("Sent custom notification")
            
        except Exception as e:
            logger.error(f"Error sending custom notification: {e}")
    
    async def start_notification_queue(self):
        """Start the notification queue processor."""
        self.is_running = True
        logger.info("Starting notification queue processor")
        
        while self.is_running:
            try:
                # Process notifications from queue
                notification = await asyncio.wait_for(self.notification_queue.get(), timeout=1.0)
                
                notification_type = notification['type']
                data = notification['data']
                
                await self.send_notification(notification_type, data)
                
            except asyncio.TimeoutError:
                # No notifications in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
    
    async def stop_notification_queue(self):
        """Stop the notification queue processor."""
        self.is_running = False
        logger.info("Stopping notification queue processor")
    
    def queue_notification(self, notification_type: NotificationType, data: Dict):
        """Queue a notification for processing."""
        try:
            self.notification_queue.put_nowait({
                'type': notification_type,
                'data': data
            })
        except asyncio.QueueFull:
            logger.warning("Notification queue is full, dropping notification")
    
    def format_price(self, price: float) -> str:
        """Format price for display."""
        return f"${price:,.2f}"
        
    def format_percentage(self, percentage: float) -> str:
        """Format percentage for display."""
        return f"{percentage:+.2f}%"
    
    def format_duration(self, duration_seconds: int) -> str:
        """Format duration for display."""
        if duration_seconds < 60:
            return f"{duration_seconds}s"
        elif duration_seconds < 3600:
            minutes = duration_seconds // 60
            seconds = duration_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for display."""
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    def should_send_notification(self, notification_type: NotificationType) -> bool:
        """Check if notification should be sent based on user preferences."""
        telegram_config = self.config.telegram
        
        if notification_type == NotificationType.TRADE_ENTRY or notification_type == NotificationType.TRADE_EXIT:
            return telegram_config.trade_notifications
        elif notification_type == NotificationType.HIGH_CONFIDENCE:
            return telegram_config.high_confidence_alerts
        elif notification_type == NotificationType.MODEL_RETRAINED:
            return telegram_config.model_retrained_alerts
        elif notification_type == NotificationType.RISK_ALERT:
            return telegram_config.risk_alerts
        elif notification_type == NotificationType.DAILY_SUMMARY:
            return telegram_config.daily_summary
        elif notification_type == NotificationType.PERFORMANCE_UPDATE:
            return telegram_config.performance_updates
        else:
            return True  # Default to sending other notifications 