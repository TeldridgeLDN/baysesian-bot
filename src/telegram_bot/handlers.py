"""
Command and callback handlers for Telegram bot.
Handles user commands, button callbacks, and message processing.
"""

import logging
from datetime import datetime, timedelta
import telegram
from telegram.ext import ContextTypes
from typing import Dict, List, Optional
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

class BotHandlers:
    """Handles all bot commands and callbacks."""
    
    def __init__(self, trading_engine, config: ConfigManager):
        self.trading_engine = trading_engine
        self.config = config
        
    async def handle_settings_callback(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle settings button callbacks."""
        try:
            query = update.callback_query
            callback_data = query.data
            
            if callback_data == "settings_risk":
                await self._show_risk_settings(query)
            elif callback_data == "settings_model":
                await self._show_model_settings(query)
            elif callback_data == "settings_position":
                await self._show_position_settings(query)
            elif callback_data == "settings_notifications":
                await self._show_notification_settings(query)
            elif callback_data == "settings_mode":
                await self._show_trading_mode_settings(query)
            elif callback_data == "settings_performance":
                await self._show_performance_settings(query)
            elif callback_data.startswith("update_"):
                await self._handle_setting_update(query, callback_data)
            elif callback_data == "back_to_settings":
                await self._show_main_settings(query)
            else:
                await query.edit_message_text("❌ Unknown settings option.")
                
        except Exception as e:
            logger.error(f"Error in settings callback: {e}")
            await update.callback_query.edit_message_text("❌ An error occurred.")
    
    async def handle_position_callback(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle position management callbacks."""
        try:
            query = update.callback_query
            callback_data = query.data
            
            if callback_data.startswith("close_position_"):
                position_id = callback_data.split("_")[-1]
                await self._handle_manual_close(query, position_id)
            elif callback_data.startswith("view_position_"):
                position_id = callback_data.split("_")[-1]
                await self._show_position_details(query, position_id)
            else:
                await query.edit_message_text("❌ Unknown position action.")
                
        except Exception as e:
            logger.error(f"Error in position callback: {e}")
            await update.callback_query.edit_message_text("❌ An error occurred.")
    
    async def handle_trade_callback(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle trade-related callbacks."""
        try:
            query = update.callback_query
            callback_data = query.data
            
            if callback_data == "retrain_model":
                await self._handle_retrain_command(query)
            elif callback_data == "view_signals":
                await self._show_recent_signals(query)
            elif callback_data == "view_performance":
                await self._show_performance_metrics(query)
            else:
                await query.edit_message_text("❌ Unknown trade action.")
                
        except Exception as e:
            logger.error(f"Error in trade callback: {e}")
            await update.callback_query.edit_message_text("❌ An error occurred.")
    
    async def _show_risk_settings(self, query):
        """Show risk management settings."""
        risk_config = self.config.trading
        
        keyboard = [
            [
                telegram.InlineKeyboardButton(f"Stop Loss: {risk_config.stop_loss_pct}%", callback_data="update_stop_loss"),
                telegram.InlineKeyboardButton(f"Take Profit: {risk_config.take_profit_pct}%", callback_data="update_take_profit")
            ],
            [
                telegram.InlineKeyboardButton(f"Max Drawdown: {risk_config.max_drawdown_pct}%", callback_data="update_drawdown"),
                telegram.InlineKeyboardButton(f"Risk per Trade: {risk_config.max_risk_per_trade_pct}%", callback_data="update_risk_per_trade")
            ],
            [
                telegram.InlineKeyboardButton(f"Max Positions: {risk_config.max_positions}", callback_data="update_max_positions"),
                telegram.InlineKeyboardButton(f"Position Timeout: {risk_config.position_timeout_hours}h", callback_data="update_timeout")
            ],
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="back_to_settings")]
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        message = """
📊 **Risk Management Settings**

Configure risk parameters to protect your capital:

• **Stop Loss** - Automatic exit on loss
• **Take Profit** - Automatic exit on profit
• **Max Drawdown** - Maximum allowed portfolio decline
• **Risk per Trade** - Maximum capital risked per trade
• **Max Positions** - Maximum concurrent positions
• **Position Timeout** - Auto-close positions after time limit

Click any setting to modify it.
        """
        
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def _show_model_settings(self, query):
        """Show model configuration settings."""
        model_config = self.config.model
        
        keyboard = [
            [
                telegram.InlineKeyboardButton(f"Confidence Threshold: {model_config.confidence_threshold}%", callback_data="update_confidence_threshold"),
                telegram.InlineKeyboardButton(f"Uncertainty Limit: {model_config.uncertainty_limit}%", callback_data="update_uncertainty_limit")
            ],
            [
                telegram.InlineKeyboardButton(f"Signal Threshold: {model_config.signal_threshold}%", callback_data="update_signal_threshold"),
                telegram.InlineKeyboardButton(f"Monte Carlo Samples: {model_config.monte_carlo_samples}", callback_data="update_mc_samples")
            ],
            [
                telegram.InlineKeyboardButton(f"Retraining Interval: {model_config.retraining_interval_hours}h", callback_data="update_retraining_interval"),
                telegram.InlineKeyboardButton(f"Training Window: {model_config.training_window_days}d", callback_data="update_training_window")
            ],
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="back_to_settings")]
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        message = """
🤖 **Model Settings**

Configure the Bayesian LSTM model parameters:

• **Confidence Threshold** - Minimum confidence for signals
• **Uncertainty Limit** - Maximum uncertainty for trades
• **Signal Threshold** - Minimum price change for signals
• **Monte Carlo Samples** - Number of prediction samples
• **Retraining Interval** - How often to retrain the model
• **Training Window** - Historical data window for training

Click any setting to modify it.
        """
        
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def _show_position_settings(self, query):
        """Show position sizing settings."""
        trading_config = self.config.trading
        
        keyboard = [
            [
                telegram.InlineKeyboardButton(f"Max Position Size: {trading_config.max_position_size_pct}%", callback_data="update_max_position_size"),
                telegram.InlineKeyboardButton(f"Position Scaling: {trading_config.position_scaling}", callback_data="update_position_scaling")
            ],
            [
                telegram.InlineKeyboardButton(f"Min Position Size: {trading_config.min_position_size_pct}%", callback_data="update_min_position_size"),
                telegram.InlineKeyboardButton(f"Confidence Scaling: {trading_config.confidence_scaling}", callback_data="update_confidence_scaling")
            ],
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="back_to_settings")]
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        message = """
💰 **Position Sizing Settings**

Configure how the bot sizes positions:

• **Max Position Size** - Maximum capital per position
• **Position Scaling** - How position size scales with confidence
• **Min Position Size** - Minimum position size
• **Confidence Scaling** - Position size based on prediction confidence

Click any setting to modify it.
        """
        
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def _show_notification_settings(self, query):
        """Show notification settings."""
        telegram_config = self.config.telegram
        
        keyboard = [
            [
                telegram.InlineKeyboardButton(f"Trade Notifications: {'✅' if telegram_config.trade_notifications else '❌'}", callback_data="update_trade_notifications"),
                telegram.InlineKeyboardButton(f"High Confidence: {'✅' if telegram_config.high_confidence_alerts else '❌'}", callback_data="update_high_confidence")
            ],
            [
                telegram.InlineKeyboardButton(f"Daily Summary: {'✅' if telegram_config.daily_summary else '❌'}", callback_data="update_daily_summary"),
                telegram.InlineKeyboardButton(f"Risk Alerts: {'✅' if telegram_config.risk_alerts else '❌'}", callback_data="update_risk_alerts")
            ],
            [
                telegram.InlineKeyboardButton(f"Model Retrained: {'✅' if telegram_config.model_retrained_alerts else '❌'}", callback_data="update_model_retrained"),
                telegram.InlineKeyboardButton(f"Performance Updates: {'✅' if telegram_config.performance_updates else '❌'}", callback_data="update_performance")
            ],
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="back_to_settings")]
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        message = """
🔔 **Notification Settings**

Configure which notifications you want to receive:

• **Trade Notifications** - Entry/exit alerts
• **High Confidence** - High confidence signal alerts
• **Daily Summary** - Daily performance summary
• **Risk Alerts** - Risk management alerts
• **Model Retrained** - Model retraining notifications
• **Performance Updates** - Performance metric updates

Click any setting to toggle it.
        """
        
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def _show_trading_mode_settings(self, query):
        """Show trading mode settings."""
        trading_config = self.config.trading
        
        mode_text = "Paper Trading" if trading_config.paper_trading else "Live Trading"
        mode_emoji = "📝" if trading_config.paper_trading else "💰"
        
        keyboard = [
            [
                telegram.InlineKeyboardButton(f"Trading Mode: {mode_emoji} {mode_text}", callback_data="update_trading_mode"),
                telegram.InlineKeyboardButton(f"Auto Trading: {'✅' if trading_config.auto_trading else '❌'}", callback_data="update_auto_trading")
            ],
            [
                telegram.InlineKeyboardButton(f"Conservative Mode: {'✅' if trading_config.conservative_mode else '❌'}", callback_data="update_conservative_mode"),
                telegram.InlineKeyboardButton(f"Backtesting: {'✅' if trading_config.backtesting_mode else '❌'}", callback_data="update_backtesting")
            ],
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="back_to_settings")]
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        message = """
⚙️ **Trading Mode Settings**

Configure the bot's trading behavior:

• **Trading Mode** - Paper trading vs live trading
• **Auto Trading** - Automatic trade execution
• **Conservative Mode** - Reduced risk settings
• **Backtesting** - Historical performance testing

⚠️ **Warning:** Live trading involves real money risk.

Click any setting to modify it.
        """
        
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def _show_performance_settings(self, query):
        """Show performance target settings."""
        trading_config = self.config.trading
        
        keyboard = [
            [
                telegram.InlineKeyboardButton(f"Target ROI: {trading_config.target_roi_pct}%", callback_data="update_target_roi"),
                telegram.InlineKeyboardButton(f"Max Drawdown: {trading_config.max_drawdown_pct}%", callback_data="update_max_drawdown")
            ],
            [
                telegram.InlineKeyboardButton(f"Win Rate Target: {trading_config.target_win_rate_pct}%", callback_data="update_win_rate_target"),
                telegram.InlineKeyboardButton(f"Sharpe Ratio Target: {trading_config.target_sharpe_ratio}", callback_data="update_sharpe_target")
            ],
            [telegram.InlineKeyboardButton("🔙 Back", callback_data="back_to_settings")]
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        message = """
📈 **Performance Settings**

Set performance targets and limits:

• **Target ROI** - Annual return target
• **Max Drawdown** - Maximum portfolio decline
• **Win Rate Target** - Target percentage of winning trades
• **Sharpe Ratio Target** - Risk-adjusted return target

These settings help monitor performance and trigger alerts.

Click any setting to modify it.
        """
        
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def _show_main_settings(self, query):
        """Show main settings menu."""
        keyboard = [
            [
                telegram.InlineKeyboardButton("📊 Risk Settings", callback_data="settings_risk"),
                telegram.InlineKeyboardButton("🤖 Model Settings", callback_data="settings_model")
            ],
            [
                telegram.InlineKeyboardButton("💰 Position Sizing", callback_data="settings_position"),
                telegram.InlineKeyboardButton("🔔 Notifications", callback_data="settings_notifications")
            ],
            [
                telegram.InlineKeyboardButton("⚙️ Trading Mode", callback_data="settings_mode"),
                telegram.InlineKeyboardButton("📈 Performance", callback_data="settings_performance")
            ]
        ]
        reply_markup = telegram.InlineKeyboardMarkup(keyboard)
        
        settings_message = """
⚙️ **Bot Settings**

Select a category to configure:

• **Risk Settings** - Stop-loss, take-profit, drawdown limits
• **Model Settings** - Confidence thresholds, uncertainty limits
• **Position Sizing** - Maximum position size, capital allocation
• **Notifications** - Alert preferences, notification frequency
• **Trading Mode** - Paper trading vs live trading
• **Performance** - Performance metrics and targets
        """
        
        await query.edit_message_text(settings_message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def _handle_setting_update(self, query, callback_data):
        """Handle setting update requests."""
        # This would typically open a conversation to update the setting
        # For now, just acknowledge the request
        setting_name = callback_data.replace("update_", "")
        
        await query.edit_message_text(
            f"⚙️ **Setting Update**\n\n"
            f"To update **{setting_name}**, please contact the bot administrator.\n\n"
            f"This feature will be implemented in a future update.",
            parse_mode='Markdown'
        )
    
    async def _handle_manual_close(self, query, position_id):
        """Handle manual position closing."""
        try:
            # Close the position
            result = self.trading_engine.close_position(position_id, "Manual close")
            
            if result['success']:
                await query.edit_message_text(
                    f"✅ **Position Closed**\n\n"
                    f"Position {position_id} has been closed.\n"
                    f"PnL: ${result['pnl']:,.2f} ({result['pnl_pct']:+.2f}%)",
                    parse_mode='Markdown'
                )
            else:
                await query.edit_message_text(
                    f"❌ **Error Closing Position**\n\n"
                    f"Could not close position {position_id}.\n"
                    f"Error: {result['error']}",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in manual close: {e}")
            await query.edit_message_text("❌ An error occurred while closing the position.")
    
    async def _show_position_details(self, query, position_id):
        """Show detailed position information."""
        try:
            position = self.trading_engine.get_position(position_id)
            
            if not position:
                await query.edit_message_text("❌ Position not found.")
                return
            
            # Calculate current PnL
            current_price = self.trading_engine.get_current_price()
            if position['position_type'] == 'LONG':
                pnl = (current_price - position['entry_price']) * position['size']
                pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
                pnl_pct = ((position['entry_price'] - current_price) / position['entry_price']) * 100
            
            pnl_symbol = "📈" if pnl >= 0 else "📉"
            
            keyboard = [
                [telegram.InlineKeyboardButton("❌ Close Position", callback_data=f"close_position_{position_id}")],
                [telegram.InlineKeyboardButton("🔙 Back", callback_data="back_to_positions")]
            ]
            reply_markup = telegram.InlineKeyboardMarkup(keyboard)
            
            message = f"""
{pnl_symbol} **Position Details**

**Position ID:** {position_id}
**Type:** {position['position_type']}
**Size:** {position['size']:.4f} BTC
**Entry Price:** ${position['entry_price']:,.2f}
**Current Price:** ${current_price:,.2f}
**Entry Time:** {position['entry_time']}
**Duration:** {self._calculate_duration(position['entry_time'])}
**Current PnL:** ${pnl:,.2f} ({pnl_pct:+.2f}%)

**Stop Loss:** ${position['stop_loss']:,.2f}
**Take Profit:** ${position['take_profit']:,.2f}
            """
            
            await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error showing position details: {e}")
            await query.edit_message_text("❌ An error occurred while fetching position details.")
    
    async def _handle_retrain_command(self, query):
        """Handle model retraining request."""
        try:
            await query.edit_message_text("🔄 **Retraining Model**\n\nPlease wait while the model is being retrained...")
            
            # Trigger model retraining
            result = self.trading_engine.retrain_model()
            
            if result['success']:
                await query.edit_message_text(
                    f"✅ **Model Retrained**\n\n"
                    f"Model has been successfully retrained.\n"
                    f"New accuracy: {result['accuracy']:.2f}%\n"
                    f"Training time: {result['training_time']:.1f} seconds",
                    parse_mode='Markdown'
                )
            else:
                await query.edit_message_text(
                    f"❌ **Retraining Failed**\n\n"
                    f"Could not retrain the model.\n"
                    f"Error: {result['error']}",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in retrain command: {e}")
            await query.edit_message_text("❌ An error occurred while retraining the model.")
    
    async def _show_recent_signals(self, query):
        """Show recent trading signals."""
        try:
            signals = self.trading_engine.get_recent_signals(limit=10)
            
            if not signals:
                await query.edit_message_text("📊 No recent signals available.")
                return
            
            message = "📊 **Recent Trading Signals**\n\n"
            
            for signal in signals:
                signal_emoji = "📈" if signal['signal_type'] == 'LONG' else "📉" if signal['signal_type'] == 'SHORT' else "⏸️"
                message += f"""
{signal_emoji} **{signal['signal_type']}** - {signal['timestamp']}
Price: ${signal['current_price']:,.2f} | Prediction: ${signal['prediction']:,.2f}
Confidence: {signal['confidence']:.1f}% | Uncertainty: {signal['uncertainty']:.1f}%
Reason: {signal['reasoning']}
---
                """
            
            await query.edit_message_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing recent signals: {e}")
            await query.edit_message_text("❌ An error occurred while fetching signals.")
    
    async def _show_performance_metrics(self, query):
        """Show performance metrics."""
        try:
            metrics = self.trading_engine.get_performance_metrics()
            
            message = f"""
📈 **Performance Metrics**

**Overall Performance:**
• Total Return: {metrics['total_return']:+.2f}%
• Annualized Return: {metrics['annualized_return']:+.2f}%
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Max Drawdown: {metrics['max_drawdown']:.2f}%

**Trading Statistics:**
• Total Trades: {metrics['total_trades']}
• Winning Trades: {metrics['winning_trades']}
• Win Rate: {metrics['win_rate']:.1f}%
• Average Win: {metrics['avg_win']:.2f}%
• Average Loss: {metrics['avg_loss']:.2f}%
• Profit Factor: {metrics['profit_factor']:.2f}

**Risk Metrics:**
• Volatility: {metrics['volatility']:.2f}%
• VaR (95%): {metrics['var_95']:.2f}%
• Expected Shortfall: {metrics['expected_shortfall']:.2f}%
• Calmar Ratio: {metrics['calmar_ratio']:.2f}

**Model Performance:**
• Prediction Accuracy: {metrics['prediction_accuracy']:.1f}%
• Average Confidence: {metrics['avg_confidence']:.1f}%
• Average Uncertainty: {metrics['avg_uncertainty']:.1f}%
            """
            
            await query.edit_message_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error showing performance metrics: {e}")
            await query.edit_message_text("❌ An error occurred while fetching performance metrics.")
    
    def _calculate_duration(self, entry_time: str) -> str:
        """Calculate duration since entry time."""
        try:
            entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            now = datetime.now(entry_dt.tzinfo)
            duration = now - entry_dt
            
            if duration.days > 0:
                return f"{duration.days}d {duration.seconds // 3600}h"
            elif duration.seconds > 3600:
                return f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
            else:
                return f"{duration.seconds // 60}m"
        except:
            return "Unknown"
    
    async def handle_error(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle bot errors."""
        logger.error(f"Bot error: {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An unexpected error occurred. Please try again later."
            ) 