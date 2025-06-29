"""
Main Telegram bot implementation for cryptocurrency trading bot.
Handles bot initialization, command routing, and user interaction.
"""

import logging
import asyncio
from datetime import datetime, timedelta
import telegram
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from typing import Dict, Optional
from utils.config import load_config
from telegram_bot.handlers import BotHandlers
from telegram_bot.notifications import NotificationManager, NotificationType

logger = logging.getLogger(__name__)

class CryptoTradingBot:
    """Main Telegram bot class for crypto trading."""
    
    def __init__(self, config_path: str = 'config/default.yaml'):
        self.config = load_config(config_path)
        # Defer TradingEngine initialization to avoid circular import
        self.trading_engine = None 
        self.application: Optional[Application] = None
        self.handlers: Optional[BotHandlers] = None
        self.notification_manager: Optional[NotificationManager] = None
        self.is_running = False
        self.authorized_users = set()
        self.setup_bot()
        
    def setup_bot(self):
        """Initialize bot application and handlers."""
        try:
            # Initialize application
            self.application = Application.builder().token(self.config.telegram.token).build()
            
            # Initialize handlers and notification manager
            self.handlers = BotHandlers(self.trading_engine, self.config)
            self.notification_manager = NotificationManager(
                self.application.bot, 
                self.config.telegram.chat_id, 
                self.config
            )
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("settings", self.settings_command))
            self.application.add_handler(CommandHandler("history", self.history_command))
            self.application.add_handler(CommandHandler("pause", self.pause_command))
            self.application.add_handler(CommandHandler("resume", self.resume_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            
            # Add callback query handler for inline keyboards
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            
            # Add error handler
            self.application.add_error_handler(self.error_handler)
            
            logger.info("Telegram bot setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Telegram bot: {e}")
            raise
        
    async def start_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        try:
            user_id = update.effective_user.id
            username = update.effective_user.username or "Unknown"
            
            # Check authorization
            if not self._is_authorized(user_id):
                await update.message.reply_text(
                    "❌ Access denied. You are not authorized to use this bot."
                )
                return
            
            welcome_message = f"""
🤖 **Bayesian Crypto Trading Bot**

Welcome, @{username}! 

This bot uses Bayesian LSTM models to predict Bitcoin price movements with uncertainty quantification.

**Available Commands:**
• `/status` - View current bot status and portfolio
• `/settings` - Configure bot parameters
• `/history` - View trading history
• `/pause` - Pause trading
• `/resume` - Resume trading
• `/help` - Show this help message

**Current Status:** {'🟢 Active' if self.is_running else '🔴 Paused'}

Type `/help` for more information.
            """
            
            await update.message.reply_text(welcome_message, parse_mode='Markdown')
            logger.info(f"User {username} (ID: {user_id}) started the bot")
            
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await update.message.reply_text("❌ An error occurred. Please try again.")
        
    async def status_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            if not self._is_authorized(update.effective_user.id):
                await update.message.reply_text("❌ Access denied.")
                return
            
            # Get current status
            portfolio_status = self.trading_engine.get_portfolio_status()
            model_status = self.trading_engine.get_model_status()
            trading_status = self.trading_engine.get_trading_status()
            
            status_message = f"""
📊 **Bot Status Report**

**🤖 Bot Status:** {'🟢 Running' if self.is_running else '🔴 Paused'}

**💰 Portfolio:**
• Total Capital: ${portfolio_status['total_capital']:,.2f}
• Available Capital: ${portfolio_status['available_capital']:,.2f}
• Active Positions: {portfolio_status['active_positions']}
• Total PnL: ${portfolio_status['total_pnl']:,.2f} ({portfolio_status['total_pnl_pct']:+.2f}%)

**🧠 Model Status:**
• Last Prediction: {model_status['last_prediction_time']}
• Prediction: ${model_status['last_prediction']:,.2f}
• Confidence: {model_status['confidence']:.1f}%
• Uncertainty: {model_status['uncertainty']:.1f}%

**📈 Trading Status:**
• Current Signal: {trading_status['current_signal']}
• Last Trade: {trading_status['last_trade_time']}
• Daily Trades: {trading_status['daily_trades']}
• Win Rate: {trading_status['win_rate']:.1f}%

**⏰ Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            await update.message.reply_text(status_message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ An error occurred while fetching status.")
        
    async def settings_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command."""
        try:
            if not self._is_authorized(update.effective_user.id):
                await update.message.reply_text("❌ Access denied.")
                return
            
            # Create settings keyboard
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
            
            await update.message.reply_text(settings_message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in settings command: {e}")
            await update.message.reply_text("❌ An error occurred while loading settings.")
        
    async def history_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command."""
        try:
            if not self._is_authorized(update.effective_user.id):
                await update.message.reply_text("❌ Access denied.")
                return
            
            # Get trading history
            history = self.trading_engine.get_trading_history(limit=10)
            
            if not history:
                await update.message.reply_text("📊 No trading history available.")
                return
            
            history_message = "📊 **Recent Trading History**\n\n"
            
            for trade in history:
                pnl_symbol = "📈" if trade['pnl'] >= 0 else "📉"
                history_message += f"""
{pnl_symbol} **{trade['position_type']}** - {trade['entry_time']}
Entry: ${trade['entry_price']:,.2f} | Exit: ${trade['exit_price']:,.2f}
PnL: ${trade['pnl']:,.2f} ({trade['pnl_pct']:+.2f}%)
Reason: {trade['exit_reason']}
---
                """
            
            # Add summary
            total_trades = len(history)
            winning_trades = sum(1 for trade in history if trade['pnl'] > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum(trade['pnl'] for trade in history)
            
            summary = f"""
**Summary (Last {total_trades} trades):**
• Win Rate: {win_rate:.1f}%
• Total PnL: ${total_pnl:,.2f}
• Average PnL: ${total_pnl/total_trades:,.2f}
            """
            
            history_message += summary
            
            await update.message.reply_text(history_message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in history command: {e}")
            await update.message.reply_text("❌ An error occurred while fetching history.")
        
    async def pause_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command."""
        try:
            if not self._is_authorized(update.effective_user.id):
                await update.message.reply_text("❌ Access denied.")
                return
            
            if not self.is_running:
                await update.message.reply_text("🔴 Bot is already paused.")
                return
            
            # Pause trading
            self.is_running = False
            self.trading_engine.pause_trading()
            
            await update.message.reply_text(
                "⏸️ **Trading Paused**\n\n"
                "The bot has been paused and will not execute new trades. "
                "Existing positions will remain open. Use `/resume` to restart trading.",
                parse_mode='Markdown'
            )
            
            logger.info(f"Trading paused by user {update.effective_user.username}")
            
        except Exception as e:
            logger.error(f"Error in pause command: {e}")
            await update.message.reply_text("❌ An error occurred while pausing trading.")
        
    async def resume_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        try:
            if not self._is_authorized(update.effective_user.id):
                await update.message.reply_text("❌ Access denied.")
                return
            
            if self.is_running:
                await update.message.reply_text("🟢 Bot is already running.")
                return
            
            # Resume trading
            self.is_running = True
            self.trading_engine.resume_trading()
            
            await update.message.reply_text(
                "▶️ **Trading Resumed**\n\n"
                "The bot is now active and will execute trades based on signals. "
                "Use `/pause` to stop trading again.",
                parse_mode='Markdown'
            )
            
            logger.info(f"Trading resumed by user {update.effective_user.username}")
            
        except Exception as e:
            logger.error(f"Error in resume command: {e}")
            await update.message.reply_text("❌ An error occurred while resuming trading.")
        
    async def help_command(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        try:
            if not self._is_authorized(update.effective_user.id):
                await update.message.reply_text("❌ Access denied.")
                return
            
            help_message = """
🤖 **Bayesian Crypto Trading Bot - Help**

**Available Commands:**

📊 **Status & Information**
• `/start` - Welcome message and bot overview
• `/status` - Current bot status, portfolio, and performance
• `/history` - Recent trading history and statistics

⚙️ **Control & Settings**
• `/settings` - Configure bot parameters and preferences
• `/pause` - Pause trading (existing positions remain open)
• `/resume` - Resume trading after pause

📚 **Help & Support**
• `/help` - Show this help message

**Key Features:**
• Bayesian LSTM price prediction with uncertainty quantification
• Intelligent position sizing based on confidence levels
• Dynamic risk management with stop-loss and take-profit
• Real-time notifications for trades and alerts
• Paper trading mode for testing

**Risk Warning:**
Cryptocurrency trading involves significant risk. This bot is for educational purposes. Never invest more than you can afford to lose.

For support or questions, contact the bot administrator.
            """
            
            await update.message.reply_text(help_message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in help command: {e}")
            await update.message.reply_text("❌ An error occurred while loading help.")
        
    async def handle_callback(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards."""
        try:
            query = update.callback_query
            await query.answer()
            
            if not self._is_authorized(update.effective_user.id):
                await query.edit_message_text("❌ Access denied.")
                return
            
            # Handle different callback types
            if query.data.startswith("settings_"):
                await self.handlers.handle_settings_callback(update, context)
            elif query.data.startswith("position_"):
                await self.handlers.handle_position_callback(update, context)
            elif query.data.startswith("trade_"):
                await self.handlers.handle_trade_callback(update, context)
            else:
                await query.edit_message_text("❌ Unknown callback type.")
                
        except Exception as e:
            logger.error(f"Error in callback handler: {e}")
            await update.callback_query.edit_message_text("❌ An error occurred.")
        
    async def error_handler(self, update: telegram.Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle bot errors."""
        logger.error(f"Bot error: {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An unexpected error occurred. Please try again later."
            )
    
    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot."""
        # For now, allow all users. In production, implement proper authorization
        return True
    
    async def start(self):
        """Start the bot."""
        try:
            logger.info("Starting Telegram bot...")
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            self.is_running = True
            logger.info("Telegram bot started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            raise
    
    async def stop(self):
        """Stop the bot."""
        try:
            logger.info("Stopping Telegram bot...")
            self.is_running = False
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
            raise
    
    async def run(self):
        """Run the bot and the trading engine concurrently."""
        # Initialize trading engine here
        from trading.engine import TradingEngine
        self.trading_engine = TradingEngine(self.config.trading)

        # Start the trading engine in the background
        trading_task = asyncio.create_task(self.run_trading_engine())
        
        # Start the bot
        await self.application.run_polling()
        
        # Ensure the trading task is cancelled on exit
        trading_task.cancel()
        try:
            await trading_task
        except asyncio.CancelledError:
            pass

    async def run_trading_engine(self):
        """Periodically run the trading engine logic."""
        while True:
            try:
                # Placeholder for fetching data and processing
                await asyncio.sleep(self.config.trading.update_interval)
                
            except Exception as e:
                logging.error(f"Error in trading engine: {e}")
                await asyncio.sleep(60)

if __name__ == '__main__':
    bot = CryptoTradingBot()
    asyncio.run(bot.run()) 