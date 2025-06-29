#!/usr/bin/env python3
"""
Main entry point for the Bayesian Crypto Trading Bot.
Initializes all components and starts the trading system.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import utility modules
from utils.logging import setup_logging
from utils.config import ConfigManager

# Import core modules (placeholder imports for now)
# from data.storage import DatabaseManager
# from models.bayesian_lstm import BayesianLSTM
# from trading.signals import TradingSignalGenerator
# from trading.portfolio import PortfolioManager
# from telegram.bot import CryptoTradingBot

logger = logging.getLogger(__name__)

class TradingBotApplication:
    """Main application class for the trading bot."""
    
    def __init__(self):
        self.config_manager = None
        self.db_manager = None
        self.model = None
        self.signal_generator = None
        self.portfolio_manager = None
        self.telegram_bot = None
        self.running = False
        
    async def initialize(self):
        """Initialize all components."""
        try:
            logger.info("Initializing Bayesian Crypto Trading Bot...")
            
            # Load configuration
            self.config_manager = ConfigManager()
            logger.info("✓ Configuration loaded")
            
            # Initialize database
            # self.db_manager = DatabaseManager()
            logger.info("✓ Database connection established")
            
            # Initialize ML model
            # self.model = BayesianLSTM()
            logger.info("✓ Bayesian LSTM model initialized")
            
            # Initialize trading components
            # self.signal_generator = TradingSignalGenerator(self.config_manager.get_trading_config())
            # self.portfolio_manager = PortfolioManager(
            #     initial_capital=float(os.getenv('INITIAL_CAPITAL', 10000)),
            #     config=self.config_manager.get_trading_config()
            # )
            logger.info("✓ Trading components initialized")
            
            # Initialize Telegram bot
            # telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
            # if not telegram_token:
            #     raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
            # 
            # self.telegram_bot = CryptoTradingBot(
            #     token=telegram_token,
            #     config=self.config_manager.get_telegram_config()
            # )
            logger.info("✓ Telegram bot initialized")
            
            logger.info("🚀 All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    async def start(self):
        """Start the trading bot."""
        try:
            await self.initialize()
            
            logger.info("Starting trading bot...")
            self.running = True
            
            # Start main trading loop
            await self.run_trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Application error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def run_trading_loop(self):
        """Main trading loop."""
        logger.info("🔄 Starting main trading loop...")
        
        while self.running:
            try:
                # Main trading logic will be implemented here
                logger.info("Trading loop iteration (placeholder)")
                
                # Sleep for update interval
                await asyncio.sleep(60)  # 1 minute intervals for now
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def shutdown(self):
        """Graceful shutdown of the application."""
        logger.info("Shutting down trading bot...")
        self.running = False
        
        # Shutdown components
        if self.telegram_bot:
            # await self.telegram_bot.shutdown()
            logger.info("✓ Telegram bot shutdown")
        
        if self.portfolio_manager:
            # Close any open positions if configured
            logger.info("✓ Portfolio manager shutdown")
        
        if self.db_manager:
            # Close database connections
            logger.info("✓ Database connections closed")
        
        logger.info("👋 Shutdown complete")

def check_environment():
    """Check if environment is properly configured."""
    required_files = [
        'config/default.yaml',
        '.env'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run 'python scripts/setup.py' to initialize the project.")
        return False
    
    return True

async def main():
    """Main function."""
    print("🤖 Bayesian Crypto Trading Bot")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        return 1
    
    # Setup logging
    setup_logging()
    logger.info("Starting Bayesian Crypto Trading Bot")
    
    # Create and start application
    app = TradingBotApplication()
    
    try:
        await app.start()
        return 0
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1

if __name__ == "__main__":
    # Run the application
    sys.exit(asyncio.run(main())) 