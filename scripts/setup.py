#!/usr/bin/env python3
"""
Setup script for Bayesian Crypto Trading Bot.
Initializes database, creates necessary directories, and validates configuration.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/models',
        'data/backups',
        'logs',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_database():
    """Create SQLite database with required tables."""
    db_path = "data/trading_bot.db"
    
    # Database schema from PRD
    schema_sql = """
    -- Price data table
    CREATE TABLE IF NOT EXISTS btc_prices (
        timestamp INTEGER PRIMARY KEY,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    -- Predictions table
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        predicted_price REAL NOT NULL,
        confidence_lower REAL NOT NULL,
        confidence_upper REAL NOT NULL,
        model_version TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    -- Trades table
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_timestamp INTEGER NOT NULL,
        exit_timestamp INTEGER,
        entry_price REAL NOT NULL,
        exit_price REAL,
        position_size REAL NOT NULL,
        position_type TEXT NOT NULL, -- 'long' or 'short'
        confidence_score REAL NOT NULL,
        stop_loss REAL,
        take_profit REAL,
        status TEXT NOT NULL, -- 'open', 'closed', 'stopped'
        pnl REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    -- Model performance table
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        mse REAL NOT NULL,
        mae REAL NOT NULL,
        directional_accuracy REAL NOT NULL,
        sharpe_ratio REAL,
        model_version TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for better query performance
    CREATE INDEX IF NOT EXISTS idx_btc_prices_timestamp ON btc_prices(timestamp);
    CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_trades_entry_timestamp ON trades(entry_timestamp);
    CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
    CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp);
    """
    
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()
        print(f"✓ Created database: {db_path}")
        print("✓ Database schema initialized successfully")
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False
    
    return True

def validate_environment():
    """Validate environment setup."""
    print("\n=== Environment Validation ===")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"✗ Python version {python_version.major}.{python_version.minor} is too old. Requires 3.8+")
        return False
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✓ Environment file (.env) exists")
    else:
        print("⚠ Environment file (.env) not found. Please copy .env.example to .env and configure")
    
    # Check required directories
    required_dirs = ['src', 'memory_bank', 'config', 'tests']
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Missing directory: {directory}")
            return False
    
    return True

def main():
    """Main setup function."""
    print("=== Bayesian Crypto Trading Bot Setup ===")
    print("Initializing project structure and database...\n")
    
    try:
        # Create directories
        print("Creating project directories...")
        create_directories()
        
        # Create database
        print("\nInitializing database...")
        if not create_database():
            print("✗ Database setup failed")
            return 1
        
        # Validate environment
        if not validate_environment():
            print("✗ Environment validation failed")
            return 1
        
        print("\n=== Setup Complete ===")
        print("✓ Project setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your settings")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Configure your Telegram bot token and API keys")
        print("4. Run the bot: python src/main.py")
        
        return 0
        
    except Exception as e:
        print(f"✗ Setup failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 