import sys
import os
import logging
import pandas as pd
from typing import List, Dict, Any

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from trading.engine import TradingConfig
from trading.backtester import Backtester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_real_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads and prepares real BTC data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}. Please ensure the file exists.")
        return []

    # Basic data validation
    required_columns = ['timestamp', 'price', 'predicted_price']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"Data file must contain the following columns: {required_columns}")
        return []

    # Convert to dictionary format and add a placeholder for interval width
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['interval_width_pct'] = 2.0  # Placeholder since real data doesn't have this
    
    # Rename 'price' to 'actual_price' to match what the engine expects
    df.rename(columns={'price': 'actual_price'}, inplace=True)

    return df.to_dict('records')

def run_and_report(config: TradingConfig, market_data: List[Dict[str, Any]], market_name: str):
    """Runs a backtest with a given config and reports results."""
    if not market_data:
        logging.warning(f"Skipping backtest for {market_name} due to no data.")
        return

    logging.info(f"--- Running backtest for Stop-Loss: {config.stop_loss_type.upper()} in {market_name.upper()} market ---")
    
    backtester = Backtester(config)
    final_state = backtester.run(market_data)
    
    final_price = market_data[-1]['actual_price']
    final_portfolio_value = backtester.trading_engine.portfolio.get_total_value(final_price)
    initial_capital = config.initial_capital
    return_percentage = ((final_portfolio_value - initial_capital) / initial_capital) * 100
    
    final_position_value = abs(final_state['position'] * final_price)
    capital_at_risk = final_position_value if final_state['position'] != 0 else 0

    print("\n--- PERFORMANCE BREAKDOWN ---")
    print(f"Market: {market_name.upper()} | Strategy: {config.stop_loss_type.upper()}")
    print(f"Total Return: {return_percentage:.2f}%")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Trades: {final_state['total_trades']}")
    print(f"Win Rate: {(final_state['win_rate'] * 100):.2f}%")
    print(f"Max Drawdown: {(final_state['max_drawdown'] * 100):.2f}%")
    print(f"Capital in Final Position: ${capital_at_risk:,.2f}")
    print("---------------------------\n")

def main():
    """Main function to run the comparative backtests on real data."""
    real_market_data = load_real_data('data/btc_data.csv')
    
    base_params = {
        "position_size": 0.5, "min_price_change_pct": 1.0,
        "max_interval_width_pct": 5.0, "confidence_threshold": 0.5,
    }

    configs = {
        'none': TradingConfig(stop_loss_type='none', **base_params),
        'fixed': TradingConfig(stop_loss_type='fixed', stop_loss=0.05, **base_params),
        'trailing': TradingConfig(stop_loss_type='trailing', trailing_stop_pct=0.03, **base_params),
    }

    if real_market_data:
        print(f"\n\n{'='*20} TESTING REAL BTC DATA {'='*20}\n")
        for config in configs.values():
            run_and_report(config, real_market_data, "Real BTC Data")

if __name__ == "__main__":
    main() 