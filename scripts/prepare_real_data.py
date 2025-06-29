import sys
import os
import asyncio
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.collectors import PriceDataCollector
from utils.config import load_config
from data.storage import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """
    Fetches real BTC data, generates simple predictions, and saves it
    in the format required for the backtester.
    """
    logging.info("Initializing configuration and database manager...")
    config = load_config()
    db_manager = DatabaseManager(config.database.db_path)
    
    # 1. Fetch real historical data
    logging.info("Fetching real historical data for BTC (last 30 days)...")
    collector = PriceDataCollector(config, db_manager)
    api_response = await collector.collect_historical_data(days=30)
    
    if not api_response.success:
        logging.error("Failed to fetch historical data. Aborting.")
        return

    fetched_data = api_response.data
    if not fetched_data:
        logging.error("No data was fetched. Aborting.")
        return
        
    logging.info(f"Successfully fetched {len(fetched_data)} data points.")
    
    # 2. Convert to DataFrame and generate predictions
    df = pd.DataFrame(fetched_data)
    
    # Ensure data is sorted by timestamp before generating predictions
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Use the 'close' price as the main price for backtesting
    df['price'] = df['close']
    
    # Simple "perfect foresight" prediction model
    df['predicted_price'] = df['price'].shift(-1)
    
    # Drop the last row as it will have no predicted price
    df.dropna(inplace=True)
    
    # 3. Save to CSV in the required format
    output_df = df[['timestamp', 'price', 'predicted_price']]
    
    output_path = 'data/btc_data.csv'
    logging.info(f"Saving prepared data to {output_path}...")
    output_df.to_csv(output_path, index=False)
    
    logging.info("Data preparation complete.")

if __name__ == "__main__":
    asyncio.run(main()) 