import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_btc_data(file_path: str, n_points: int = 1000):
    """
    Generates a realistic-looking sample BTC data file for backtesting.
    """
    start_price = 60000
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_points)]
    timestamps.reverse()

    prices = []
    current_price = start_price
    volatility = 0.02

    for _ in range(n_points):
        drift = np.random.uniform(-0.005, 0.005)
        current_price *= (1 + np.random.normal(drift, volatility))
        prices.append(current_price)

    # Simple prediction: assume next price is current price + some noise
    predicted_prices = [p * (1 + np.random.normal(0, volatility * 0.5)) for p in prices]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'predicted_price': predicted_prices
    })

    df.to_csv(file_path, index=False)
    print(f"Sample data created at {file_path}")

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    create_sample_btc_data('data/btc_data.csv') 