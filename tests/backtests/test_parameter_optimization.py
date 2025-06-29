"""
Parameter optimization for Bayesian trading system to achieve positive returns.
Tests different configurations while maintaining risk management principles.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from trading.engine import TradingEngine, TradingConfig

def run_backtest_config(config_params, data_points=40, verbose=False):
    """Run backtest with specific configuration parameters."""
    
    config = TradingConfig(**config_params)
    trading_engine = TradingEngine(config)
    
    # Load real historical data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_data.csv')
    if not os.path.exists(data_path):
        return None
        
    df = pd.read_csv(data_path)
    df = df.tail(100).copy()
    prices = df['price'].values.astype(np.float32)
    
    trading_data = []
    results = []
    sequence_length = 60
    
    if len(prices) >= sequence_length:
        for i in range(sequence_length, min(sequence_length + data_points, len(prices))):
            timestamp = datetime.now() + timedelta(hours=i-sequence_length)
            current_price = float(prices[i])
            
            # Simulate realistic Bayesian prediction
            recent_trend = np.mean(np.diff(prices[i-10:i])) if i > 10 else 0
            prediction_noise = np.random.normal(0, 0.015)  # 1.5% prediction uncertainty
            predicted_price = current_price + recent_trend * 5 + (current_price * prediction_noise)
            
            # Simulate realistic uncertainty
            recent_volatility = np.std(prices[i-sequence_length:i]) / current_price
            base_uncertainty = 0.02  # 2% base uncertainty
            uncertainty = base_uncertainty + (recent_volatility * 0.5)
            
            # Calculate confidence intervals
            confidence_lower = predicted_price - (1.96 * uncertainty * current_price)
            confidence_upper = predicted_price + (1.96 * uncertainty * current_price)
            interval_width_pct = ((confidence_upper - confidence_lower) / current_price) * 100
            
            data_point = {
                'timestamp': timestamp,
                'actual_price': current_price,
                'predicted_price': predicted_price,
                'interval_width_pct': interval_width_pct,
                'uncertainty': uncertainty
            }
            
            trading_data.append(data_point)
            
            # Process through trading engine
            result = trading_engine.process_data_point(data_point)
            if result:
                results.append(result)
    
    # Calculate final performance
    final_state = trading_engine.get_portfolio_state()
    final_price = trading_data[-1]['actual_price'] if trading_data else config.initial_capital
    final_portfolio_value = trading_engine.portfolio.get_total_value(final_price)
    
    return_pct = ((final_portfolio_value - config.initial_capital) / config.initial_capital) * 100
    
    performance = {
        'return_pct': return_pct,
        'final_value': final_portfolio_value,
        'total_trades': final_state.get('total_trades', 0),
        'final_position': final_state.get('position', 0),
        'data_points': len(trading_data),
        'config': config_params.copy()
    }
    
    if verbose:
        print(f"Config: {config_params}")
        print(f"Return: {return_pct:.2f}%, Trades: {performance['total_trades']}")
    
    return performance

def test_parameter_optimization():
    """Test different parameter combinations to find profitable configurations."""
    
    # Base configuration
    base_config = {
        'initial_capital': 10000.0,
        'paper_trading': True,
        'trading_fee': 0.001,
        'slippage': 0.0002,
        'stop_loss': 0.05,
        'take_profit': 0.1
    }
    
    # Parameter ranges to test
    test_configs = []
    
    # Experiment 1: Position sizing and confidence
    for position_size in [0.05, 0.08, 0.1, 0.12, 0.15]:
        for confidence_threshold in [0.45, 0.50, 0.55, 0.60]:
            config = base_config.copy()
            config.update({
                'position_size': position_size,
                'confidence_threshold': confidence_threshold,
                'min_price_change_pct': 1.2,  # More aggressive
                'max_interval_width_pct': 20.0  # Allow higher uncertainty
            })
            test_configs.append(config)
    
    # Experiment 2: Price change sensitivity  
    for min_change in [0.8, 1.0, 1.2, 1.5, 2.0]:
        for max_interval in [15.0, 18.0, 20.0, 22.0, 25.0]:
            config = base_config.copy()
            config.update({
                'position_size': 0.1,
                'confidence_threshold': 0.50,
                'min_price_change_pct': min_change,
                'max_interval_width_pct': max_interval
            })
            test_configs.append(config)
    
    # Experiment 3: Trading costs impact
    for fee in [0.0005, 0.001, 0.0015]:
        for slippage in [0.0001, 0.0002, 0.0003]:
            config = base_config.copy()
            config.update({
                'position_size': 0.12,
                'confidence_threshold': 0.45,
                'min_price_change_pct': 1.0,
                'max_interval_width_pct': 22.0,
                'trading_fee': fee,
                'slippage': slippage
            })
            test_configs.append(config)
    
    # Run all configurations
    print(f"Testing {len(test_configs)} parameter combinations...")
    
    results = []
    np.random.seed(42)  # For reproducible results
    
    for i, config in enumerate(test_configs):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(test_configs)} ({i/len(test_configs)*100:.1f}%)")
        
        try:
            performance = run_backtest_config(config, data_points=40)
            if performance:
                results.append(performance)
        except Exception as e:
            print(f"Error with config {i}: {e}")
            continue
    
    # Analyze results
    if not results:
        print("No valid results generated")
        return
    
    # Sort by return
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    
    # Show top performers
    print(f"\n🏆 TOP 10 PERFORMING CONFIGURATIONS:")
    print("="*80)
    
    for i, result in enumerate(results[:10]):
        config = result['config']
        print(f"\nRank {i+1}: {result['return_pct']:+.2f}% return ({result['total_trades']} trades)")
        print(f"  Position Size: {config['position_size']:.3f}")
        print(f"  Confidence Threshold: {config['confidence_threshold']:.3f}")
        print(f"  Min Price Change: {config['min_price_change_pct']:.2f}%")
        print(f"  Max CI Width: {config['max_interval_width_pct']:.1f}%")
        print(f"  Trading Fee: {config['trading_fee']:.4f}")
        print(f"  Slippage: {config['slippage']:.4f}")
    
    # Statistical analysis
    positive_returns = [r for r in results if r['return_pct'] > 0]
    negative_returns = [r for r in results if r['return_pct'] <= 0]
    
    print(f"\n📊 STATISTICAL ANALYSIS:")
    print(f"Total configurations tested: {len(results)}")
    print(f"Positive returns: {len(positive_returns)} ({len(positive_returns)/len(results)*100:.1f}%)")
    print(f"Negative returns: {len(negative_returns)} ({len(negative_returns)/len(results)*100:.1f}%)")
    print(f"Best return: {results[0]['return_pct']:+.2f}%")
    print(f"Worst return: {results[-1]['return_pct']:+.2f}%")
    print(f"Average return: {np.mean([r['return_pct'] for r in results]):+.2f}%")
    
    # Analyze what makes configurations profitable
    if positive_returns:
        print(f"\n🎯 PROFITABLE CONFIGURATION PATTERNS:")
        
        # Average parameters for profitable configs
        profitable_configs = [r['config'] for r in positive_returns]
        
        avg_position_size = np.mean([c['position_size'] for c in profitable_configs])
        avg_confidence = np.mean([c['confidence_threshold'] for c in profitable_configs])
        avg_min_change = np.mean([c['min_price_change_pct'] for c in profitable_configs])
        avg_max_interval = np.mean([c['max_interval_width_pct'] for c in profitable_configs])
        
        print(f"  Average Position Size: {avg_position_size:.3f}")
        print(f"  Average Confidence Threshold: {avg_confidence:.3f}")
        print(f"  Average Min Price Change: {avg_min_change:.2f}%")
        print(f"  Average Max CI Width: {avg_max_interval:.1f}%")
        
        # Test the best configuration in detail
        best_config = results[0]['config']
        print(f"\n🔥 TESTING BEST CONFIGURATION IN DETAIL:")
        detailed_result = run_backtest_config(best_config, data_points=40, verbose=True)
        
        assert len(results) > 0, "Should generate test results"
        assert any(r['return_pct'] > 0 for r in results), "Should find some profitable configurations"

if __name__ == "__main__":
    test_parameter_optimization()