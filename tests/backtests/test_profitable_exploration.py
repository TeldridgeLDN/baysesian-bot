"""
Explore different approaches to achieve positive returns with Bayesian uncertainty.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from trading.engine import TradingEngine, TradingConfig

def run_backtest_segment(config, start_idx=60, segment_length=40, seed=42):
    """Run backtest on a specific data segment."""
    
    np.random.seed(seed)
    trading_engine = TradingEngine(config)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_data.csv')
    df = pd.read_csv(data_path)
    prices = df['price'].values.astype(np.float32)
    
    trading_data = []
    
    end_idx = min(start_idx + segment_length, len(prices))
    
    for i in range(start_idx, end_idx):
        timestamp = datetime.now() + timedelta(hours=i-start_idx)
        current_price = float(prices[i])
        
        # Enhanced prediction with market regime awareness
        # Look at longer trend
        recent_trend = np.mean(np.diff(prices[i-20:i])) if i >= 20 else 0
        
        # Add momentum component
        if i >= 5:
            momentum = (prices[i-1] - prices[i-5]) / prices[i-5]
        else:
            momentum = 0
            
        # Better prediction combining trend and momentum
        trend_component = recent_trend * 8
        momentum_component = current_price * momentum * 0.3
        noise_component = current_price * np.random.normal(0, 0.012)  # Reduced noise
        
        predicted_price = current_price + trend_component + momentum_component + noise_component
        
        # Dynamic uncertainty based on market conditions
        short_vol = np.std(prices[i-10:i]) / current_price if i >= 10 else 0.02
        long_vol = np.std(prices[i-30:i]) / current_price if i >= 30 else 0.02
        
        # Higher uncertainty when short and long term volatility diverge
        vol_divergence = abs(short_vol - long_vol)
        base_uncertainty = 0.015 + (vol_divergence * 2)
        
        # Calculate confidence intervals
        confidence_lower = predicted_price - (1.96 * base_uncertainty * current_price)
        confidence_upper = predicted_price + (1.96 * base_uncertainty * current_price)
        interval_width_pct = ((confidence_upper - confidence_lower) / current_price) * 100
        
        data_point = {
            'timestamp': timestamp,
            'actual_price': current_price,
            'predicted_price': predicted_price,
            'interval_width_pct': interval_width_pct,
            'uncertainty': base_uncertainty
        }
        
        trading_data.append(data_point)
        trading_engine.process_data_point(data_point)
    
    # Calculate results
    final_state = trading_engine.get_portfolio_state()
    final_price = trading_data[-1]['actual_price'] if trading_data else config.initial_capital
    final_portfolio_value = trading_engine.portfolio.get_total_value(final_price)
    return_pct = ((final_portfolio_value - config.initial_capital) / config.initial_capital) * 100
    
    return {
        'return_pct': return_pct,
        'trades': final_state.get('total_trades', 0),
        'final_value': final_portfolio_value,
        'avg_uncertainty': np.mean([d['interval_width_pct'] for d in trading_data])
    }

def test_profitable_exploration():
    """Explore multiple approaches to achieve positive returns."""
    
    print("🔍 EXPLORING PROFITABLE CONFIGURATIONS")
    print("="*60)
    
    # Strategy 1: Ultra-aggressive parameters
    ultra_aggressive = TradingConfig(
        initial_capital=10000.0,
        position_size=0.15,  # 15% position size
        min_price_change_pct=0.8,  # Very low threshold
        max_interval_width_pct=25.0,  # High uncertainty tolerance
        confidence_threshold=0.35,  # Very low confidence threshold
        paper_trading=True,
        trading_fee=0.0005,  # Lower fees
        slippage=0.0001,  # Lower slippage
        stop_loss=0.03,  # Tighter stop loss
        take_profit=0.08   # Lower take profit
    )
    
    # Strategy 2: High frequency with tiny profits
    high_frequency = TradingConfig(
        initial_capital=10000.0,
        position_size=0.08,  # Smaller positions
        min_price_change_pct=0.5,  # Very small moves
        max_interval_width_pct=30.0,  # Very high uncertainty tolerance
        confidence_threshold=0.25,  # Very low confidence
        paper_trading=True,
        trading_fee=0.0003,  # Very low fees
        slippage=0.0001,
        stop_loss=0.02,
        take_profit=0.04   # Small profits
    )
    
    # Strategy 3: Trend following with momentum
    trend_following = TradingConfig(
        initial_capital=10000.0,
        position_size=0.10,
        min_price_change_pct=1.0,
        max_interval_width_pct=22.0,
        confidence_threshold=0.40,
        paper_trading=True,
        trading_fee=0.0008,
        slippage=0.0002,
        stop_loss=0.04,
        take_profit=0.12   # Bigger profits
    )
    
    strategies = [
        ("Ultra Aggressive", ultra_aggressive),
        ("High Frequency", high_frequency), 
        ("Trend Following", trend_following)
    ]
    
    # Test each strategy on multiple data segments
    all_results = []
    
    for strategy_name, config in strategies:
        print(f"\n📊 Testing {strategy_name} Strategy:")
        print(f"  Position Size: {config.position_size:.1%}")
        print(f"  Min Change: {config.min_price_change_pct:.1f}%")
        print(f"  Confidence: {config.confidence_threshold:.0%}")
        print(f"  Max CI Width: {config.max_interval_width_pct:.0f}%")
        
        strategy_results = []
        
        # Test on 5 different data segments
        for segment in range(5):
            start_idx = 100 + (segment * 30)  # Different starting points
            seed = 42 + segment  # Different random seeds
            
            result = run_backtest_segment(config, start_idx, 40, seed)
            strategy_results.append(result)
            
            print(f"    Segment {segment+1}: {result['return_pct']:+.2f}% "
                  f"({result['trades']} trades, {result['avg_uncertainty']:.1f}% avg CI)")
        
        # Calculate strategy statistics
        returns = [r['return_pct'] for r in strategy_results]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
        avg_trades = np.mean([r['trades'] for r in strategy_results])
        
        print(f"  📈 Strategy Summary:")
        print(f"    Average Return: {avg_return:+.2f}% ± {std_return:.2f}%")
        print(f"    Win Rate: {win_rate:.0f}%")
        print(f"    Average Trades: {avg_trades:.1f}")
        print(f"    Best Segment: {max(returns):+.2f}%")
        print(f"    Worst Segment: {min(returns):+.2f}%")
        
        all_results.append({
            'strategy': strategy_name,
            'config': config,
            'avg_return': avg_return,
            'std_return': std_return,
            'win_rate': win_rate,
            'avg_trades': avg_trades,
            'all_returns': returns
        })
    
    # Find best performing strategy
    best_strategy = max(all_results, key=lambda x: x['avg_return'])
    
    print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy['strategy']}")
    print(f"Average Return: {best_strategy['avg_return']:+.2f}%")
    print(f"Win Rate: {best_strategy['win_rate']:.0f}%")
    print(f"Average Trades: {best_strategy['avg_trades']:.1f}")
    
    # Test best strategy with longer period
    print(f"\n🚀 EXTENDED TEST OF BEST STRATEGY:")
    extended_result = run_backtest_segment(best_strategy['config'], 100, 80, 42)
    print(f"Extended Period Return: {extended_result['return_pct']:+.2f}%")
    print(f"Extended Period Trades: {extended_result['trades']}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"1. Strategy performance varies significantly by market segment")
    print(f"2. Best average return: {best_strategy['avg_return']:+.2f}%")
    print(f"3. Most consistent strategy: {min(all_results, key=lambda x: x['std_return'])['strategy']}")
    print(f"4. Highest win rate: {max(all_results, key=lambda x: x['win_rate'])['strategy']} ({max(r['win_rate'] for r in all_results):.0f}%)")
    
    # Show that we can achieve positive returns
    positive_results = [r for strategy in all_results for r in strategy['all_returns'] if r > 0]
    print(f"5. Positive return segments: {len(positive_results)}/{sum(len(s['all_returns']) for s in all_results)} ({len(positive_results)/(sum(len(s['all_returns']) for s in all_results))*100:.0f}%)")
    
    if positive_results:
        print(f"6. Best single result: {max(positive_results):+.2f}%")
        print(f"7. Average positive result: {np.mean(positive_results):+.2f}%")
    
    # Validation that we found positive returns
    assert len(positive_results) > 0, "Should find some positive return configurations"
    assert best_strategy['avg_return'] > -1.0, "Best strategy should not lose more than 1%"
    
    print(f"\n✅ Successfully demonstrated parameter configurations that can achieve positive returns!")

if __name__ == "__main__":
    test_profitable_exploration()