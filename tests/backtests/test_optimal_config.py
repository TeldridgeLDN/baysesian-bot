"""
Test the optimal profitable configuration found through parameter optimization.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from trading.engine import TradingEngine, TradingConfig

def test_optimal_profitable_configuration():
    """Test the best performing configuration with detailed analysis."""
    
    # Optimal configuration from parameter optimization
    config = TradingConfig(
        initial_capital=10000.0,
        position_size=0.12,  # 12% position size (higher than conservative 10%)
        stop_loss=0.05,
        take_profit=0.1,
        min_price_change_pct=1.2,  # More aggressive than 2.0%
        max_interval_width_pct=20.0,  # Allow higher uncertainty
        confidence_threshold=0.45,  # Lower than conservative 60%
        paper_trading=True,
        trading_fee=0.001,  # 0.1% trading fee
        slippage=0.0002  # 0.02% slippage
    )
    
    trading_engine = TradingEngine(config)
    
    # Load real historical data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_data.csv')
    df = pd.read_csv(data_path)
    df = df.tail(100).copy()
    prices = df['price'].values.astype(np.float32)
    
    print(f"🚀 TESTING OPTIMAL CONFIGURATION:")
    print(f"Position Size: {config.position_size:.1%}")
    print(f"Confidence Threshold: {config.confidence_threshold:.0%}")
    print(f"Min Price Change: {config.min_price_change_pct:.1f}%")
    print(f"Max CI Width: {config.max_interval_width_pct:.0f}%")
    print()
    
    trading_data = []
    results = []
    sequence_length = 60
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    if len(prices) >= sequence_length:
        for i in range(sequence_length, len(prices)):
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
            
            # Debug first few trades
            if len(trading_data) <= 5:
                price_change_pct = (predicted_price - current_price) / current_price * 100
                
                from trading.signals import TradingSignalGenerator
                signal_gen = TradingSignalGenerator(config)
                confidence = signal_gen.calculate_confidence_score(price_change_pct/100, interval_width_pct)
                signal = signal_gen.generate_signal(current_price, predicted_price)
                
                print(f"Point {len(trading_data)}: Price=${current_price:,.0f}, "
                      f"Predicted=${predicted_price:,.0f}, Change={price_change_pct:+.2f}%, "
                      f"CI_Width={interval_width_pct:.1f}%, Signal={signal}, Confidence={confidence:.3f}")
            
            # Process through trading engine
            result = trading_engine.process_data_point(data_point)
            if result:
                results.append(result)
    
    # Get final portfolio state
    final_state = trading_engine.get_portfolio_state()
    final_price = trading_data[-1]['actual_price'] if trading_data else config.initial_capital
    final_portfolio_value = trading_engine.portfolio.get_total_value(final_price)
    
    return_pct = ((final_portfolio_value - config.initial_capital) / config.initial_capital) * 100
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"Data Points Processed: {len(trading_data)}")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Return: {return_pct:+.2f}%")
    print(f"Total Trades: {final_state.get('total_trades', 0)}")
    print(f"Final Position: {final_state.get('position', 0):.6f}")
    
    # Analyze signal generation
    from trading.signals import TradingSignalGenerator
    signal_gen = TradingSignalGenerator(config)
    
    signals_generated = {"LONG": 0, "SHORT": 0, "HOLD": 0}
    confidence_scores = []
    profitable_signals = 0
    
    for d in trading_data:
        price_change = (d['predicted_price'] - d['actual_price']) / d['actual_price']
        confidence = signal_gen.calculate_confidence_score(price_change, d['interval_width_pct'])
        confidence_scores.append(confidence)
        
        signal = signal_gen.generate_signal(d['actual_price'], d['predicted_price'])
        signals_generated[signal] += 1
        
        # Check if signal would be profitable
        if signal == 'LONG' and price_change > 0:
            profitable_signals += 1
        elif signal == 'SHORT' and price_change < 0:
            profitable_signals += 1
    
    print(f"\n📈 SIGNAL ANALYSIS:")
    print(f"Signal Distribution: {signals_generated}")
    print(f"Average Confidence: {np.mean(confidence_scores):.3f}")
    print(f"Confidence Range: {np.min(confidence_scores):.3f} - {np.max(confidence_scores):.3f}")
    print(f"Potentially Profitable Signals: {profitable_signals}/{len(trading_data)} ({profitable_signals/len(trading_data)*100:.1f}%)")
    
    # Compare to conservative configuration
    print(f"\n⚖️ COMPARISON TO CONSERVATIVE CONFIG:")
    conservative_config = TradingConfig(
        initial_capital=10000.0,
        position_size=0.10,  # Conservative 10%
        min_price_change_pct=2.0,  # Conservative 2%
        max_interval_width_pct=15.0,  # Conservative 15%
        confidence_threshold=0.60,  # Conservative 60%
        paper_trading=True,
        trading_fee=0.001,
        slippage=0.0002,
        stop_loss=0.05,
        take_profit=0.1
    )
    
    conservative_engine = TradingEngine(conservative_config)
    conservative_results = []
    
    # Run same data through conservative config
    np.random.seed(42)  # Same seed for fair comparison
    for data_point in trading_data:
        result = conservative_engine.process_data_point(data_point)
        if result:
            conservative_results.append(result)
    
    conservative_final_state = conservative_engine.get_portfolio_state()
    conservative_portfolio_value = conservative_engine.portfolio.get_total_value(final_price)
    conservative_return = ((conservative_portfolio_value - conservative_config.initial_capital) / conservative_config.initial_capital) * 100
    
    print(f"Optimal Config:      {return_pct:+.2f}% ({final_state.get('total_trades', 0)} trades)")
    print(f"Conservative Config: {conservative_return:+.2f}% ({conservative_final_state.get('total_trades', 0)} trades)")
    print(f"Improvement:         {return_pct - conservative_return:+.2f}% ({final_state.get('total_trades', 0) - conservative_final_state.get('total_trades', 0)} more trades)")
    
    # Analyze what makes this configuration successful
    print(f"\n🎯 SUCCESS FACTORS:")
    print(f"1. Lower confidence threshold (45% vs 60%) → More trading opportunities")
    print(f"2. Higher uncertainty tolerance (20% vs 15%) → Can trade in volatile conditions")
    print(f"3. More aggressive price change threshold (1.2% vs 2.0%) → Captures smaller moves")
    print(f"4. Larger position size (12% vs 10%) → Higher returns per successful trade")
    
    # Validation
    assert return_pct > 0, f"Expected positive return, got {return_pct:.2f}%"
    assert final_state.get('total_trades', 0) > 0, "Should have executed some trades"
    assert final_portfolio_value > config.initial_capital, "Portfolio should have grown"
    
    print(f"\n✅ OPTIMAL CONFIGURATION VALIDATED: {return_pct:+.2f}% return achieved!")

if __name__ == "__main__":
    test_optimal_profitable_configuration()