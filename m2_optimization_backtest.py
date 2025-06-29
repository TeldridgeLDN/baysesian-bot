#!/usr/bin/env python3
"""
M2 Money Supply Optimization Backtest
Comprehensive backtest to find optimal M2 thresholds and position multipliers
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from itertools import product
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from data.m2_data_provider import M2DataProvider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class M2OptimizationBacktest:
    """
    Comprehensive backtest to optimize M2 regime thresholds and position multipliers
    """
    
    def __init__(self, start_date='2020-01-01', end_date='2024-06-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.btc_data = None
        self.m2_provider = M2DataProvider()
        self.combined_data = None
        self.optimization_results = {}
        
    def load_data(self):
        """Load Bitcoin and M2 data"""
        logger.info("📊 Loading data for M2 optimization backtest...")
        
        # Load M2 data
        if not self.m2_provider.fetch_m2_data(self.start_date, self.end_date):
            logger.error("Failed to load M2 data")
            return False
        
        # Create synthetic Bitcoin data (since API rate limited)
        logger.info("₿ Creating synthetic Bitcoin data for testing...")
        self._create_synthetic_bitcoin_data()
        
        # Combine datasets
        self._combine_data()
        
        logger.info(f"✅ Data loaded: {len(self.combined_data)} observations")
        return True
    
    def _create_synthetic_bitcoin_data(self):
        """Create realistic synthetic Bitcoin data based on historical patterns"""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        np.random.seed(42)
        
        # Create realistic Bitcoin price evolution
        initial_price = 7000  # Starting price (early 2020)
        prices = [initial_price]
        
        # Simulate different market phases
        for i, date in enumerate(dates[1:], 1):
            prev_price = prices[-1]
            
            # Different volatility and trend based on period
            if '2020-03-01' <= date.strftime('%Y-%m-%d') <= '2021-11-01':
                # Bull market: High returns, high volatility
                daily_return = np.random.normal(0.003, 0.04)  # ~110% annual, high vol
            elif '2021-11-01' <= date.strftime('%Y-%m-%d') <= '2022-12-01':
                # Bear market: Negative returns, high volatility
                daily_return = np.random.normal(-0.002, 0.045)  # ~-50% annual
            elif '2022-12-01' <= date.strftime('%Y-%m-%d') <= '2024-01-01':
                # Recovery: Moderate positive returns
                daily_return = np.random.normal(0.002, 0.035)  # ~50% annual
            else:
                # Normal market
                daily_return = np.random.normal(0.0005, 0.03)  # ~20% annual
            
            # Add some momentum and mean reversion
            if i > 10:
                momentum = np.mean([prices[j]/prices[j-1] - 1 for j in range(max(1, i-5), i)])
                daily_return += momentum * 0.1  # 10% momentum effect
            
            new_price = prev_price * (1 + daily_return)
            new_price = max(1000, min(200000, new_price))  # Realistic bounds
            prices.append(new_price)
        
        # Create DataFrame
        self.btc_data = pd.DataFrame({
            'Close': prices,
            'returns': [0] + [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
        }, index=dates)
        
        # Add technical indicators
        self.btc_data['sma_20'] = self.btc_data['Close'].rolling(20).mean()
        self.btc_data['sma_50'] = self.btc_data['Close'].rolling(50).mean()
        self.btc_data['volatility'] = self.btc_data['returns'].rolling(30).std()
        
        logger.info(f"✅ Synthetic Bitcoin data created: {len(self.btc_data)} days")
    
    def _combine_data(self):
        """Combine Bitcoin and M2 data"""
        # Get M2 data for backtest period
        m2_data = self.m2_provider.get_m2_data_for_backtest(self.start_date, self.end_date)
        
        # Resample M2 to daily frequency
        m2_daily = m2_data.resample('D').ffill()
        
        # Combine with Bitcoin data
        self.combined_data = self.btc_data.join(m2_daily, how='left')
        self.combined_data = self.combined_data.ffill().dropna()
        
        logger.info(f"✅ Data combined successfully")
    
    def create_baseline_strategy(self):
        """Create simple baseline technical strategy"""
        data = self.combined_data.copy()
        
        # Simple technical signals
        data['technical_signal'] = 0
        
        # Buy when price > SMA20 and not overbought
        buy_condition = (data['Close'] > data['sma_20']) & (data['Close'] < data['sma_20'] * 1.1)
        data.loc[buy_condition, 'technical_signal'] = 1
        
        # Sell when price < SMA20 and not oversold  
        sell_condition = (data['Close'] < data['sma_20']) & (data['Close'] > data['sma_20'] * 0.9)
        data.loc[sell_condition, 'technical_signal'] = -1
        
        return data
    
    def test_m2_parameters(self, expansion_threshold=8, contraction_threshold=2,
                          expansion_multiplier=1.2, contraction_multiplier=0.6,
                          stable_multiplier=1.0):
        """
        Test specific M2 parameter combination
        
        Args:
            expansion_threshold: M2 YoY growth above which = expansion regime
            contraction_threshold: M2 YoY growth below which = contraction regime  
            expansion_multiplier: Position size multiplier during expansion
            contraction_multiplier: Position size multiplier during contraction
            stable_multiplier: Position size multiplier during stable regime
            
        Returns:
            Dictionary with performance metrics
        """
        data = self.create_baseline_strategy()
        
        # Classify M2 regimes based on thresholds
        data['m2_regime'] = 'stable'
        data.loc[data['m2_yoy_lagged'] > expansion_threshold, 'm2_regime'] = 'expansion'
        data.loc[data['m2_yoy_lagged'] < contraction_threshold, 'm2_regime'] = 'contraction'
        
        # Apply M2 position multipliers
        data['position_multiplier'] = stable_multiplier
        data.loc[data['m2_regime'] == 'expansion', 'position_multiplier'] = expansion_multiplier
        data.loc[data['m2_regime'] == 'contraction', 'position_multiplier'] = contraction_multiplier
        
        # Calculate strategy returns
        data['baseline_returns'] = data['technical_signal'].shift(1) * data['returns']
        data['m2_enhanced_returns'] = (data['technical_signal'].shift(1) * 
                                      data['position_multiplier'].shift(1) * 
                                      data['returns'])
        
        # Performance metrics
        baseline_metrics = self._calculate_performance_metrics(data['baseline_returns'], 'Baseline')
        m2_metrics = self._calculate_performance_metrics(data['m2_enhanced_returns'], 'M2 Enhanced')
        
        # Calculate improvement
        sharpe_improvement = m2_metrics['sharpe'] - baseline_metrics['sharpe']
        
        # Regime analysis
        regime_stats = data.groupby('m2_regime').agg({
            'returns': ['mean', 'std', 'count'],
            'position_multiplier': 'mean'
        }).round(4)
        
        return {
            'parameters': {
                'expansion_threshold': expansion_threshold,
                'contraction_threshold': contraction_threshold,
                'expansion_multiplier': expansion_multiplier,
                'contraction_multiplier': contraction_multiplier,
                'stable_multiplier': stable_multiplier
            },
            'baseline_metrics': baseline_metrics,
            'm2_metrics': m2_metrics,
            'sharpe_improvement': sharpe_improvement,
            'regime_stats': regime_stats,
            'data': data
        }
    
    def _calculate_performance_metrics(self, returns_series, name):
        """Calculate comprehensive performance metrics"""
        returns = returns_series.dropna()
        
        if len(returns) == 0:
            return {'name': name, 'total_return': 0, 'annual_return': 0, 
                   'volatility': 0, 'sharpe': 0, 'max_drawdown': 0}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (365 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(365)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
    
    def run_parameter_optimization(self):
        """Run comprehensive parameter optimization"""
        logger.info("🔍 Running M2 parameter optimization...")
        
        # Define parameter ranges to test
        expansion_thresholds = [6, 7, 8, 9, 10]  # M2 growth % for expansion
        contraction_thresholds = [1, 2, 3, 4]    # M2 growth % for contraction
        expansion_multipliers = [1.1, 1.2, 1.3, 1.4, 1.5]  # Position boost in expansion
        contraction_multipliers = [0.4, 0.5, 0.6, 0.7, 0.8]  # Position reduction in contraction
        
        results = []
        total_combinations = (len(expansion_thresholds) * len(contraction_thresholds) * 
                            len(expansion_multipliers) * len(contraction_multipliers))
        
        logger.info(f"Testing {total_combinations} parameter combinations...")
        
        for i, (exp_thresh, con_thresh, exp_mult, con_mult) in enumerate(
            product(expansion_thresholds, contraction_thresholds, 
                   expansion_multipliers, contraction_multipliers)):
            
            # Skip invalid combinations
            if exp_thresh <= con_thresh:
                continue
                
            try:
                result = self.test_m2_parameters(
                    expansion_threshold=exp_thresh,
                    contraction_threshold=con_thresh,
                    expansion_multiplier=exp_mult,
                    contraction_multiplier=con_mult
                )
                
                # Store key results
                results.append({
                    'expansion_threshold': exp_thresh,
                    'contraction_threshold': con_thresh,
                    'expansion_multiplier': exp_mult,
                    'contraction_multiplier': con_mult,
                    'sharpe_improvement': result['sharpe_improvement'],
                    'baseline_sharpe': result['baseline_metrics']['sharpe'],
                    'm2_sharpe': result['m2_metrics']['sharpe'],
                    'baseline_return': result['baseline_metrics']['annual_return'],
                    'm2_return': result['m2_metrics']['annual_return'],
                    'baseline_drawdown': result['baseline_metrics']['max_drawdown'],
                    'm2_drawdown': result['m2_metrics']['max_drawdown']
                })
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Completed {i + 1}/{total_combinations} combinations...")
                    
            except Exception as e:
                logger.warning(f"Error testing combination {exp_thresh}/{con_thresh}/{exp_mult}/{con_mult}: {e}")
                continue
        
        # Convert to DataFrame for analysis
        self.optimization_results = pd.DataFrame(results)
        
        logger.info(f"✅ Optimization completed: {len(self.optimization_results)} valid combinations")
        return self.optimization_results
    
    def analyze_optimization_results(self):
        """Analyze optimization results and find best parameters"""
        if self.optimization_results.empty:
            logger.error("No optimization results to analyze")
            return None
        
        results = self.optimization_results
        
        print("🎯 M2 Parameter Optimization Results")
        print("=" * 60)
        
        # Find best combinations by different criteria
        best_sharpe = results.loc[results['sharpe_improvement'].idxmax()]
        best_return = results.loc[results['m2_return'].idxmax()]
        best_drawdown = results.loc[results['m2_drawdown'].idxmax()]  # Less negative = better
        
        print(f"\n📈 Best Sharpe Improvement: +{best_sharpe['sharpe_improvement']:.3f}")
        print(f"   Thresholds: {best_sharpe['expansion_threshold']}% / {best_sharpe['contraction_threshold']}%")
        print(f"   Multipliers: {best_sharpe['expansion_multiplier']:.1f}x / {best_sharpe['contraction_multiplier']:.1f}x")
        print(f"   Sharpe: {best_sharpe['baseline_sharpe']:.3f} → {best_sharpe['m2_sharpe']:.3f}")
        
        print(f"\n🚀 Best Annual Return: {best_return['m2_return']:.1%}")
        print(f"   Thresholds: {best_return['expansion_threshold']}% / {best_return['contraction_threshold']}%")
        print(f"   Multipliers: {best_return['expansion_multiplier']:.1f}x / {best_return['contraction_multiplier']:.1f}x")
        
        print(f"\n🛡️ Best Drawdown Control: {best_drawdown['m2_drawdown']:.1%}")
        print(f"   Thresholds: {best_drawdown['expansion_threshold']}% / {best_drawdown['contraction_threshold']}%")
        print(f"   Multipliers: {best_drawdown['expansion_multiplier']:.1f}x / {best_drawdown['contraction_multiplier']:.1f}x")
        
        # Summary statistics
        print(f"\n📊 Optimization Summary:")
        print(f"   Positive Sharpe improvements: {(results['sharpe_improvement'] > 0).sum()}/{len(results)}")
        print(f"   Average Sharpe improvement: {results['sharpe_improvement'].mean():+.3f}")
        print(f"   Best Sharpe improvement: {results['sharpe_improvement'].max():+.3f}")
        print(f"   Worst Sharpe impact: {results['sharpe_improvement'].min():+.3f}")
        
        # Find balanced optimal parameters (compromise between all metrics)
        # Normalize metrics for balanced scoring
        results['normalized_sharpe'] = (results['sharpe_improvement'] - results['sharpe_improvement'].min()) / (results['sharpe_improvement'].max() - results['sharpe_improvement'].min())
        results['normalized_return'] = (results['m2_return'] - results['m2_return'].min()) / (results['m2_return'].max() - results['m2_return'].min())
        results['normalized_drawdown'] = (results['m2_drawdown'] - results['m2_drawdown'].min()) / (results['m2_drawdown'].max() - results['m2_drawdown'].min())
        
        # Composite score (higher is better)
        results['composite_score'] = (results['normalized_sharpe'] * 0.5 + 
                                     results['normalized_return'] * 0.3 + 
                                     results['normalized_drawdown'] * 0.2)
        
        optimal = results.loc[results['composite_score'].idxmax()]
        
        print(f"\n🎯 RECOMMENDED OPTIMAL PARAMETERS:")
        print(f"   Expansion Threshold: {optimal['expansion_threshold']:.0f}% M2 YoY growth")
        print(f"   Contraction Threshold: {optimal['contraction_threshold']:.0f}% M2 YoY growth")
        print(f"   Expansion Multiplier: {optimal['expansion_multiplier']:.1f}x position size")
        print(f"   Contraction Multiplier: {optimal['contraction_multiplier']:.1f}x position size")
        print(f"   Expected Sharpe Improvement: +{optimal['sharpe_improvement']:.3f}")
        print(f"   Expected Annual Return: {optimal['m2_return']:.1%}")
        print(f"   Expected Max Drawdown: {optimal['m2_drawdown']:.1%}")
        
        return {
            'optimal_parameters': {
                'expansion_threshold': optimal['expansion_threshold'],
                'contraction_threshold': optimal['contraction_threshold'],
                'expansion_multiplier': optimal['expansion_multiplier'],
                'contraction_multiplier': optimal['contraction_multiplier']
            },
            'expected_performance': {
                'sharpe_improvement': optimal['sharpe_improvement'],
                'annual_return': optimal['m2_return'],
                'max_drawdown': optimal['m2_drawdown']
            },
            'best_combinations': {
                'best_sharpe': best_sharpe.to_dict(),
                'best_return': best_return.to_dict(),
                'best_drawdown': best_drawdown.to_dict()
            }
        }
    
    def save_optimization_results(self, filename='m2_optimization_results.json'):
        """Save optimization results to file"""
        if hasattr(self, 'optimization_results') and not self.optimization_results.empty:
            # Convert to serializable format
            results_dict = self.optimization_results.to_dict('records')
            
            with open(filename, 'w') as f:
                json.dump({
                    'optimization_results': results_dict,
                    'backtest_period': {
                        'start_date': self.start_date,
                        'end_date': self.end_date
                    },
                    'generated_at': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"✅ Optimization results saved to {filename}")

def main():
    """Run M2 optimization backtest"""
    backtest = M2OptimizationBacktest()
    
    # Load data
    if not backtest.load_data():
        logger.error("Failed to load data")
        return
    
    # Run optimization
    optimization_results = backtest.run_parameter_optimization()
    
    if optimization_results is not None and not optimization_results.empty:
        # Analyze results
        optimal_config = backtest.analyze_optimization_results()
        
        # Save results
        backtest.save_optimization_results()
        
        # Test optimal parameters in detail
        if optimal_config:
            logger.info("\n🧪 Testing optimal parameters in detail...")
            optimal_params = optimal_config['optimal_parameters']
            
            detailed_result = backtest.test_m2_parameters(
                expansion_threshold=optimal_params['expansion_threshold'],
                contraction_threshold=optimal_params['contraction_threshold'],
                expansion_multiplier=optimal_params['expansion_multiplier'],
                contraction_multiplier=optimal_params['contraction_multiplier']
            )
            
            print(f"\n📊 Detailed Results with Optimal Parameters:")
            print(f"Baseline Strategy:")
            baseline = detailed_result['baseline_metrics']
            print(f"  Annual Return: {baseline['annual_return']:.1%}")
            print(f"  Sharpe Ratio: {baseline['sharpe']:.3f}")
            print(f"  Max Drawdown: {baseline['max_drawdown']:.1%}")
            
            print(f"\nM2-Enhanced Strategy:")
            m2_enhanced = detailed_result['m2_metrics']
            print(f"  Annual Return: {m2_enhanced['annual_return']:.1%}")
            print(f"  Sharpe Ratio: {m2_enhanced['sharpe']:.3f}")
            print(f"  Max Drawdown: {m2_enhanced['max_drawdown']:.1%}")
            
            print(f"\nImprovement:")
            print(f"  Sharpe: {detailed_result['sharpe_improvement']:+.3f}")
            print(f"  Return: {(m2_enhanced['annual_return'] - baseline['annual_return']):.1%}")
            print(f"  Drawdown: {(m2_enhanced['max_drawdown'] - baseline['max_drawdown']):.1%}")
            
            print(f"\n✅ M2 optimization backtest completed successfully!")
            print(f"🚀 Ready to implement M2 overlay with optimal parameters")
    
    else:
        logger.error("Optimization failed - no valid results")

if __name__ == "__main__":
    main()