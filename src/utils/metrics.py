"""
Performance metrics calculation for the crypto trading bot.
Handles calculation of trading and model performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Calculates various performance metrics for trading and model performance."""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price series."""
        if len(prices) < 2:
            return []
        return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
    def calculate_sharpe_ratio(self, returns: List[float], 
                             risk_free_rate: float = None) -> float:
        """Calculate Sharpe ratio."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0
            
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                
        return max_dd
        
    def calculate_win_rate(self, trade_pnls: List[float]) -> float:
        """Calculate win rate from trade PnLs."""
        if not trade_pnls:
            return 0.0
            
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        return winning_trades / len(trade_pnls)
        
    def calculate_profit_factor(self, trade_pnls: List[float]) -> float:
        """Calculate profit factor."""
        if not trade_pnls:
            return 0.0
            
        gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
        
    def calculate_model_accuracy(self, predictions: List[float], 
                               actuals: List[float]) -> Dict[str, float]:
        """Calculate model prediction accuracy metrics."""
        if len(predictions) != len(actuals) or not predictions:
            return {'mse': 0.0, 'mae': 0.0, 'directional_accuracy': 0.0}
            
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        # Mean Squared Error
        mse = np.mean((predictions_array - actuals_array) ** 2)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions_array - actuals_array))
        
        # Directional Accuracy
        pred_directions = np.diff(predictions_array) > 0
        actual_directions = np.diff(actuals_array) > 0
        
        if len(pred_directions) == 0:
            directional_accuracy = 0.0
        else:
            directional_accuracy = np.mean(pred_directions == actual_directions)
            
        return {
            'mse': float(mse),
            'mae': float(mae),
            'directional_accuracy': float(directional_accuracy)
        }
        
    def calculate_portfolio_metrics(self, trades: List[Dict], 
                                  initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        if not trades:
            return {}
            
        # Extract PnLs
        pnls = [trade.get('pnl', 0) for trade in trades if trade.get('pnl') is not None]
        
        if not pnls:
            return {}
            
        # Calculate equity curve
        equity_curve = [initial_capital]
        for pnl in pnls:
            equity_curve.append(equity_curve[-1] + pnl)
            
        # Calculate returns
        returns = self.calculate_returns(equity_curve)
        
        return {
            'total_return': (equity_curve[-1] - initial_capital) / initial_capital,
            'total_trades': len(trades),
            'winning_trades': sum(1 for pnl in pnls if pnl > 0),
            'losing_trades': sum(1 for pnl in pnls if pnl < 0),
            'win_rate': self.calculate_win_rate(pnls),
            'profit_factor': self.calculate_profit_factor(pnls),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'average_win': np.mean([pnl for pnl in pnls if pnl > 0]) if any(pnl > 0 for pnl in pnls) else 0,
            'average_loss': np.mean([pnl for pnl in pnls if pnl < 0]) if any(pnl < 0 for pnl in pnls) else 0,
            'current_equity': equity_curve[-1]
        } 