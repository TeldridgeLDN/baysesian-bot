"""
Portfolio and risk management for cryptocurrency trading.
Handles position sizing, risk limits, and capital allocation.
"""

import logging
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position."""
    id: str
    entry_timestamp: datetime
    entry_price: float
    position_size: float
    position_type: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    confidence_score: float
    status: str = 'open'
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

class TradingConfig(BaseModel):
    initial_capital: float = 10000.0
    position_size: float = 0.1
    stop_loss: float = 0.05
    take_profit: float = 0.1
    min_price_change_pct: float = 2.0
    max_interval_width_pct: float = 3.0
    confidence_threshold: float = 0.60
    paper_trading: bool = True
    stop_loss_type: Literal['none', 'fixed', 'trailing'] = 'fixed'
    trailing_stop_pct: float = 0.02

class PortfolioManager:
    """Manages trading portfolio and risk parameters."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.cash = config.initial_capital
        self.position = 0.0  # Current position size in BTC
        self.position_entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # For trailing stops
        self.high_water_mark = 0.0
        
        # For performance tracking
        self.peak_portfolio_value = config.initial_capital
        self.max_drawdown = 0.0
        
    def get_total_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        position_value = self.position * current_price
        return self.cash + position_value
    
    def get_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        return {
            'cash': self.cash,
            'position': self.position,
            'position_entry_price': self.position_entry_price,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': self.total_pnl,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0,
            'total_return': (self.total_pnl / self.config.initial_capital) * 100,
            'max_drawdown': self.max_drawdown
        }
    
    def update(self, trade_result: Dict[str, Any]):
        """Update portfolio state based on trade result."""
        self.cash = trade_result['cash']
        self.position = trade_result['position']
        self.position_entry_price = trade_result['entry_price']
        
        if trade_result.get('is_closed', False):
            self.total_trades += 1
            pnl = trade_result.get('pnl', 0.0)
            self.total_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1
            self.high_water_mark = 0.0  # Reset on close
        else:  # Position opened
            self.high_water_mark = self.position_entry_price

    def update_performance_metrics(self, current_price: float):
        """Update performance metrics like max drawdown."""
        current_value = self.get_total_value(current_price)
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss has been triggered."""
        if self.position == 0 or self.config.stop_loss_type == 'none':
            return False

        if self.position > 0:  # Long position
            if self.config.stop_loss_type == 'fixed':
                stop_price = self.position_entry_price * (1 - self.config.stop_loss)
                return current_price <= stop_price
            elif self.config.stop_loss_type == 'trailing':
                self.high_water_mark = max(self.high_water_mark, current_price)
                stop_price = self.high_water_mark * (1 - self.config.trailing_stop_pct)
                return current_price <= stop_price
        
        elif self.position < 0:  # Short position
            if self.config.stop_loss_type == 'fixed':
                stop_price = self.position_entry_price * (1 + self.config.stop_loss)
                return current_price >= stop_price
            elif self.config.stop_loss_type == 'trailing':
                self.high_water_mark = min(self.high_water_mark, current_price)
                stop_price = self.high_water_mark * (1 + self.config.trailing_stop_pct)
                return current_price >= stop_price
        
        return False
        
    def can_open_position(self, position_size: float) -> bool:
        """Check if new position can be opened within risk limits."""
        # Implementation placeholder
        pass
        
    def calculate_position_size(self, confidence_score: float, 
                              current_price: float) -> float:
        """Calculate position size based on confidence and capital."""
        # Implementation placeholder
        pass
        
    def open_position(self, signal_data: Dict) -> Optional[Position]:
        """Open new trading position."""
        # Implementation placeholder
        pass
        
    def close_position(self, position_id: str, exit_price: float, 
                      reason: str) -> Dict:
        """Close existing position and calculate PnL."""
        # Implementation placeholder
        pass
        
    def update_stop_losses(self, current_price: float):
        """Update trailing stop losses for open positions."""
        # Implementation placeholder
        pass
        
    def get_portfolio_metrics(self) -> Dict:
        """Calculate current portfolio performance metrics."""
        # Implementation placeholder
        pass 