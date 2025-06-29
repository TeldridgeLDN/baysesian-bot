"""
Trade execution interface for cryptocurrency trading.
Handles order placement, monitoring, and execution logic.
"""

import logging
import uuid
from typing import Dict, Optional, List, Any, Literal
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel

from utils.config import TradingConfig

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Represents a trading order."""
    id: str
    position_id: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: float
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None

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
    trading_fee: float = 0.001  # 0.1% trading fee
    slippage: float = 0.0002   # 0.02% slippage

class TradeExecutor:
    """
    Handles trade execution, both simulated (paper trading) and live.
    """
    def __init__(self, config: TradingConfig):
        """
        Initialize the trade executor.
        Args:
            config: A TradingConfig object.
        """
        self.config = config
        self.paper_trading = config.paper_trading
        self.open_orders: List[Order] = []
        self.trade_history: List[Order] = []
        
        if self.paper_trading:
            logger.info("TradeExecutor initialized in PAPER TRADING mode.")
        else:
            logger.warning("TradeExecutor initialized in LIVE TRADING mode. Real funds will be used.")
            # self.exchange_api = self._initialize_exchange_api() # Placeholder for real exchange integration

    def _close_long(self, price: float, portfolio: Any) -> Dict[str, Any]:
        # Apply slippage and fees when selling
        effective_price = price * (1 - self.config.slippage)
        gross_proceeds = portfolio.position * effective_price
        trading_fee = gross_proceeds * self.config.trading_fee
        net_proceeds = gross_proceeds - trading_fee
        
        pnl = net_proceeds - (portfolio.position * portfolio.position_entry_price)
        return {
            'cash': portfolio.cash + net_proceeds,
            'position': 0.0,
            'entry_price': 0.0,
            'is_closed': True,
            'pnl': pnl,
            'trading_fee': trading_fee,
            'slippage_cost': portfolio.position * price * self.config.slippage
        }

    def _close_short(self, price: float, portfolio: Any) -> Dict[str, Any]:
        # Apply slippage and fees when covering short
        effective_price = price * (1 + self.config.slippage)  # Buy back at higher price
        gross_cost = -portfolio.position * effective_price  # Position is negative for shorts
        trading_fee = gross_cost * self.config.trading_fee
        net_cost = gross_cost + trading_fee
        
        # For shorts: pnl = (entry_price - exit_price) * abs(position_size)
        pnl = (-portfolio.position * portfolio.position_entry_price) - net_cost
        return {
            'cash': portfolio.cash - net_cost,
            'position': 0.0,
            'entry_price': 0.0,
            'is_closed': True,
            'pnl': pnl,
            'trading_fee': trading_fee,
            'slippage_cost': -portfolio.position * price * self.config.slippage
        }

    def execute_trade(self, signal: str, price: float, portfolio: Any, position_ratio: float = 1.0) -> Optional[Dict[str, Any]]:
        """Execute a trade based on the signal and confidence."""
        if signal == 'CLOSE':
            if portfolio.position > 0:
                return self._close_long(price, portfolio)
            elif portfolio.position < 0:
                return self._close_short(price, portfolio)
            return None

        current_value = portfolio.get_total_value(price)
        trade_size = current_value * self.config.position_size * position_ratio
        
        if signal == 'LONG':
            if portfolio.position < 0:
                return self._close_short(price, portfolio)
            
            if portfolio.position == 0:
                # Apply slippage and fees when buying
                effective_price = price * (1 + self.config.slippage)
                position_size = trade_size / effective_price
                gross_cost = position_size * effective_price
                trading_fee = gross_cost * self.config.trading_fee
                total_cost = gross_cost + trading_fee
                
                return {
                    'cash': portfolio.cash - total_cost,
                    'position': position_size,
                    'entry_price': effective_price,  # Use effective price including slippage
                    'is_closed': False,
                    'pnl': 0.0,
                    'trading_fee': trading_fee,
                    'slippage_cost': position_size * price * self.config.slippage
                }
                
        elif signal == 'SHORT':
            if portfolio.position > 0:
                return self._close_long(price, portfolio)
            
            if portfolio.position == 0:
                # Apply slippage and fees when shorting
                effective_price = price * (1 - self.config.slippage)  # Sell at lower price
                position_size = trade_size / effective_price
                gross_proceeds = position_size * effective_price
                trading_fee = gross_proceeds * self.config.trading_fee
                net_proceeds = gross_proceeds - trading_fee
                
                return {
                    'cash': portfolio.cash + net_proceeds,
                    'position': -position_size,
                    'entry_price': effective_price,  # Use effective price including slippage
                    'is_closed': False,
                    'pnl': 0.0,
                    'trading_fee': trading_fee,
                    'slippage_cost': position_size * price * self.config.slippage
                }
        
        return None

    def execute_market_order(self, order_data: Dict) -> Dict:
        """
        Execute a market order.
        """
        side = OrderSide.BUY if order_data['side'] == 'buy' else OrderSide.SELL
        order = Order(
            id=str(uuid.uuid4()),
            position_id=order_data['position_id'],
            order_type=OrderType.MARKET,
            side=side,
            quantity=order_data['quantity'],
            price=order_data.get('current_price')
        )
        
        if self.paper_trading:
            return self._simulate_market_order(order)
        
        logger.warning("Live order execution not implemented.")
        return {'success': False, 'error': 'Live trading not implemented.'}

    def _simulate_market_order(self, order: Order) -> Dict:
        """Simulate the execution of a market order for paper trading."""
        order.status = OrderStatus.FILLED
        order.filled_price = order.price
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        
        self.trade_history.append(order)
        logger.info(f"Simulated {order.side.name} market order {order.id} for {order.quantity:.6f} at ${order.price or 0:,.2f}.")
        
        return {
            'success': True, 'order_id': order.id,
            'filled_price': order.filled_price, 'filled_quantity': order.filled_quantity
        }

    def place_stop_loss_order(self, position_id: str, stop_price: float, quantity: float):
        """Places a stop-loss order."""
        order = Order(
            id=str(uuid.uuid4()), position_id=position_id, order_type=OrderType.STOP_LOSS,
            side=OrderSide.SELL, quantity=quantity, price=stop_price
        )
        self.open_orders.append(order)
        logger.info(f"Placed STOP LOSS order for position {position_id} at ${stop_price:,.2f}.")

    def place_take_profit_order(self, position_id: str, take_profit_price: float, quantity: float):
        """Places a take-profit order."""
        order = Order(
            id=str(uuid.uuid4()), position_id=position_id, order_type=OrderType.TAKE_PROFIT,
            side=OrderSide.SELL, quantity=quantity, price=take_profit_price
        )
        self.open_orders.append(order)
        logger.info(f"Placed TAKE PROFIT order for pos {position_id} at ${take_profit_price:,.2f}.")

    def cancel_position_orders(self, position_id: str):
        """Cancel all open orders associated with a position."""
        orders_to_cancel = [o for o in self.open_orders if o.position_id == position_id]
        for order in orders_to_cancel:
            order.status = OrderStatus.CANCELED
            self.open_orders.remove(order)
            logger.info(f"Canceled order {order.id} ({order.order_type.name}) for pos {position_id}.")

    def reset_paper_trading(self):
        """Resets the state of the paper trading executor."""
        self.open_orders = []
        self.trade_history = []
        logger.info("TradeExecutor state has been reset.")

    def update_config(self, new_config: TradingConfig):
        """Update the configuration for the trade executor."""
        self.config = new_config
        self.paper_trading = new_config.paper_trading
        logger.info(f"TradeExecutor config updated. Paper trading: {self.paper_trading}")

    def _initialize_exchange_api(self):
        # Placeholder for real exchange API initialization
        raise NotImplementedError("Live exchange API is not implemented.")

    def get_paper_trading_balance(self) -> Dict:
        """Get current paper trading balance."""
        return {
            'cash_balance': self.paper_trading_balance,
            'btc_balance': self.paper_trading_btc,
            'total_value_usd': self.paper_trading_balance + (self.paper_trading_btc * 50000)  # Approximate BTC price
        }
    
    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get recent order history."""
        sorted_orders = sorted(self.orders, key=lambda x: x.created_at, reverse=True)
        return [
            {
                'id': order.id,
                'position_id': order.position_id,
                'type': order.order_type.value,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': order.price,
                'status': order.status.value,
                'created_at': order.created_at.isoformat(),
                'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                'filled_price': order.filled_price,
                'commission': order.commission
            }
            for order in sorted_orders[:limit]
        ] 