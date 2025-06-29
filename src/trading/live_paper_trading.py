"""
Live Paper Trading Integration.

This module integrates the paper trading system with the main trading engine,
enabling live parameter adaptation using real market data without financial risk.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from .paper_trading import PaperTradingEngine, PaperTradingMode
from .engine import TradingEngine
from .adaptive_parameters import AdaptiveParameterManager, ParameterAdjustment
from ..data.collectors import PriceDataCollector
from ..utils.config import ConfigManager, TradingConfig

logger = logging.getLogger(__name__)

class LivePaperTradingManager:
    """
    Manages live paper trading with real-time parameter adaptation.
    
    Features:
    - Live market data integration
    - Real-time paper trading execution
    - Automatic parameter adaptation based on performance
    - Performance comparison and A/B testing
    - Safe strategy validation before live deployment
    """
    
    def __init__(self, config_manager: ConfigManager, 
                 initial_capital: float = 10000.0,
                 adaptation_enabled: bool = True):
        """Initialize live paper trading manager."""
        self.config_manager = config_manager
        self.trading_config = config_manager.get_trading_config()
        self.initial_capital = initial_capital
        
        # Initialize paper trading engine
        self.paper_engine = PaperTradingEngine(
            initial_capital=initial_capital,
            mode=PaperTradingMode.LIVE_PAPER
        )
        
        # Initialize adaptive parameter manager
        self.adaptive_manager = AdaptiveParameterManager()
        
        # Trading state
        self.is_running = False
        self.current_parameters = self._get_current_parameters()
        self.price_data_history: List[Dict[str, Any]] = []
        self.last_price_update = None
        
        # Performance tracking
        self.session_start_time = None
        self.adaptation_events: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Settings
        self.adaptation_enabled = adaptation_enabled
        self.data_update_interval = 60  # seconds
        self.adaptation_check_interval = 300  # 5 minutes
        self.min_trades_for_adaptation = 5
        
        logger.info(f"LivePaperTradingManager initialized: capital=${initial_capital:,.0f}, "
                   f"adaptation={'enabled' if adaptation_enabled else 'disabled'}")
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current trading parameters from config."""
        return {
            'confidence_threshold': self.trading_config.confidence_threshold,
            'min_price_change_pct': self.trading_config.min_price_change_pct,
            'max_interval_width_pct': self.trading_config.max_interval_width_pct,
            'max_position_size': self.trading_config.max_position_size,
            'stop_loss_multiplier': self.trading_config.stop_loss_multiplier,
            'take_profit_ratio': self.trading_config.take_profit_ratio,
            'position_timeout_hours': self.trading_config.position_timeout_hours,
            'risk_grading_enabled': self.trading_config.risk_grading_enabled
        }
    
    async def start_live_paper_trading(self) -> bool:
        """Start live paper trading session."""
        try:
            if self.is_running:
                logger.warning("Live paper trading already running")
                return False
            
            self.is_running = True
            self.session_start_time = datetime.now()
            
            logger.info("🚀 Starting live paper trading session...")
            logger.info(f"Initial parameters: {self.current_parameters}")
            
            # Start main trading loop
            await self._trading_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting live paper trading: {e}")
            self.is_running = False
            return False
    
    async def stop_live_paper_trading(self) -> Dict[str, Any]:
        """Stop live paper trading and return session summary."""
        try:
            self.is_running = False
            
            # Get final performance summary
            current_price = self.last_price_update.get('price', 50000) if self.last_price_update else 50000
            final_performance = self.paper_engine.get_performance_summary(current_price)
            
            # Get adaptation summary
            adaptation_summary = self.paper_engine.get_adaptation_summary()
            
            session_summary = {
                'session_duration_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
                'final_performance': final_performance,
                'adaptation_summary': adaptation_summary,
                'total_adaptations': len(self.adaptation_events),
                'price_updates_received': len(self.price_data_history),
                'parameters_at_end': self.current_parameters
            }
            
            logger.info("📊 Live paper trading session stopped")
            logger.info(f"Session summary: {session_summary}")
            
            return session_summary
            
        except Exception as e:
            logger.error(f"Error stopping live paper trading: {e}")
            return {'error': str(e)}
    
    async def _trading_loop(self):
        """Main trading loop for live paper trading."""
        logger.info("Starting live paper trading loop...")
        
        while self.is_running:
            try:
                # Get latest market data
                await self._update_market_data()
                
                # Process trading signals
                await self._process_trading_signals()
                
                # Check for position exits
                await self._check_position_management()
                
                # Check for parameter adaptation
                if self.adaptation_enabled:
                    await self._check_parameter_adaptation()
                
                # Wait before next iteration
                await asyncio.sleep(self.data_update_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _update_market_data(self):
        """Update market data for trading decisions."""
        try:
            # In a real implementation, this would fetch live data
            # For now, simulate with realistic price movements
            if self.last_price_update:
                last_price = self.last_price_update['price']
                # Simulate realistic price movement (±2% max)
                price_change = (np.random.random() - 0.5) * 0.04  # ±2%
                new_price = last_price * (1 + price_change)
            else:
                new_price = 50000.0  # Starting price
            
            # Create price data point
            price_data = {
                'timestamp': datetime.now(),
                'price': new_price,
                'volume': np.random.uniform(100, 1000),  # Simulated volume
                'source': 'paper_trading_simulation'
            }
            
            self.price_data_history.append(price_data)
            self.last_price_update = price_data
            
            # Keep only recent history (last 100 points)
            if len(self.price_data_history) > 100:
                self.price_data_history = self.price_data_history[-100:]
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _process_trading_signals(self):
        """Process trading signals and execute paper trades."""
        try:
            if not self.last_price_update or len(self.price_data_history) < 10:
                return  # Need sufficient data
            
            current_price = self.last_price_update['price']
            
            # Generate mock prediction data (in real implementation, use ML model)
            prediction_data = self._generate_mock_prediction(current_price)
            
            # Generate trading signal based on parameters
            signal = self._generate_trading_signal(prediction_data)
            
            # Execute paper trade if signal generated
            if signal != 'HOLD':
                trade_result = self.paper_engine.execute_paper_trade(
                    signal=signal,
                    current_price=current_price,
                    prediction_data=prediction_data,
                    parameters=self.current_parameters
                )
                
                if trade_result:
                    logger.info(f"Executed paper trade: {trade_result['action']} @ ${current_price:,.0f}")
            
        except Exception as e:
            logger.error(f"Error processing trading signals: {e}")
    
    def _generate_mock_prediction(self, current_price: float) -> Dict[str, Any]:
        """Generate mock ML prediction data for testing."""
        # Simulate Bayesian LSTM prediction with uncertainty
        price_change_pct = np.random.normal(0, 2.0)  # ±2% typical change
        predicted_price = current_price * (1 + price_change_pct / 100)
        
        # Simulate uncertainty (wider intervals = more uncertainty)
        uncertainty_pct = np.random.uniform(15, 25)  # 15-25% uncertainty
        interval_half_width = current_price * uncertainty_pct / 100 / 2
        
        confidence = max(0.3, min(0.9, 1.0 - (uncertainty_pct - 15) / 10))  # Inverse of uncertainty
        
        return {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'interval_width_pct': uncertainty_pct,
            'confidence_lower': predicted_price - interval_half_width,
            'confidence_upper': predicted_price + interval_half_width,
            'price_change_pct': price_change_pct,
            'model_version': 'paper_trading_mock_v1.0'
        }
    
    def _generate_trading_signal(self, prediction_data: Dict[str, Any]) -> str:
        """Generate trading signal based on prediction and parameters."""
        try:
            confidence = prediction_data['confidence']
            price_change_pct = abs(prediction_data['price_change_pct'])
            interval_width_pct = prediction_data['interval_width_pct']
            
            # Check parameter thresholds
            confidence_threshold = self.current_parameters['confidence_threshold']
            min_price_change = self.current_parameters['min_price_change_pct']
            max_uncertainty = self.current_parameters['max_interval_width_pct']
            
            # Generate signal based on thresholds
            if (confidence >= confidence_threshold and 
                price_change_pct >= min_price_change and 
                interval_width_pct <= max_uncertainty):
                
                # Determine direction based on predicted price change
                if prediction_data['price_change_pct'] > 0:
                    return 'BUY'
                else:
                    return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return 'HOLD'
    
    async def _check_position_management(self):
        """Check for position exits (stop loss, take profit, timeout)."""
        try:
            if not self.last_price_update:
                return
            
            current_price = self.last_price_update['price']
            exit_result = self.paper_engine.check_position_exits(current_price)
            
            if exit_result:
                logger.info(f"Position closed: {exit_result['exit_reason']} @ ${current_price:,.0f}, "
                           f"P&L: ${exit_result['pnl']:,.0f} ({exit_result['pnl_pct']:+.2f}%)")
                
                # Record performance snapshot
                performance = self.paper_engine.get_performance_summary(current_price)
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'performance': performance,
                    'trigger': 'position_exit'
                })
            
        except Exception as e:
            logger.error(f"Error checking position management: {e}")
    
    async def _check_parameter_adaptation(self):
        """Check if parameter adaptation should be triggered."""
        try:
            if not self.last_price_update:
                return
            
            current_price = self.last_price_update['price']
            current_performance = self.paper_engine.get_performance_summary(current_price)
            
            # Check if we should adapt parameters
            adaptation_data = self.paper_engine.check_adaptation_trigger(
                current_performance, 
                self.current_parameters
            )
            
            if adaptation_data:
                logger.info(f"Parameter adaptation triggered: {adaptation_data['total_suggestions']} suggestions")
                
                # Apply adaptation
                new_parameters = self.paper_engine.apply_parameter_adaptation(
                    adaptation_data, 
                    self.current_parameters
                )
                
                if new_parameters != self.current_parameters:
                    # Record adaptation event
                    adaptation_event = {
                        'timestamp': datetime.now(),
                        'old_parameters': self.current_parameters.copy(),
                        'new_parameters': new_parameters.copy(),
                        'trigger_performance': current_performance,
                        'adaptation_data': adaptation_data
                    }
                    
                    self.adaptation_events.append(adaptation_event)
                    self.current_parameters = new_parameters
                    
                    logger.info("✅ Parameters adapted successfully")
                    
                    # Also update trading config for consistency
                    self._update_trading_config(new_parameters)
            
        except Exception as e:
            logger.error(f"Error checking parameter adaptation: {e}")
    
    def _update_trading_config(self, new_parameters: Dict[str, Any]):
        """Update trading config with new parameters."""
        try:
            for param_name, value in new_parameters.items():
                if hasattr(self.trading_config, param_name):
                    setattr(self.trading_config, param_name, value)
            
            logger.debug("Trading config updated with new parameters")
            
        except Exception as e:
            logger.error(f"Error updating trading config: {e}")
    
    def get_live_status(self) -> Dict[str, Any]:
        """Get current live paper trading status."""
        try:
            current_price = self.last_price_update.get('price', 0) if self.last_price_update else 0
            performance = self.paper_engine.get_performance_summary(current_price)
            adaptation_summary = self.paper_engine.get_adaptation_summary()
            
            return {
                'is_running': self.is_running,
                'session_duration_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600 if self.session_start_time else 0,
                'current_price': current_price,
                'current_parameters': self.current_parameters,
                'performance': performance,
                'adaptation_status': adaptation_summary,
                'total_adaptations': len(self.adaptation_events),
                'data_updates_received': len(self.price_data_history),
                'last_update': self.last_price_update['timestamp'].isoformat() if self.last_price_update else None,
                'adaptation_enabled': self.adaptation_enabled
            }
            
        except Exception as e:
            logger.error(f"Error getting live status: {e}")
            return {'error': str(e)}
    
    def start_ab_test(self, parameter_sets: Dict[str, Dict[str, Any]], 
                     test_duration_hours: float = 24.0) -> bool:
        """Start A/B testing different parameter sets."""
        try:
            success = self.paper_engine.start_ab_test(parameter_sets)
            
            if success:
                logger.info(f"Started A/B test with {len(parameter_sets)} parameter sets for {test_duration_hours} hours")
                
                # Schedule test end
                asyncio.create_task(self._end_ab_test_after_duration(test_duration_hours))
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting A/B test: {e}")
            return False
    
    async def _end_ab_test_after_duration(self, duration_hours: float):
        """End A/B test after specified duration."""
        await asyncio.sleep(duration_hours * 3600)
        
        if self.paper_engine.ab_test_active:
            logger.info("A/B test duration completed, returning to single parameter set")
            self.paper_engine.ab_test_active = False
            self.paper_engine.mode = PaperTradingMode.LIVE_PAPER
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export complete session data for analysis."""
        try:
            base_export = self.paper_engine.export_trading_session()
            
            # Add live trading specific data
            base_export['live_paper_trading'] = {
                'adaptation_events': self.adaptation_events,
                'performance_history': self.performance_history,
                'price_data_points': len(self.price_data_history),
                'adaptation_enabled': self.adaptation_enabled,
                'session_start': self.session_start_time.isoformat() if self.session_start_time else None
            }
            
            return base_export
            
        except Exception as e:
            logger.error(f"Error exporting session data: {e}")
            return {'error': str(e)}

# Import numpy for mock data generation
import numpy as np