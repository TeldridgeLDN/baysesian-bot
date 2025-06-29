from typing import List, Dict, Any
from trading.engine import TradingEngine, TradingConfig
from utils.charts import plot_performance

class Backtester:
    def __init__(self, config: TradingConfig):
        self.trading_engine = TradingEngine(config)
        
    def run(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run backtest on historical data."""
        results = []
        for data_point in data:
            result = self.trading_engine.process_data_point(data_point)
            if result:
                results.append(result)
        
        # Generate performance chart
        if results:
            plot_performance(results)
            
        return self.trading_engine.get_portfolio_state()
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get backtesting results."""
        # This method might need to be implemented if you need detailed results
        # For now, it's a placeholder
        return [] 