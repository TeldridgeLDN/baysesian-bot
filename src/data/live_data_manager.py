"""
Live Data Manager for fetching real-time cryptocurrency data from Binance.
Handles data collection, validation, and storage for the trading bot.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT library not available. Install with: pip install ccxt")

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data point structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    additional_data: Dict[str, Any] = None

class LiveDataManager:
    """
    Manages live data collection from Binance API.
    
    Features:
    - Real-time OHLCV data collection
    - Data validation and quality checks
    - Rate limiting and error handling
    - Historical data backfill
    - Data storage and caching
    """
    
    def __init__(self, api_key: str, secret_key: str, config: Any):
        """Initialize live data manager."""
        self.api_key = api_key
        self.secret_key = secret_key
        self.config = config
        
        # Exchange setup
        self.exchange = None
        self.initialized = False
        
        # Data storage
        self.data_cache: Dict[str, List[MarketData]] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Configuration
        self.symbols = ['BTC/USDT']  # Primary trading symbol
        self.timeframe = '1h'  # 1-hour candles
        self.max_cache_size = 1000  # Keep last 1000 data points
        self.update_interval = 60  # Update every 60 seconds
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        # Data quality tracking
        self.data_quality_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'data_gaps': 0,
            'invalid_data_points': 0
        }
        
        logger.info("LiveDataManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize connection to Binance API."""
        if not CCXT_AVAILABLE:
            logger.error("CCXT library not available - cannot fetch live data")
            return False
        
        try:
            # Initialize Binance exchange
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'sandbox': False,  # Use live data
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # Spot trading
                }
            })
            
            # Test connection
            await self._test_connection()
            
            # Load markets
            markets = await self.exchange.load_markets()
            logger.info(f"Connected to Binance - {len(markets)} markets available")
            
            # Initialize data cache for each symbol
            for symbol in self.symbols:
                self.data_cache[symbol] = []
                self.last_update[symbol] = datetime.now() - timedelta(hours=1)
            
            # Fetch initial historical data
            await self._fetch_initial_data()
            
            self.initialized = True
            logger.info("✅ LiveDataManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LiveDataManager: {e}")
            return False
    
    async def _test_connection(self):
        """Test API connection."""
        try:
            # Simple API call to test connection
            ticker = await self.exchange.fetch_ticker('BTC/USDT')
            logger.info(f"API connection test successful - BTC/USDT: ${ticker['last']:.2f}")
            
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            raise
    
    async def _fetch_initial_data(self):
        """Fetch initial historical data for analysis."""
        logger.info("Fetching initial historical data...")
        
        try:
            # Fetch last 180 hours of data (7.5 days)
            since = int((datetime.now() - timedelta(hours=180)).timestamp() * 1000)
            
            for symbol in self.symbols:
                # Rate limiting
                await self._rate_limit()
                
                # Fetch OHLCV data
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=180
                )
                
                # Convert to MarketData objects
                market_data_points = []
                for candle in ohlcv:
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(candle[0] / 1000),
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5])
                    )
                    
                    if self._validate_data_point(market_data):
                        market_data_points.append(market_data)
                    else:
                        self.data_quality_stats['invalid_data_points'] += 1
                
                self.data_cache[symbol] = market_data_points
                self.last_update[symbol] = datetime.now()
                
                logger.info(f"Loaded {len(market_data_points)} historical data points for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to fetch initial data: {e}")
            raise
    
    async def fetch_latest_data(self, symbol: str = 'BTC/USDT') -> Optional[MarketData]:
        """Fetch the latest market data for a symbol."""
        if not self.initialized:
            logger.error("LiveDataManager not initialized")
            return None
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Fetch latest ticker data
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Fetch latest OHLCV candle
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=1
            )
            
            self.data_quality_stats['total_requests'] += 1
            
            if ohlcv and len(ohlcv) > 0:
                candle = ohlcv[0]
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    additional_data={
                        'bid': ticker.get('bid'),
                        'ask': ticker.get('ask'),
                        'last': ticker.get('last'),
                        'change_24h': ticker.get('change'),
                        'percentage_24h': ticker.get('percentage')
                    }
                )
                
                if self._validate_data_point(market_data):
                    # Update cache
                    self._update_cache(symbol, market_data)
                    self.data_quality_stats['successful_requests'] += 1
                    self.last_update[symbol] = datetime.now()
                    
                    return market_data
                else:
                    self.data_quality_stats['invalid_data_points'] += 1
                    logger.warning(f"Invalid data point received for {symbol}")
            
            self.data_quality_stats['failed_requests'] += 1
            return None
            
        except Exception as e:
            self.data_quality_stats['failed_requests'] += 1
            logger.error(f"Error fetching latest data for {symbol}: {e}")
            return None
    
    def _validate_data_point(self, data: MarketData) -> bool:
        """Validate a market data point."""
        # Basic validation checks
        if data.open <= 0 or data.high <= 0 or data.low <= 0 or data.close <= 0:
            return False
        
        if data.volume < 0:
            return False
        
        # Price consistency checks
        if data.high < max(data.open, data.close):
            return False
        
        if data.low > min(data.open, data.close):
            return False
        
        # Extreme price movement check (>50% in one candle is suspicious)
        max_price = max(data.open, data.high, data.low, data.close)
        min_price = min(data.open, data.high, data.low, data.close)
        
        if max_price / min_price > 1.5:  # More than 50% movement
            logger.warning(f"Extreme price movement detected: {min_price} to {max_price}")
            return False
        
        # Timestamp validation (not too old, not in future)
        now = datetime.now()
        if data.timestamp > now + timedelta(minutes=5):  # Future data
            return False
        
        if data.timestamp < now - timedelta(days=30):  # Very old data
            return False
        
        return True
    
    def _update_cache(self, symbol: str, data: MarketData):
        """Update the data cache with new data point."""
        if symbol not in self.data_cache:
            self.data_cache[symbol] = []
        
        cache = self.data_cache[symbol]
        
        # Check if this data point already exists (avoid duplicates)
        for existing_data in cache:
            if existing_data.timestamp == data.timestamp:
                return  # Data point already exists
        
        # Add new data point
        cache.append(data)
        
        # Sort by timestamp
        cache.sort(key=lambda x: x.timestamp)
        
        # Trim cache if too large
        if len(cache) > self.max_cache_size:
            self.data_cache[symbol] = cache[-self.max_cache_size:]
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_historical_data(self, symbol: str = 'BTC/USDT', 
                          hours: int = 24) -> List[MarketData]:
        """Get historical data from cache."""
        if symbol not in self.data_cache:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cache = self.data_cache[symbol]
        
        return [data for data in cache if data.timestamp >= cutoff_time]
    
    def get_latest_price(self, symbol: str = 'BTC/USDT') -> Optional[float]:
        """Get the latest price for a symbol."""
        if symbol not in self.data_cache or not self.data_cache[symbol]:
            return None
        
        latest_data = self.data_cache[symbol][-1]
        return latest_data.close
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get data quality statistics."""
        total_requests = self.data_quality_stats['total_requests']
        success_rate = (
            self.data_quality_stats['successful_requests'] / max(1, total_requests) * 100
        )
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.data_quality_stats['successful_requests'],
            'failed_requests': self.data_quality_stats['failed_requests'],
            'success_rate_pct': success_rate,
            'data_gaps': self.data_quality_stats['data_gaps'],
            'invalid_data_points': self.data_quality_stats['invalid_data_points'],
            'cache_status': {
                symbol: {
                    'data_points': len(cache),
                    'latest_update': self.last_update.get(symbol, 'Never').isoformat() 
                    if isinstance(self.last_update.get(symbol), datetime) else 'Never',
                    'oldest_data': cache[0].timestamp.isoformat() if cache else 'No data',
                    'newest_data': cache[-1].timestamp.isoformat() if cache else 'No data'
                }
                for symbol, cache in self.data_cache.items()
            }
        }
    
    def to_dataframe(self, symbol: str = 'BTC/USDT', hours: int = 24) -> pd.DataFrame:
        """Convert cached data to pandas DataFrame for analysis."""
        data_points = self.get_historical_data(symbol, hours)
        
        if not data_points:
            return pd.DataFrame()
        
        data_dict = {
            'timestamp': [d.timestamp for d in data_points],
            'open': [d.open for d in data_points],
            'high': [d.high for d in data_points],
            'low': [d.low for d in data_points],
            'close': [d.close for d in data_points],
            'volume': [d.volume for d in data_points]
        }
        
        df = pd.DataFrame(data_dict)
        df.set_index('timestamp', inplace=True)
        return df
    
    async def start_continuous_updates(self):
        """Start continuous data updates in background."""
        logger.info("Starting continuous data updates...")
        
        while self.initialized:
            try:
                # Update data for all symbols
                for symbol in self.symbols:
                    await self.fetch_latest_data(symbol)
                
                # Wait for next update cycle
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous updates: {e}")
                await asyncio.sleep(30)  # Brief pause on error
    
    def stop(self):
        """Stop the data manager."""
        self.initialized = False
        logger.info("LiveDataManager stopped")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.stop()