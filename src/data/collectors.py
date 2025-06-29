"""
Data collectors for cryptocurrency price data.
Handles API data fetching from CoinGecko and Binance with robust error handling,
rate limiting, and automated failover between sources.
"""

import requests
import time
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import statistics

from ..utils.config import ConfigManager
from .storage import DatabaseManager, PriceData

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Available data sources for price data."""
    COINGECKO = "coingecko"
    BINANCE = "binance"

@dataclass
class APIResponse:
    """Wrapper for API response data."""
    success: bool
    data: Optional[List[Dict]] = None
    error: Optional[str] = None
    source: Optional[DataSource] = None
    timestamp: Optional[datetime] = None

class RateLimiter:
    """Rate limiter for API requests with per-source tracking."""
    
    def __init__(self):
        self._requests = {}  # source -> [(timestamp, count)]
        self._lock = asyncio.Lock()
    
    async def can_make_request(self, source: DataSource, max_per_minute: int) -> bool:
        """Check if we can make a request without exceeding rate limits."""
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            
            # Clean old requests
            if source in self._requests:
                self._requests[source] = [
                    (ts, count) for ts, count in self._requests[source] 
                    if ts > cutoff
                ]
            else:
                self._requests[source] = []
            
            # Count current requests in the last minute
            current_count = sum(count for _, count in self._requests[source])
            return current_count < max_per_minute
    
    async def record_request(self, source: DataSource, count: int = 1):
        """Record a request for rate limiting purposes."""
        async with self._lock:
            now = datetime.now()
            if source not in self._requests:
                self._requests[source] = []
            self._requests[source].append((now, count))

class DataValidator:
    """Validates fetched price data for quality and outliers."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.max_price_change_pct = self.config.trading.max_price_change_pct
        self.min_volume_threshold = 1000
    
    def validate_ohlcv_data(self, data: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate OHLCV data structure and values."""
        errors = []
        
        if not data:
            errors.append("Empty data set")
            return False, errors
        
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        for i, candle in enumerate(data):
            # Check required fields
            missing_fields = [field for field in required_fields if field not in candle]
            if missing_fields:
                errors.append(f"Candle {i}: Missing fields {missing_fields}")
                continue
            
            # Validate OHLC relationships
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            if not (l <= o <= h and l <= c <= h):
                errors.append(f"Candle {i}: Invalid OHLC relationship")
            
            # Check for zero or negative values
            if any(val <= 0 for val in [o, h, l, c]):
                errors.append(f"Candle {i}: Zero or negative price values")
            
            # Check volume
            if candle['volume'] < 0:
                errors.append(f"Candle {i}: Negative volume")
            
            # Check for extreme price changes
            if i > 0:
                prev_close = data[i-1]['close']
                price_change_pct = abs((c - prev_close) / prev_close * 100)
                if price_change_pct > self.max_price_change_pct:
                    errors.append(f"Candle {i}: Extreme price change {price_change_pct:.2f}%")
        
        return len(errors) == 0, errors
    
    def detect_outliers(self, data: List[Dict], field: str = 'close') -> List[int]:
        """Detect outliers using IQR method."""
        if len(data) < 10:  # Need minimum data for outlier detection
            return []
        
        values = [candle[field] for candle in data if field in candle]
        if len(values) < 10:
            return []
        
        # Calculate IQR
        q1 = statistics.quantiles(values, n=4)[0]
        q3 = statistics.quantiles(values, n=4)[2]
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outlier indices
        outliers = []
        for i, candle in enumerate(data):
            if field in candle:
                value = candle[field]
                if value < lower_bound or value > upper_bound:
                    outliers.append(i)
        
        return outliers

class PriceDataCollector:
    """Collects price data from various cryptocurrency APIs with robust error handling."""
    
    def __init__(self, config: ConfigManager, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.rate_limiter = RateLimiter()
        self.validator = DataValidator(config)
        
        # API configurations
        self.coingecko_config = self.config.api.coingecko
        self.binance_config = self.config.api.binance
        
        # Request session with timeouts
        self.session = requests.Session()
        self.session.timeout = (10, 30)  # (connect, read) timeouts
        
        logger.info("PriceDataCollector initialized")
    
    async def collect_historical_data(self, days: int = None) -> APIResponse:
        """Collect historical price data with automatic failover."""
        if days is None:
            days = self.config.trading.historical_days
        
        logger.info(f"Starting historical data collection for {days} days")
        
        # Try primary source first
        primary_source = DataSource(self.coingecko_config['name'])
        response = await self._fetch_with_retry(primary_source, days)
        
        if response.success:
            logger.info(f"Successfully collected data from primary source: {primary_source.value}")
            return response
        
        # Fallback to backup source
        backup_source = DataSource(self.binance_config['name'])
        logger.warning(f"Primary source failed, trying backup: {backup_source.value}")
        
        response = await self._fetch_with_retry(backup_source, days)
        
        if response.success:
            logger.info(f"Successfully collected data from backup source: {backup_source.value}")
        else:
            logger.error("Both primary and backup sources failed")
        
        return response
    
    async def collect_current_data(self) -> APIResponse:
        """Collect current price data."""
        logger.debug("Collecting current price data")
        
        # Try primary source first
        primary_source = DataSource(self.coingecko_config['name'])
        response = await self._fetch_current_with_retry(primary_source)
        
        if response.success:
            return response
        
        # Fallback to backup source
        backup_source = DataSource(self.binance_config['name'])
        logger.warning("Primary source failed for current data, trying backup")
        
        return await self._fetch_current_with_retry(backup_source)
    
    async def _fetch_with_retry(self, source: DataSource, days: int) -> APIResponse:
        """Fetch data with exponential backoff retry."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Check rate limits
                rate_limit = (self.coingecko_config["rate_limit_per_minute"] 
                            if source == DataSource.COINGECKO 
                            else self.binance_config["rate_limit_per_minute"])
                
                if not await self.rate_limiter.can_make_request(source, rate_limit):
                    logger.warning(f"Rate limit reached for {source.value}, waiting...")
                    await asyncio.sleep(60)  # Wait a minute
                
                # Make the request
                if source == DataSource.COINGECKO:
                    response = await self._fetch_coingecko_historical(days)
                else:
                    response = await self._fetch_binance_historical(days)
                
                # Record the request for rate limiting
                await self.rate_limiter.record_request(source)
                
                if response.success:
                    return response
                
                # If not successful, log and retry
                logger.warning(f"Attempt {attempt + 1} failed for {source.value}: {response.error}")
                
            except Exception as e:
                logger.error(f"Exception in attempt {attempt + 1} for {source.value}: {str(e)}")
            
            # Exponential backoff
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        
        return APIResponse(success=False, error=f"All {max_retries} attempts failed", source=source)
    
    async def _fetch_current_with_retry(self, source: DataSource) -> APIResponse:
        """Fetch current data with retry logic."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Check rate limits
                rate_limit = (self.coingecko_config["rate_limit_per_minute"] 
                            if source == DataSource.COINGECKO 
                            else self.binance_config["rate_limit_per_minute"])
                
                if not await self.rate_limiter.can_make_request(source, rate_limit):
                    await asyncio.sleep(60)
                
                # Make the request
                if source == DataSource.COINGECKO:
                    response = await self._fetch_coingecko_current()
                else:
                    response = await self._fetch_binance_current()
                
                await self.rate_limiter.record_request(source)
                
                if response.success:
                    return response
                
                logger.warning(f"Current data attempt {attempt + 1} failed for {source.value}")
                
            except Exception as e:
                logger.error(f"Exception in current data attempt {attempt + 1}: {str(e)}")
            
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        
        return APIResponse(success=False, error="All attempts failed", source=source)
    
    async def _fetch_coingecko_historical(self, days: int) -> APIResponse:
        """Fetch historical data from CoinGecko API."""
        try:
            url = f"{self.coingecko_config['base_url']}/coins/bitcoin/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }
            
            response = self.session.get(url, params=params, timeout=self.coingecko_config['timeout_seconds'])
            response.raise_for_status()
            
            data = response.json()
            
            # Convert CoinGecko format to our standard format
            ohlcv_data = self._convert_coingecko_to_ohlcv(data)
            
            # Validate data
            is_valid, errors = self.validator.validate_ohlcv_data(ohlcv_data)
            if not is_valid:
                return APIResponse(
                    success=False, 
                    error=f"Data validation failed: {'; '.join(errors[:5])}", 
                    source=DataSource.COINGECKO
                )
            
            return APIResponse(
                success=True, 
                data=ohlcv_data, 
                source=DataSource.COINGECKO,
                timestamp=datetime.now()
            )
            
        except requests.exceptions.RequestException as e:
            return APIResponse(success=False, error=f"Request error: {str(e)}", source=DataSource.COINGECKO)
        except Exception as e:
            return APIResponse(success=False, error=f"Unexpected error: {str(e)}", source=DataSource.COINGECKO)
    
    async def _fetch_coingecko_current(self) -> APIResponse:
        """Fetch current price from CoinGecko API."""
        try:
            url = f"{self.coingecko_config['base_url']}/simple/price"
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=self.coingecko_config['timeout_seconds'])
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to our format
            current_data = [{
                'timestamp': int(datetime.now().timestamp()),
                'open': data['bitcoin']['usd'],  # Approximate for current
                'high': data['bitcoin']['usd'],
                'low': data['bitcoin']['usd'],
                'close': data['bitcoin']['usd'],
                'volume': data['bitcoin'].get('usd_24h_vol', 0)
            }]
            
            return APIResponse(
                success=True, 
                data=current_data, 
                source=DataSource.COINGECKO,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return APIResponse(success=False, error=str(e), source=DataSource.COINGECKO)
    
    async def _fetch_binance_historical(self, days: int) -> APIResponse:
        """Fetch historical data from Binance API."""
        try:
            url = f"{self.binance_config['base_url']}/klines"
            
            # Calculate start time
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = self.session.get(url, params=params, timeout=self.binance_config['timeout_seconds'])
            response.raise_for_status()
            
            data = response.json()
            
            # Convert Binance format to our standard format
            ohlcv_data = self._convert_binance_to_ohlcv(data)
            
            # Validate data
            is_valid, errors = self.validator.validate_ohlcv_data(ohlcv_data)
            if not is_valid:
                return APIResponse(
                    success=False, 
                    error=f"Data validation failed: {'; '.join(errors[:5])}", 
                    source=DataSource.BINANCE
                )
            
            return APIResponse(
                success=True, 
                data=ohlcv_data, 
                source=DataSource.BINANCE,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return APIResponse(success=False, error=str(e), source=DataSource.BINANCE)
    
    async def _fetch_binance_current(self) -> APIResponse:
        """Fetch current price from Binance API."""
        try:
            url = f"{self.binance_config['base_url']}/ticker/24hr"
            params = {'symbol': 'BTCUSDT'}
            
            response = self.session.get(url, params=params, timeout=self.binance_config['timeout_seconds'])
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to our format
            current_data = [{
                'timestamp': int(datetime.now().timestamp()),
                'open': float(data['openPrice']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'close': float(data['lastPrice']),
                'volume': float(data['volume'])
            }]
            
            return APIResponse(
                success=True, 
                data=current_data, 
                source=DataSource.BINANCE,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return APIResponse(success=False, error=str(e), source=DataSource.BINANCE)
    
    def _convert_coingecko_to_ohlcv(self, data: Dict) -> List[Dict]:
        """Convert CoinGecko market_chart data to OHLCV format."""
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not prices:
            return []
        
        # CoinGecko doesn't provide OHLC, so we approximate
        ohlcv_data = []
        for i, (timestamp, price) in enumerate(prices):
            volume = volumes[i][1] if i < len(volumes) else 0
            
            ohlcv_data.append({
                'timestamp': int(timestamp / 1000),  # Convert to seconds
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            })
        
        return ohlcv_data
    
    def _convert_binance_to_ohlcv(self, data: List) -> List[Dict]:
        """Convert Binance klines data to OHLCV format."""
        ohlcv_data = []
        
        for kline in data:
            ohlcv_data.append({
                'timestamp': int(kline[0] / 1000),  # Convert to seconds
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        
        return ohlcv_data
    
    async def store_data(self, response: APIResponse) -> bool:
        """Store collected data in the database."""
        if not response.success or not response.data:
            logger.error("Cannot store invalid data")
            return False
        
        try:
            stored_count = 0
            for candle in response.data:
                price_data = PriceData(
                    timestamp=candle['timestamp'],
                    open=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle['volume']
                )
                
                # Use upsert to handle duplicates
                if self.db_manager.upsert_price_data(price_data):
                    stored_count += 1
            
            logger.info(f"Stored {stored_count} price data points from {response.source.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing data: {str(e)}")
            return False
    
    def get_data_quality_report(self, data: List[Dict]) -> Dict:
        """Generate a data quality report."""
        if not data:
            return {'status': 'empty', 'issues': ['No data provided']}
        
        # Validate data
        is_valid, errors = self.validator.validate_ohlcv_data(data)
        
        # Detect outliers
        outliers = self.validator.detect_outliers(data, 'close')
        
        # Calculate completeness
        total_candles = len(data)
        complete_candles = sum(1 for candle in data 
                             if all(field in candle and candle[field] is not None 
                                   for field in ['open', 'high', 'low', 'close', 'volume']))
        
        completeness = (complete_candles / total_candles * 100) if total_candles > 0 else 0
        
        report = {
            'status': 'valid' if is_valid and len(outliers) < total_candles * 0.1 else 'issues',
            'total_candles': total_candles,
            'complete_candles': complete_candles,
            'completeness_pct': round(completeness, 2),
            'validation_errors': errors,
            'outlier_count': len(outliers),
            'outlier_indices': outliers[:10],  # Show first 10 outliers
            'time_range': {
                'start': datetime.fromtimestamp(min(c['timestamp'] for c in data)).isoformat(),
                'end': datetime.fromtimestamp(max(c['timestamp'] for c in data)).isoformat()
            } if data else None
        }
        
        return report
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("PriceDataCollector closed")

# Convenience functions for easy usage
async def collect_and_store_historical_data(config: ConfigManager, db_manager: DatabaseManager, days: int = None) -> bool:
    """Convenience function for historical data collection."""
    collector = PriceDataCollector(config, db_manager)
    try:
        response = await collector.collect_historical_data(days)
        if response.success:
            return await collector.store_data(response)
        return False
    finally:
        collector.close()

async def collect_and_store_current_data(config: ConfigManager, db_manager: DatabaseManager) -> bool:
    """Convenience function for current data collection."""
    collector = PriceDataCollector(config, db_manager)
    try:
        response = await collector.collect_current_data()
        if response.success:
            return await collector.store_data(response)
        return False
    finally:
        collector.close() 