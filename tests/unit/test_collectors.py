"""
Unit tests for the API data collection system.
Tests all components: PriceDataCollector, RateLimiter, DataValidator, and failover mechanisms.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import requests
from typing import Dict, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.collectors import (
    PriceDataCollector, RateLimiter, DataValidator, DataSource, 
    APIResponse, collect_and_store_historical_data, collect_and_store_current_data
)
from src.utils.config import ConfigManager
from src.data.storage import DatabaseManager, PriceData

class TestDataValidator:
    """Test the DataValidator class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = Mock(spec=ConfigManager)
        config.data.data_validation.max_price_change_pct = 20.0
        config.data.data_validation.min_volume_threshold = 1000
        return config
    
    @pytest.fixture
    def validator(self, config):
        """Create a DataValidator instance."""
        return DataValidator(config)
    
    def test_validate_empty_data(self, validator):
        """Test validation of empty data."""
        is_valid, errors = validator.validate_ohlcv_data([])
        assert not is_valid
        assert "Empty data set" in errors
    
    def test_validate_valid_ohlcv_data(self, validator):
        """Test validation of valid OHLCV data."""
        data = [
            {
                'timestamp': 1640995200,
                'open': 47000.0,
                'high': 47500.0,
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            },
            {
                'timestamp': 1640998800,
                'open': 47200.0,
                'high': 47800.0,
                'low': 47000.0,
                'close': 47600.0,
                'volume': 1800.0
            }
        ]
        
        is_valid, errors = validator.validate_ohlcv_data(data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_missing_fields(self, validator):
        """Test validation with missing fields."""
        data = [
            {
                'timestamp': 1640995200,
                'open': 47000.0,
                'high': 47500.0,
                # Missing 'low', 'close', 'volume'
            }
        ]
        
        is_valid, errors = validator.validate_ohlcv_data(data)
        assert not is_valid
        assert "Missing fields" in errors[0]
    
    def test_validate_invalid_ohlc_relationship(self, validator):
        """Test validation with invalid OHLC relationships."""
        data = [
            {
                'timestamp': 1640995200,
                'open': 47000.0,
                'high': 46000.0,  # High < Open (invalid)
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            }
        ]
        
        is_valid, errors = validator.validate_ohlcv_data(data)
        assert not is_valid
        assert "Invalid OHLC relationship" in errors[0]
    
    def test_validate_extreme_price_change(self, validator):
        """Test validation with extreme price changes."""
        data = [
            {
                'timestamp': 1640995200,
                'open': 47000.0,
                'high': 47500.0,
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            },
            {
                'timestamp': 1640998800,
                'open': 47200.0,
                'high': 60000.0,  # >20% change from previous close
                'low': 47000.0,
                'close': 58000.0,
                'volume': 1800.0
            }
        ]
        
        is_valid, errors = validator.validate_ohlcv_data(data)
        assert not is_valid
        assert "Extreme price change" in errors[0]
    
    def test_detect_outliers(self, validator):
        """Test outlier detection."""
        # Create data with one clear outlier
        data = []
        for i in range(20):
            price = 47000.0 + (i * 100)  # Normal progression
            data.append({
                'timestamp': 1640995200 + (i * 3600),
                'open': price,
                'high': price + 50,
                'low': price - 50,
                'close': price,
                'volume': 1500.0
            })
        
        # Add outlier
        data.append({
            'timestamp': 1640995200 + (20 * 3600),
            'open': 100000.0,  # Clear outlier
            'high': 100000.0,
            'low': 100000.0,
            'close': 100000.0,
            'volume': 1500.0
        })
        
        outliers = validator.detect_outliers(data, 'close')
        assert len(outliers) > 0
        assert 20 in outliers  # The outlier index

class TestRateLimiter:
    """Test the RateLimiter class."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a RateLimiter instance."""
        return RateLimiter()
    
    @pytest.mark.asyncio
    async def test_can_make_request_initial(self, rate_limiter):
        """Test initial request permission."""
        can_make = await rate_limiter.can_make_request(DataSource.COINGECKO, 100)
        assert can_make
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, rate_limiter):
        """Test rate limit enforcement."""
        # Record requests up to the limit
        for _ in range(99):
            await rate_limiter.record_request(DataSource.COINGECKO)
        
        # Should still be able to make one more
        can_make = await rate_limiter.can_make_request(DataSource.COINGECKO, 100)
        assert can_make
        
        # Record one more request
        await rate_limiter.record_request(DataSource.COINGECKO)
        
        # Now should be blocked
        can_make = await rate_limiter.can_make_request(DataSource.COINGECKO, 100)
        assert not can_make
    
    @pytest.mark.asyncio
    async def test_separate_source_limits(self, rate_limiter):
        """Test that different sources have separate limits."""
        # Fill up CoinGecko limit
        for _ in range(100):
            await rate_limiter.record_request(DataSource.COINGECKO)
        
        # CoinGecko should be blocked
        can_make_cg = await rate_limiter.can_make_request(DataSource.COINGECKO, 100)
        assert not can_make_cg
        
        # Binance should still be available
        can_make_binance = await rate_limiter.can_make_request(DataSource.BINANCE, 1200)
        assert can_make_binance

class TestPriceDataCollector:
    """Test the PriceDataCollector class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = Mock(spec=ConfigManager)
        config.data.historical_days = 180
        config.data.price_data_sources.primary = "coingecko"
        config.data.price_data_sources.backup = "binance"
        config.data.data_validation.max_price_change_pct = 20.0
        config.data.data_validation.min_volume_threshold = 1000
        
        # API configs
        config.api.coingecko.base_url = "https://api.coingecko.com/api/v3"
        config.api.coingecko.rate_limit_per_minute = 100
        config.api.coingecko.timeout_seconds = 30
        config.api.binance.base_url = "https://api.binance.com/api/v3"
        config.api.binance.rate_limit_per_minute = 1200
        config.api.binance.timeout_seconds = 30
        
        return config
    
    @pytest.fixture
    def db_manager(self):
        """Create a mock database manager."""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.upsert_price_data.return_value = True
        return db_manager
    
    @pytest.fixture
    def collector(self, config, db_manager):
        """Create a PriceDataCollector instance."""
        return PriceDataCollector(config, db_manager)
    
    def test_init(self, collector, config, db_manager):
        """Test collector initialization."""
        assert collector.config == config
        assert collector.db_manager == db_manager
        assert collector.rate_limiter is not None
        assert collector.validator is not None
        assert collector.session is not None
    
    @pytest.mark.asyncio
    async def test_collect_historical_data_success(self, collector):
        """Test successful historical data collection."""
        # Mock the internal fetch method
        mock_response = APIResponse(
            success=True,
            data=[{
                'timestamp': 1640995200,
                'open': 47000.0,
                'high': 47500.0,
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            }],
            source=DataSource.COINGECKO,
            timestamp=datetime.now()
        )
        
        with patch.object(collector, '_fetch_with_retry', return_value=mock_response):
            response = await collector.collect_historical_data(30)
            
            assert response.success
            assert response.data is not None
            assert len(response.data) == 1
            assert response.source == DataSource.COINGECKO
    
    @pytest.mark.asyncio
    async def test_collect_historical_data_failover(self, collector):
        """Test failover to backup source."""
        # Mock primary source failure
        primary_failure = APIResponse(success=False, error="API Error", source=DataSource.COINGECKO)
        
        # Mock backup source success
        backup_success = APIResponse(
            success=True,
            data=[{
                'timestamp': 1640995200,
                'open': 47000.0,
                'high': 47500.0,
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            }],
            source=DataSource.BINANCE,
            timestamp=datetime.now()
        )
        
        with patch.object(collector, '_fetch_with_retry', side_effect=[primary_failure, backup_success]):
            response = await collector.collect_historical_data(30)
            
            assert response.success
            assert response.source == DataSource.BINANCE
    
    @pytest.mark.asyncio
    async def test_collect_current_data_success(self, collector):
        """Test successful current data collection."""
        mock_response = APIResponse(
            success=True,
            data=[{
                'timestamp': int(datetime.now().timestamp()),
                'open': 47000.0,
                'high': 47000.0,
                'low': 47000.0,
                'close': 47000.0,
                'volume': 1500.0
            }],
            source=DataSource.COINGECKO,
            timestamp=datetime.now()
        )
        
        with patch.object(collector, '_fetch_current_with_retry', return_value=mock_response):
            response = await collector.collect_current_data()
            
            assert response.success
            assert response.data is not None
            assert len(response.data) == 1
    
    def test_convert_coingecko_to_ohlcv(self, collector):
        """Test CoinGecko data conversion."""
        coingecko_data = {
            'prices': [[1640995200000, 47000.0], [1640998800000, 47200.0]],
            'total_volumes': [[1640995200000, 1500.0], [1640998800000, 1800.0]]
        }
        
        ohlcv_data = collector._convert_coingecko_to_ohlcv(coingecko_data)
        
        assert len(ohlcv_data) == 2
        assert ohlcv_data[0]['timestamp'] == 1640995200
        assert ohlcv_data[0]['close'] == 47000.0
        assert ohlcv_data[0]['volume'] == 1500.0
    
    def test_convert_binance_to_ohlcv(self, collector):
        """Test Binance data conversion."""
        binance_data = [
            [1640995200000, "47000.0", "47500.0", "46500.0", "47200.0", "1500.0", 1640998800000, "71040000.0", 1234, "750.0", "35520000.0", "0"],
            [1640998800000, "47200.0", "47800.0", "47000.0", "47600.0", "1800.0", 1641002400000, "85680000.0", 1456, "900.0", "42840000.0", "0"]
        ]
        
        ohlcv_data = collector._convert_binance_to_ohlcv(binance_data)
        
        assert len(ohlcv_data) == 2
        assert ohlcv_data[0]['timestamp'] == 1640995200
        assert ohlcv_data[0]['open'] == 47000.0
        assert ohlcv_data[0]['high'] == 47500.0
        assert ohlcv_data[0]['low'] == 46500.0
        assert ohlcv_data[0]['close'] == 47200.0
        assert ohlcv_data[0]['volume'] == 1500.0
    
    @pytest.mark.asyncio
    async def test_store_data_success(self, collector, db_manager):
        """Test successful data storage."""
        response = APIResponse(
            success=True,
            data=[{
                'timestamp': 1640995200,
                'open': 47000.0,
                'high': 47500.0,
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            }],
            source=DataSource.COINGECKO
        )
        
        result = await collector.store_data(response)
        
        assert result
        db_manager.upsert_price_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_data_failure(self, collector, db_manager):
        """Test data storage with invalid response."""
        response = APIResponse(success=False, error="Test error")
        
        result = await collector.store_data(response)
        
        assert not result
        db_manager.upsert_price_data.assert_not_called()
    
    def test_get_data_quality_report(self, collector):
        """Test data quality report generation."""
        data = [
            {
                'timestamp': 1640995200,
                'open': 47000.0,
                'high': 47500.0,
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            },
            {
                'timestamp': 1640998800,
                'open': 47200.0,
                'high': 47800.0,
                'low': 47000.0,
                'close': 47600.0,
                'volume': 1800.0
            }
        ]
        
        report = collector.get_data_quality_report(data)
        
        assert report['status'] == 'valid'
        assert report['total_candles'] == 2
        assert report['complete_candles'] == 2
        assert report['completeness_pct'] == 100.0
        assert len(report['validation_errors']) == 0
        assert 'time_range' in report
    
    def test_get_data_quality_report_empty(self, collector):
        """Test data quality report for empty data."""
        report = collector.get_data_quality_report([])
        
        assert report['status'] == 'empty'
        assert 'No data provided' in report['issues']

class TestAPIIntegration:
    """Integration tests for API calls (mocked)."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = Mock(spec=ConfigManager)
        config.data.historical_days = 7
        config.data.price_data_sources.primary = "coingecko"
        config.data.price_data_sources.backup = "binance"
        config.data.data_validation.max_price_change_pct = 20.0
        config.data.data_validation.min_volume_threshold = 1000
        
        config.api.coingecko.base_url = "https://api.coingecko.com/api/v3"
        config.api.coingecko.rate_limit_per_minute = 100
        config.api.coingecko.timeout_seconds = 30
        config.api.binance.base_url = "https://api.binance.com/api/v3"
        config.api.binance.rate_limit_per_minute = 1200
        config.api.binance.timeout_seconds = 30
        
        return config
    
    @pytest.fixture
    def db_manager(self):
        """Create a mock database manager."""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.upsert_price_data.return_value = True
        return db_manager
    
    @pytest.mark.asyncio
    async def test_coingecko_historical_api_call(self, config, db_manager):
        """Test CoinGecko historical API call (mocked)."""
        collector = PriceDataCollector(config, db_manager)
        
        # Mock the requests.Session.get method
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'prices': [[1640995200000, 47000.0]],
            'total_volumes': [[1640995200000, 1500.0]]
        }
        
        with patch.object(collector.session, 'get', return_value=mock_response):
            response = await collector._fetch_coingecko_historical(7)
            
            assert response.success
            assert len(response.data) == 1
            assert response.source == DataSource.COINGECKO
    
    @pytest.mark.asyncio
    async def test_binance_historical_api_call(self, config, db_manager):
        """Test Binance historical API call (mocked)."""
        collector = PriceDataCollector(config, db_manager)
        
        # Mock the requests.Session.get method
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            [1640995200000, "47000.0", "47500.0", "46500.0", "47200.0", "1500.0", 1640998800000, "71040000.0", 1234, "750.0", "35520000.0", "0"]
        ]
        
        with patch.object(collector.session, 'get', return_value=mock_response):
            response = await collector._fetch_binance_historical(7)
            
            assert response.success
            assert len(response.data) == 1
            assert response.source == DataSource.BINANCE
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, config, db_manager):
        """Test API error handling."""
        collector = PriceDataCollector(config, db_manager)
        
        # Mock a request exception
        with patch.object(collector.session, 'get', side_effect=requests.exceptions.RequestException("Network error")):
            response = await collector._fetch_coingecko_historical(7)
            
            assert not response.success
            assert "Request error" in response.error

class TestConvenienceFunctions:
    """Test the convenience functions."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = Mock(spec=ConfigManager)
        config.data.historical_days = 7
        config.data.price_data_sources.primary = "coingecko"
        config.data.price_data_sources.backup = "binance"
        config.data.data_validation.max_price_change_pct = 20.0
        config.data.data_validation.min_volume_threshold = 1000
        
        config.api.coingecko.base_url = "https://api.coingecko.com/api/v3"
        config.api.coingecko.rate_limit_per_minute = 100
        config.api.coingecko.timeout_seconds = 30
        config.api.binance.base_url = "https://api.binance.com/api/v3"
        config.api.binance.rate_limit_per_minute = 1200
        config.api.binance.timeout_seconds = 30
        
        return config
    
    @pytest.fixture
    def db_manager(self):
        """Create a mock database manager."""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.upsert_price_data.return_value = True
        return db_manager
    
    @pytest.mark.asyncio
    async def test_collect_and_store_historical_data(self, config, db_manager):
        """Test the convenience function for historical data."""
        with patch('src.data.collectors.PriceDataCollector') as MockCollector:
            mock_instance = MockCollector.return_value
            mock_instance.collect_historical_data.return_value = APIResponse(
                success=True,
                data=[{'timestamp': 1640995200, 'open': 47000.0, 'high': 47500.0, 'low': 46500.0, 'close': 47200.0, 'volume': 1500.0}]
            )
            mock_instance.store_data.return_value = True
            mock_instance.close.return_value = None
            
            result = await collect_and_store_historical_data(config, db_manager, 30)
            
            assert result
            MockCollector.assert_called_once_with(config, db_manager)
            mock_instance.collect_historical_data.assert_called_once_with(30)
            mock_instance.store_data.assert_called_once()
            mock_instance.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collect_and_store_current_data(self, config, db_manager):
        """Test the convenience function for current data."""
        with patch('src.data.collectors.PriceDataCollector') as MockCollector:
            mock_instance = MockCollector.return_value
            mock_instance.collect_current_data.return_value = APIResponse(
                success=True,
                data=[{'timestamp': int(datetime.now().timestamp()), 'open': 47000.0, 'high': 47000.0, 'low': 47000.0, 'close': 47000.0, 'volume': 1500.0}]
            )
            mock_instance.store_data.return_value = True
            mock_instance.close.return_value = None
            
            result = await collect_and_store_current_data(config, db_manager)
            
            assert result
            MockCollector.assert_called_once_with(config, db_manager)
            mock_instance.collect_current_data.assert_called_once()
            mock_instance.store_data.assert_called_once()
            mock_instance.close.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 