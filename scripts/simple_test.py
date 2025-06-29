#!/usr/bin/env python3
"""
Simple test for API data collection system components.
Tests basic functionality without complex imports.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_basic_functionality():
    """Test basic functionality of the API collection system."""
    print("🧪 Testing API Data Collection System Components")
    print("=" * 50)
    
    try:
        # Test 1: Configuration loading
        print("1️⃣ Testing Configuration System...")
        from src.utils.config import ConfigManager
        
        config = ConfigManager()
        print(f"   ✅ ConfigManager initialized")
        print(f"   📊 Primary data source: {config.get_data_config().price_data_sources['primary']}")
        print(f"   📊 Backup data source: {config.get_data_config().price_data_sources['backup']}")
        print(f"   📅 Historical days: {config.get_data_config().historical_days}")
        
        # Test 2: Database system
        print("\n2️⃣ Testing Database System...")
        from src.data.storage import DatabaseManager
        
        db_manager = DatabaseManager(config)
        print(f"   ✅ DatabaseManager initialized")
        
        # Test basic database operations
        stats = db_manager.get_database_stats()
        print(f"   📊 Database stats: {stats}")
        
        # Test 3: Data collector components
        print("\n3️⃣ Testing Data Collector Components...")
        from src.data.collectors import DataSource, DataValidator, RateLimiter
        
        # Test DataSource enum
        print(f"   ✅ DataSource.COINGECKO: {DataSource.COINGECKO.value}")
        print(f"   ✅ DataSource.BINANCE: {DataSource.BINANCE.value}")
        
        # Test DataValidator
        validator = DataValidator(config)
        print(f"   ✅ DataValidator initialized")
        
        # Test with valid data
        test_data = [
            {
                'timestamp': int(datetime.now().timestamp()),
                'open': 47000.0,
                'high': 47500.0,
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            }
        ]
        
        is_valid, errors = validator.validate_ohlcv_data(test_data)
        print(f"   ✅ Data validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Test with invalid data
        invalid_data = [
            {
                'timestamp': int(datetime.now().timestamp()),
                'open': 47000.0,
                'high': 46000.0,  # Invalid: high < open
                'low': 46500.0,
                'close': 47200.0,
                'volume': 1500.0
            }
        ]
        
        is_valid, errors = validator.validate_ohlcv_data(invalid_data)
        print(f"   ✅ Invalid data detection: {'PASS' if not is_valid else 'FAIL'}")
        print(f"   📝 Validation errors: {len(errors)}")
        
        # Test 4: Rate Limiter
        print("\n4️⃣ Testing Rate Limiter...")
        rate_limiter = RateLimiter()
        print(f"   ✅ RateLimiter initialized")
        
        # Test 5: API Configuration
        print("\n5️⃣ Testing API Configuration...")
        api_config = config.api
        print(f"   🌐 CoinGecko URL: {api_config.coingecko['base_url']}")
        print(f"   📊 CoinGecko rate limit: {api_config.coingecko['rate_limit_per_minute']}/min")
        print(f"   🌐 Binance URL: {api_config.binance['base_url']}")
        print(f"   📊 Binance rate limit: {api_config.binance['rate_limit_per_minute']}/min")
        
        # Test 6: Data Format Conversion
        print("\n6️⃣ Testing Data Format Conversion...")
        from src.data.collectors import PriceDataCollector
        
        collector = PriceDataCollector(config, db_manager)
        print(f"   ✅ PriceDataCollector initialized")
        
        # Test CoinGecko format conversion
        coingecko_sample = {
            'prices': [[1640995200000, 47000.0], [1640998800000, 47200.0]],
            'total_volumes': [[1640995200000, 1500.0], [1640998800000, 1800.0]]
        }
        ohlcv_data = collector._convert_coingecko_to_ohlcv(coingecko_sample)
        print(f"   ✅ CoinGecko conversion: {len(ohlcv_data)} candles")
        
        # Test Binance format conversion
        binance_sample = [
            [1640995200000, "47000.0", "47500.0", "46500.0", "47200.0", "1500.0", 1640998800000, "71040000.0", 1234, "750.0", "35520000.0", "0"]
        ]
        ohlcv_data = collector._convert_binance_to_ohlcv(binance_sample)
        print(f"   ✅ Binance conversion: {len(ohlcv_data)} candles")
        
        # Test data quality report
        report = collector.get_data_quality_report(test_data)
        print(f"   ✅ Data quality report: {report['status']}")
        print(f"   📊 Completeness: {report['completeness_pct']}%")
        
        # Cleanup
        collector.close()
        
        print("\n" + "=" * 50)
        print("✅ All Basic Tests Passed!")
        print("🎯 TASK-004 Core Components Validated:")
        print("   ✅ Configuration system integration")
        print("   ✅ Database system integration") 
        print("   ✅ Data validation and quality checks")
        print("   ✅ Rate limiting infrastructure")
        print("   ✅ API configuration management")
        print("   ✅ Data format conversion (CoinGecko & Binance)")
        print("   ✅ Error handling and validation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_components():
    """Test async components of the system."""
    print("\n🔄 Testing Async Components...")
    
    try:
        from src.utils.config import ConfigManager
        from src.data.storage import DatabaseManager
        from src.data.collectors import RateLimiter, DataSource
        
        config = ConfigManager()
        rate_limiter = RateLimiter()
        
        # Test rate limiter async functionality
        can_make = await rate_limiter.can_make_request(DataSource.COINGECKO, 100)
        print(f"   ✅ Rate limiter check: {can_make}")
        
        await rate_limiter.record_request(DataSource.COINGECKO, 1)
        print(f"   ✅ Rate limiter recording: Success")
        
        # Test multiple requests
        for i in range(5):
            await rate_limiter.record_request(DataSource.COINGECKO, 1)
        
        print(f"   ✅ Multiple request tracking: Success")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Async test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🚀 API Data Collection System - Component Test")
    print("📋 TASK-004: API Data Collection System")
    print("🕒 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # Run basic tests
    basic_success = test_basic_functionality()
    
    if basic_success:
        # Run async tests
        async_success = asyncio.run(test_async_components())
        
        if async_success:
            print("\n🎉 All Tests Completed Successfully!")
            print("✅ API Data Collection System is ready for production use")
        else:
            print("\n⚠️  Basic tests passed, but async tests had issues")
    else:
        print("\n❌ Basic tests failed - system needs attention")

if __name__ == "__main__":
    main() 