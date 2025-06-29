"""
Unit tests for the feature engineering and preprocessing pipeline.
Tests all technical indicators, normalization, and sequence creation functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.data.preprocessors import PriceDataPreprocessor, TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicator calculations."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        self.indicators = TechnicalIndicators()
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        rsi = self.indicators.calculate_rsi(self.sample_data['close'])
        
        # RSI should be between 0 and 100
        self.assertTrue((rsi.dropna() >= 0).all())
        self.assertTrue((rsi.dropna() <= 100).all())
        
        # Should have NaN values for initial period
        self.assertTrue(rsi.iloc[:13].isna().all())
        
        # Should have valid values after initial period
        self.assertFalse(rsi.iloc[14:].isna().any())
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        macd_data = self.indicators.calculate_macd(self.sample_data['close'])
        
        # Should return dictionary with required keys
        required_keys = ['macd', 'signal', 'histogram']
        self.assertEqual(set(macd_data.keys()), set(required_keys))
        
        # All values should be pandas Series
        for key in required_keys:
            self.assertIsInstance(macd_data[key], pd.Series)
        
        # Histogram should equal MACD - Signal
        histogram_calc = macd_data['macd'] - macd_data['signal']
        pd.testing.assert_series_equal(
            macd_data['histogram'].dropna(),
            histogram_calc.dropna(),
            check_names=False
        )
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb_data = self.indicators.calculate_bollinger_bands(self.sample_data['close'])
        
        # Should return dictionary with required keys
        required_keys = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position']
        self.assertEqual(set(bb_data.keys()), set(required_keys))
        
        # All values should be pandas Series
        for key in required_keys:
            self.assertIsInstance(bb_data[key], pd.Series)
        
        # Middle band should be the SMA
        sma_20 = self.sample_data['close'].rolling(window=20).mean()
        pd.testing.assert_series_equal(
            bb_data['bb_middle'].dropna(),
            sma_20.dropna(),
            check_names=False
        )
        
        # BB width should be non-negative
        self.assertTrue((bb_data['bb_width'].dropna() >= 0).all())
        
        # BB position should be finite (allowing any range due to outliers)
        self.assertTrue(np.isfinite(bb_data['bb_position'].dropna()).all())
    
    def test_calculate_volume_indicators(self):
        """Test volume indicator calculations."""
        volume_data = self.indicators.calculate_volume_indicators(
            self.sample_data['close'], 
            self.sample_data['volume']
        )
        
        # Should return dictionary with required keys
        required_keys = ['vwap', 'volume_roc', 'obv', 'volume_sma']
        self.assertEqual(set(volume_data.keys()), set(required_keys))
        
        # All values should be pandas Series
        for key in required_keys:
            self.assertIsInstance(volume_data[key], pd.Series)
        
        # VWAP should be reasonable relative to prices
        vwap_valid = volume_data['vwap'].dropna()
        price_valid = self.sample_data['close'].iloc[len(self.sample_data) - len(vwap_valid):]
        
        # VWAP should be within reasonable range of prices
        price_ratio = vwap_valid / price_valid
        self.assertTrue((price_ratio > 0.5).all())
        self.assertTrue((price_ratio < 2.0).all())

class TestPriceDataPreprocessor(unittest.TestCase):
    """Test the main preprocessing pipeline."""
    
    def setUp(self):
        """Set up test data and preprocessor."""
        # Create comprehensive test data
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
        
        # Generate realistic OHLCV data
        base_price = 100
        returns = np.random.normal(0, 0.02, n_samples)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC from close prices
        opens = [prices[0]] + prices[:-1]  # Previous close as next open
        highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.01))) for o, c in zip(opens, prices)]
        lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.01))) for o, c in zip(opens, prices)]
        volumes = np.random.randint(1000, 10000, n_samples)
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        self.preprocessor = PriceDataPreprocessor(sequence_length=20, validation_split=0.2)
    
    def test_init(self):
        """Test preprocessor initialization."""
        self.assertEqual(self.preprocessor.sequence_length, 20)
        self.assertEqual(self.preprocessor.validation_split, 0.2)
        self.assertEqual(len(self.preprocessor.scalers), 0)
        self.assertEqual(len(self.preprocessor.feature_columns), 0)
        self.assertIsInstance(self.preprocessor.indicators, TechnicalIndicators)
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Test with valid data
        cleaned = self.preprocessor.clean_data(self.sample_data.copy())
        self.assertEqual(len(cleaned), len(self.sample_data))
        
        # Test with missing columns
        invalid_data = self.sample_data.drop('volume', axis=1)
        with self.assertRaises(ValueError):
            self.preprocessor.clean_data(invalid_data)
        
        # Test with invalid OHLC relationships
        invalid_ohlc = self.sample_data.copy()
        invalid_ohlc.loc[0, 'high'] = invalid_ohlc.loc[0, 'low'] - 1  # High < Low
        cleaned = self.preprocessor.clean_data(invalid_ohlc)
        self.assertEqual(len(cleaned), len(self.sample_data) - 1)
        
        # Test with negative prices
        invalid_prices = self.sample_data.copy()
        invalid_prices.loc[0, 'close'] = -1
        cleaned = self.preprocessor.clean_data(invalid_prices)
        self.assertEqual(len(cleaned), len(self.sample_data) - 1)
    
    def test_create_technical_indicators(self):
        """Test technical indicator creation."""
        with_indicators = self.preprocessor.create_technical_indicators(self.sample_data)
        
        # Should have more columns than original
        self.assertGreater(len(with_indicators.columns), len(self.sample_data.columns))
        
        # Check for specific indicators
        expected_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'vwap', 'volume_roc', 'obv', 'volume_sma',
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, with_indicators.columns)
        
        # Check that indicators have reasonable values
        self.assertTrue((with_indicators['rsi'].dropna() >= 0).all())
        self.assertTrue((with_indicators['rsi'].dropna() <= 100).all())
        self.assertTrue((with_indicators['bb_upper'] >= with_indicators['bb_lower']).dropna().all())
    
    def test_process_pipeline(self):
        """Test complete preprocessing pipeline."""
        X, y, validation_report = self.preprocessor.process_pipeline(self.sample_data)
        
        # Check outputs
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(validation_report, dict)
        
        # Check shapes
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], self.preprocessor.sequence_length)
        self.assertGreater(X.shape[2], 10)  # Should have many features
        
        # Check that scalers are fitted
        self.assertGreater(len(self.preprocessor.scalers), 0)
        self.assertGreater(len(self.preprocessor.feature_columns), 0)
        
        # Check validation report
        self.assertIn('quality_score', validation_report)
        self.assertGreater(validation_report['quality_score'], 0.5)

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Run tests
    unittest.main(verbosity=2) 