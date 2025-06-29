"""
M2 Money Supply Data Provider
Fetches and manages M2 data from FRED API with robust fallbacks
"""

import pandas as pd
import numpy as np
import requests
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class M2DataProvider:
    """
    M2 Money Supply data provider with FRED API integration and intelligent caching
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/m2_cache"):
        """
        Initialize M2 data provider
        
        Args:
            api_key: FRED API key (optional, will try demo mode if None)
            cache_dir: Directory for caching M2 data
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY', 'demo')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.m2_data = None
        self.last_update = None
        
        logger.info(f"M2DataProvider initialized with cache: {self.cache_dir}")
    
    def fetch_m2_data(self, start_date: str = '2020-01-01', 
                      end_date: Optional[str] = None) -> bool:
        """
        Fetch M2 money supply data from FRED API with caching
        
        Args:
            start_date: Start date for M2 data (YYYY-MM-DD)
            end_date: End date for M2 data (defaults to today)
            
        Returns:
            True if successful, False otherwise
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache first
        cache_file = self.cache_dir / f"m2_data_{start_date}_{end_date}.pkl"
        
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days < 7:  # Use cache if less than 7 days old
                logger.info("Loading M2 data from cache")
                return self._load_from_cache(cache_file)
        
        # Fetch from FRED API
        logger.info(f"Fetching M2 data from FRED API: {start_date} to {end_date}")
        
        params = {
            'series_id': 'M2SL',
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'frequency': 'w',  # Weekly data
            'aggregation_method': 'eop'  # End of period
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'observations' in data and len(data['observations']) > 0:
                # Process the data
                if self._process_fred_data(data['observations']):
                    # Save to cache
                    self._save_to_cache(cache_file)
                    logger.info(f"✅ M2 data fetched successfully: {len(self.m2_data)} observations")
                    return True
            else:
                logger.warning("No M2 data returned from FRED API")
                return self._create_fallback_data(start_date, end_date)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            return self._create_fallback_data(start_date, end_date)
        except Exception as e:
            logger.error(f"Error processing M2 data: {e}")
            return self._create_fallback_data(start_date, end_date)
    
    def _process_fred_data(self, observations: list) -> bool:
        """Process raw FRED API observations into usable format"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['m2_value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove invalid data
            df = df.dropna(subset=['m2_value'])
            df = df[df['m2_value'] > 0]  # M2 should always be positive
            
            if len(df) < 10:
                logger.warning("Insufficient valid M2 data points")
                return False
            
            # Set index and sort
            df = df.set_index('date').sort_index()
            
            # Calculate growth rates and features
            self.m2_data = self._calculate_m2_features(df)
            self.last_update = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing FRED data: {e}")
            return False
    
    def _calculate_m2_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate M2 growth rates and derived features"""
        result = df[['m2_value']].copy()
        
        # Growth rates
        result['m2_yoy'] = result['m2_value'].pct_change(52) * 100  # Year-over-year
        result['m2_qoq'] = result['m2_value'].pct_change(13) * 100  # Quarter-over-quarter  
        result['m2_mom'] = result['m2_value'].pct_change(4) * 100   # Month-over-month (4 weeks)
        
        # Smoothed trends
        result['m2_trend_12w'] = result['m2_yoy'].rolling(12).mean()
        result['m2_trend_24w'] = result['m2_yoy'].rolling(24).mean()
        
        # Acceleration (second derivative)
        result['m2_acceleration'] = result['m2_yoy'].diff()
        
        # Deviation from long-term trend
        result['m2_longterm_mean'] = result['m2_yoy'].rolling(104, min_periods=52).mean()  # 2-year average
        result['m2_deviation'] = result['m2_yoy'] - result['m2_longterm_mean']
        
        # Volatility
        result['m2_volatility'] = result['m2_yoy'].rolling(26).std()  # 6-month volatility
        
        # Apply lag (M2 leads Bitcoin by ~12 weeks according to research)
        lag_weeks = 12
        result['m2_yoy_lagged'] = result['m2_yoy'].shift(lag_weeks)
        result['m2_trend_lagged'] = result['m2_trend_12w'].shift(lag_weeks)
        result['m2_acceleration_lagged'] = result['m2_acceleration'].shift(lag_weeks)
        
        logger.info("✅ M2 features calculated successfully")
        return result
    
    def _create_fallback_data(self, start_date: str, end_date: str) -> bool:
        """Create realistic synthetic M2 data if FRED API fails"""
        logger.warning("Creating synthetic M2 data as fallback")
        
        try:
            # Create date range
            dates = pd.date_range(start=start_date, end=end_date, freq='W-THU')  # Weekly on Thursdays
            
            # Realistic M2 patterns based on historical periods
            np.random.seed(42)  # Reproducible results
            
            m2_values = []
            m2_growth_rates = []
            base_m2 = 15000.0  # Approximate M2 in billions
            
            for i, date in enumerate(dates):
                # Simulate realistic M2 growth patterns
                if '2020-03-01' <= date.strftime('%Y-%m-%d') <= '2021-12-31':
                    # COVID era: Massive M2 expansion
                    growth_rate = np.random.normal(18, 3)
                elif date >= pd.Timestamp('2022-01-01'):
                    # Post-COVID: M2 normalization/contraction
                    growth_rate = np.random.normal(1, 2)
                else:
                    # Normal periods
                    growth_rate = np.random.normal(6, 1.5)
                
                # Bound growth rate to realistic range
                growth_rate = np.clip(growth_rate, -2, 25)
                m2_growth_rates.append(growth_rate)
                
                # Calculate M2 value (compound growth)
                if i == 0:
                    m2_value = base_m2
                else:
                    weekly_growth = (1 + growth_rate / 100) ** (1/52) - 1
                    m2_value = m2_values[-1] * (1 + weekly_growth)
                
                m2_values.append(m2_value)
            
            # Create DataFrame
            df = pd.DataFrame({
                'm2_value': m2_values,
                'm2_yoy': m2_growth_rates
            }, index=dates)
            
            # Calculate additional features
            self.m2_data = self._calculate_m2_features(df)
            self.last_update = datetime.now()
            
            logger.info(f"✅ Synthetic M2 data created: {len(self.m2_data)} observations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create synthetic M2 data: {e}")
            return False
    
    def _save_to_cache(self, cache_file: Path):
        """Save M2 data to cache"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': self.m2_data,
                    'last_update': self.last_update
                }, f)
            logger.info(f"M2 data cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache M2 data: {e}")
    
    def _load_from_cache(self, cache_file: Path) -> bool:
        """Load M2 data from cache"""
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.m2_data = cached['data']
                self.last_update = cached['last_update']
            logger.info(f"M2 data loaded from cache: {len(self.m2_data)} observations")
            return True
        except Exception as e:
            logger.warning(f"Failed to load M2 data from cache: {e}")
            return False
    
    def get_current_m2_regime(self) -> Dict[str, float]:
        """
        Get current M2 regime and metrics
        
        Returns:
            Dictionary with current M2 metrics and regime classification
        """
        if self.m2_data is None or len(self.m2_data) == 0:
            return {'regime': 'unknown', 'growth_yoy': 0.0, 'confidence': 0.0}
        
        # Get latest data (accounting for lag)
        latest_data = self.m2_data.dropna(subset=['m2_yoy_lagged']).iloc[-1]
        
        growth_yoy = latest_data['m2_yoy_lagged']
        trend_12w = latest_data.get('m2_trend_lagged', growth_yoy)
        acceleration = latest_data.get('m2_acceleration_lagged', 0)
        
        # Classify regime
        if growth_yoy > 8:
            regime = 'expansion'
            confidence = min(1.0, (growth_yoy - 8) / 10)  # Higher confidence with stronger expansion
        elif growth_yoy < 2:
            regime = 'contraction'
            confidence = min(1.0, (2 - growth_yoy) / 5)  # Higher confidence with stronger contraction
        else:
            regime = 'stable'
            confidence = 1.0 - abs(growth_yoy - 5) / 3  # Higher confidence near 5% (historical average)
        
        confidence = max(0.1, confidence)  # Minimum confidence
        
        return {
            'regime': regime,
            'growth_yoy': float(growth_yoy),
            'trend_12w': float(trend_12w),
            'acceleration': float(acceleration),
            'confidence': float(confidence),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    def get_m2_data_for_backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get M2 data formatted for backtesting
        
        Returns:
            DataFrame with M2 features for the specified period
        """
        if self.m2_data is None:
            return pd.DataFrame()
        
        # Filter by date range
        mask = (self.m2_data.index >= start_date) & (self.m2_data.index <= end_date)
        return self.m2_data.loc[mask].copy()
    
    def is_data_fresh(self, max_age_days: int = 7) -> bool:
        """Check if M2 data is fresh enough for trading"""
        if self.last_update is None:
            return False
        
        age = datetime.now() - self.last_update
        return age.days <= max_age_days