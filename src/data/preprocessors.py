"""
Data preprocessing and feature engineering for price data.
Handles technical indicators, normalization, and sequence creation for LSTM input.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical indicator calculations for price data."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle division by zero
        loss = np.maximum(loss, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure RSI is within bounds
        rsi = np.clip(rsi, 0, 100)
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        # Handle edge case where std is 0 or very small
        std = std.fillna(0)
        std = np.maximum(std, 1e-10)  # Minimum std to avoid division by zero
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Ensure upper >= lower (handle numerical precision issues)
        upper_band = np.maximum(upper_band, lower_band)
        
        # Calculate width and position safely
        bb_width = (upper_band - lower_band) / np.maximum(sma, 1e-10)
        band_range = np.maximum(upper_band - lower_band, 1e-10)
        bb_position = (prices - lower_band) / band_range
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': bb_width,
            'bb_position': bb_position
        }
    
    @staticmethod
    def calculate_volume_indicators(prices: pd.Series, volume: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate volume-weighted indicators."""
        # Volume Weighted Average Price (VWAP)
        vwap = (prices * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # Volume Rate of Change
        volume_roc = volume.pct_change(periods=period)
        
        # On Balance Volume (OBV)
        obv = (volume * np.sign(prices.diff())).cumsum()
        
        return {
            'vwap': vwap,
            'volume_roc': volume_roc,
            'obv': obv,
            'volume_sma': volume.rolling(window=period).mean()
        }

class PriceDataPreprocessor:
    """Comprehensive price data preprocessing and feature engineering pipeline."""
    
    def __init__(self, sequence_length: int = 60, validation_split: float = 0.2):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Length of sequences for LSTM input
            validation_split: Fraction of data to use for validation
        """
        self.sequence_length = sequence_length
        self.validation_split = validation_split
        self.scalers = {}
        self.feature_columns = []
        self.indicators = TechnicalIndicators()
        
        logger.info(f"PriceDataPreprocessor initialized with sequence_length={sequence_length}")
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data."""
        logger.info(f"Cleaning data with {len(data)} rows")
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
            data = data[~invalid_ohlc].reset_index(drop=True)
        
        # Remove rows with zero or negative prices/volume
        invalid_values = (
            (data['open'] <= 0) |
            (data['high'] <= 0) |
            (data['low'] <= 0) |
            (data['close'] <= 0) |
            (data['volume'] < 0)
        )
        
        if invalid_values.any():
            logger.warning(f"Found {invalid_values.sum()} rows with invalid price/volume values")
            data = data[~invalid_values].reset_index(drop=True)
        
        # Fill any remaining NaN values
        data = data.ffill().bfill()
        
        # Final check - fill any remaining NaN with 0
        data = data.fillna(0)
        
        logger.info(f"Data cleaning completed. Final dataset: {len(data)} rows")
        return data
        
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators."""
        logger.info("Creating technical indicators")
        
        df = data.copy()
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # RSI
        df['rsi'] = self.indicators.calculate_rsi(df['close'])
        
        # MACD
        macd_data = self.indicators.calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.indicators.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_data['bb_upper']
        df['bb_middle'] = bb_data['bb_middle']
        df['bb_lower'] = bb_data['bb_lower']
        df['bb_width'] = bb_data['bb_width']
        df['bb_position'] = bb_data['bb_position']
        
        # Volume indicators
        volume_data = self.indicators.calculate_volume_indicators(df['close'], df['volume'])
        df['vwap'] = volume_data['vwap']
        df['volume_roc'] = volume_data['volume_roc']
        df['obv'] = volume_data['obv']
        df['volume_sma'] = volume_data['volume_sma']
        
        # Additional technical features
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Price ratios
        df['close_sma20_ratio'] = df['close'] / df['sma_20']
        df['volume_sma_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        # Fill any NaN values created during indicator calculation
        df = df.ffill().bfill().fillna(0)
        
        logger.info(f"Created {len(df.columns) - len(data.columns)} technical indicators")
        return df
        
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features and ratios."""
        logger.info("Creating price features")
        
        df = data.copy()
        
        # Returns at different periods
        for period in [1, 3, 5, 10]:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)
            df[f'volatility_{period}d'] = df['close'].rolling(window=period).std() / df['close']
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low ratios
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        # Gap analysis
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Fill any NaN values
        df = df.ffill().bfill().fillna(0)
        
        logger.info("Price features created")
        return df
        
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        logger.info("Creating volume features")
        
        df = data.copy()
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Price-volume relationship
        df['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        
        # Volume-weighted features
        df['vwap_ratio'] = df['close'] / df['vwap']
        
        # Fill any NaN values
        df = df.ffill().bfill().fillna(0)
        
        logger.info("Volume features created")
        return df
        
    def normalize_data(self, data: pd.DataFrame, fit_scalers: bool = True) -> pd.DataFrame:
        """Normalize features using min-max scaling."""
        logger.info(f"Normalizing data (fit_scalers={fit_scalers})")
        
        df = data.copy()
        
        # Define feature groups for different scaling approaches
        price_features = ['open', 'high', 'low', 'close', 'vwap', 'bb_upper', 'bb_middle', 'bb_lower',
                         'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26']
        
        volume_features = ['volume', 'obv', 'volume_sma']
        
        ratio_features = ['rsi', 'bb_position', 'close_sma20_ratio', 'volume_sma_ratio', 'vwap_ratio',
                         'hl_ratio', 'oc_ratio', 'bb_width', 'volatility_ratio']
        
        change_features = ['price_change', 'return_1d', 'return_3d', 'return_5d', 'return_10d',
                          'log_return', 'volume_change', 'volume_roc', 'gap', 'macd', 'macd_signal',
                          'macd_histogram', 'price_volume_trend']
        
        # Scale different feature groups
        feature_groups = {
            'price': price_features,
            'volume': volume_features,
            'ratio': ratio_features,
            'change': change_features
        }
        
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                continue
                
            if fit_scalers:
                scaler = MinMaxScaler(feature_range=(0, 1))
                df[available_features] = scaler.fit_transform(df[available_features].fillna(0))
                self.scalers[group_name] = scaler
            else:
                if group_name in self.scalers:
                    df[available_features] = self.scalers[group_name].transform(df[available_features].fillna(0))
        
        # Store feature columns for later use
        if fit_scalers:
            self.feature_columns = [col for col in df.columns if col not in ['timestamp']]
        
        logger.info(f"Normalization completed. Features: {len(self.feature_columns)}")
        return df
        
    def create_sequences(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        logger.info(f"Creating sequences with length {self.sequence_length}")
        
        # Select feature columns (exclude timestamp and target)
        feature_cols = [col for col in self.feature_columns if col != target_column and col != 'timestamp']
        
        # Prepare data
        features = data[feature_cols].values
        targets = data[target_column].values
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y
        
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        split_idx = int(len(X) * (1 - self.validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}")
        return X_train, X_val, y_train, y_val
        
    def validate_features(self, data: pd.DataFrame) -> Dict[str, any]:
        """Validate feature quality and completeness."""
        logger.info("Validating features")
        
        validation_report = {
            'total_features': len(self.feature_columns),
            'total_samples': len(data),
            'missing_values': {},
            'feature_ranges': {},
            'correlations': {},
            'quality_score': 0.0
        }
        
        # Check for missing values
        for col in self.feature_columns:
            if col in data.columns:
                missing_pct = data[col].isnull().sum() / len(data) * 100
                validation_report['missing_values'][col] = missing_pct
        
        # Check feature ranges
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.feature_columns:
                validation_report['feature_ranges'][col] = {
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std())
                }
        
        # Calculate quality score
        missing_score = 1.0 - (sum(validation_report['missing_values'].values()) / len(self.feature_columns) / 100)
        range_score = 1.0  # All features should be normalized to [0,1]
        
        validation_report['quality_score'] = (missing_score + range_score) / 2
        
        logger.info(f"Feature validation completed. Quality score: {validation_report['quality_score']:.3f}")
        return validation_report
        
    def process_pipeline(self, data: pd.DataFrame, fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Complete preprocessing pipeline."""
        logger.info("Starting complete preprocessing pipeline")
        
        # Step 1: Clean data
        cleaned_data = self.clean_data(data)
        
        # Step 2: Create technical indicators
        with_indicators = self.create_technical_indicators(cleaned_data)
        
        # Step 3: Create price features
        with_price_features = self.create_price_features(with_indicators)
        
        # Step 4: Create volume features
        with_volume_features = self.create_volume_features(with_price_features)
        
        # Step 5: Normalize data
        normalized_data = self.normalize_data(with_volume_features, fit_scalers=fit_scalers)
        
        # Step 6: Create sequences
        X, y = self.create_sequences(normalized_data)
        
        # Step 7: Validate features
        validation_report = self.validate_features(normalized_data)
        
        logger.info("Preprocessing pipeline completed successfully")
        return X, y, validation_report
        
    def get_feature_importance(self, data: pd.DataFrame, target_column: str = 'close') -> Dict[str, float]:
        """Calculate feature importance using correlation analysis."""
        logger.info("Calculating feature importance")
        
        correlations = {}
        for col in self.feature_columns:
            if col != target_column and col in data.columns:
                corr = abs(data[col].corr(data[target_column]))
                if not np.isnan(corr):
                    correlations[col] = corr
        
        # Sort by importance
        sorted_correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Feature importance calculated for {len(sorted_correlations)} features")
        return sorted_correlations
        
    def save_scalers(self, filepath: str):
        """Save fitted scalers for later use."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length
            }, f)
        logger.info(f"Scalers saved to {filepath}")
        
    def load_scalers(self, filepath: str):
        """Load previously fitted scalers."""
        import pickle
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.scalers = saved_data['scalers']
            self.feature_columns = saved_data['feature_columns']
            self.sequence_length = saved_data['sequence_length']
        logger.info(f"Scalers loaded from {filepath}") 