"""
Database storage operations for price data, predictions, and trades.
Handles SQLite database operations and schema management.
"""

import sqlite3
import logging
import json
import shutil
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Price data structure."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class PredictionData:
    """Prediction data structure."""
    timestamp: int
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    model_version: str

@dataclass
class TradeData:
    """Trade data structure."""
    entry_timestamp: int
    entry_price: float
    position_size: float
    position_type: str  # 'long' or 'short'
    confidence_score: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_timestamp: Optional[int] = None
    exit_price: Optional[float] = None
    status: str = 'open'  # 'open', 'closed', 'stopped'
    pnl: Optional[float] = None

@dataclass
class ModelMetrics:
    """Model performance metrics structure."""
    timestamp: int
    mse: float
    mae: float
    directional_accuracy: float
    model_version: str
    sharpe_ratio: Optional[float] = None

@dataclass
class PerformanceSnapshot:
    """Performance snapshot data structure."""
    timestamp: int
    total_return_pct: float
    win_rate_pct: float
    avg_confidence: float
    avg_uncertainty_pct: float
    trades_per_day: float
    max_drawdown_pct: float
    portfolio_value: float
    active_positions: int
    sharpe_ratio: Optional[float] = None

@dataclass
class ParameterAdjustment:
    """Parameter adjustment record structure."""
    timestamp: int
    parameter_name: str
    old_value: float
    new_value: float
    adjustment_reason: str
    trigger_metric: str
    trigger_value: float
    performance_before: Optional[str] = None

@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    timestamp: int
    alert_type: str
    severity: str
    message: str
    threshold_value: float
    actual_value: float
    acknowledged: bool = False
    acknowledged_at: Optional[int] = None
    resolved: bool = False
    resolved_at: Optional[int] = None

class DatabaseManager:
    """Manages SQLite database operations with full CRUD support, validation, and backup."""
    
    # SQL Schema definitions from PRD Section 4.1.2
    SCHEMA = {
        'btc_prices': '''
            CREATE TABLE IF NOT EXISTS btc_prices (
                timestamp INTEGER PRIMARY KEY,
                open REAL NOT NULL CHECK(open > 0),
                high REAL NOT NULL CHECK(high > 0),
                low REAL NOT NULL CHECK(low > 0),
                close REAL NOT NULL CHECK(close > 0),
                volume REAL NOT NULL CHECK(volume >= 0),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CHECK(high >= low),
                CHECK(high >= open),
                CHECK(high >= close),
                CHECK(low <= open),
                CHECK(low <= close)
            )
        ''',
        'predictions': '''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                predicted_price REAL NOT NULL CHECK(predicted_price > 0),
                confidence_lower REAL NOT NULL CHECK(confidence_lower > 0),
                confidence_upper REAL NOT NULL CHECK(confidence_upper > 0),
                model_version TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CHECK(confidence_upper >= confidence_lower),
                CHECK(predicted_price >= confidence_lower),
                CHECK(predicted_price <= confidence_upper)
            )
        ''',
        'trades': '''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_timestamp INTEGER NOT NULL,
                exit_timestamp INTEGER,
                entry_price REAL NOT NULL CHECK(entry_price > 0),
                exit_price REAL CHECK(exit_price IS NULL OR exit_price > 0),
                position_size REAL NOT NULL CHECK(position_size > 0),
                position_type TEXT NOT NULL CHECK(position_type IN ('long', 'short')),
                confidence_score REAL NOT NULL CHECK(confidence_score >= 0 AND confidence_score <= 1),
                stop_loss REAL CHECK(stop_loss IS NULL OR stop_loss > 0),
                take_profit REAL CHECK(take_profit IS NULL OR take_profit > 0),
                status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed', 'stopped')),
                pnl REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                CHECK(exit_timestamp IS NULL OR exit_timestamp >= entry_timestamp)
            )
        ''',
        'model_metrics': '''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                mse REAL NOT NULL CHECK(mse >= 0),
                mae REAL NOT NULL CHECK(mae >= 0),
                directional_accuracy REAL NOT NULL CHECK(directional_accuracy >= 0 AND directional_accuracy <= 1),
                sharpe_ratio REAL,
                model_version TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        'performance_snapshots': '''
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                total_return_pct REAL NOT NULL,
                win_rate_pct REAL NOT NULL CHECK(win_rate_pct >= 0 AND win_rate_pct <= 100),
                avg_confidence REAL NOT NULL CHECK(avg_confidence >= 0 AND avg_confidence <= 1),
                avg_uncertainty_pct REAL NOT NULL CHECK(avg_uncertainty_pct >= 0),
                trades_per_day REAL NOT NULL CHECK(trades_per_day >= 0),
                max_drawdown_pct REAL NOT NULL CHECK(max_drawdown_pct >= 0),
                sharpe_ratio REAL,
                portfolio_value REAL NOT NULL CHECK(portfolio_value > 0),
                active_positions INTEGER NOT NULL CHECK(active_positions >= 0),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        'parameter_adjustments': '''
            CREATE TABLE IF NOT EXISTS parameter_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                parameter_name TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                adjustment_reason TEXT NOT NULL,
                trigger_metric TEXT NOT NULL,
                trigger_value REAL NOT NULL,
                performance_before TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        'performance_alerts': '''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                alert_type TEXT NOT NULL CHECK(alert_type IN ('win_rate', 'drawdown', 'confidence', 'uncertainty', 'trading_frequency', 'overtrading')),
                severity TEXT NOT NULL CHECK(severity IN ('low', 'medium', 'high', 'critical')),
                message TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                actual_value REAL NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_at INTEGER,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        '''
    }
    
    # Performance optimization indexes
    INDEXES = {
        'idx_btc_prices_timestamp': 'CREATE INDEX IF NOT EXISTS idx_btc_prices_timestamp ON btc_prices(timestamp DESC)',
        'idx_predictions_timestamp': 'CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC)',
        'idx_predictions_model': 'CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version, timestamp DESC)',
        'idx_trades_entry_timestamp': 'CREATE INDEX IF NOT EXISTS idx_trades_entry_timestamp ON trades(entry_timestamp DESC)',
        'idx_trades_status': 'CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status, entry_timestamp DESC)',
        'idx_model_metrics_timestamp': 'CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp DESC)',
        'idx_model_metrics_version': 'CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON model_metrics(model_version, timestamp DESC)',
        # New indexes for performance tracking tables
        'idx_performance_snapshots_timestamp': 'CREATE INDEX IF NOT EXISTS idx_performance_snapshots_timestamp ON performance_snapshots(timestamp DESC)',
        'idx_parameter_adjustments_timestamp': 'CREATE INDEX IF NOT EXISTS idx_parameter_adjustments_timestamp ON parameter_adjustments(timestamp DESC)',
        'idx_parameter_adjustments_name': 'CREATE INDEX IF NOT EXISTS idx_parameter_adjustments_name ON parameter_adjustments(parameter_name, timestamp DESC)',
        'idx_performance_alerts_timestamp': 'CREATE INDEX IF NOT EXISTS idx_performance_alerts_timestamp ON performance_alerts(timestamp DESC)',
        'idx_performance_alerts_type': 'CREATE INDEX IF NOT EXISTS idx_performance_alerts_type ON performance_alerts(alert_type, timestamp DESC)',
        'idx_performance_alerts_severity': 'CREATE INDEX IF NOT EXISTS idx_performance_alerts_severity ON performance_alerts(severity, acknowledged, timestamp DESC)'
    }
    
    def __init__(self, db_path: str = "data/trading_bot.db", backup_path: str = "data/backups/"):
        self.db_path = Path(db_path)
        self.backup_path = Path(backup_path)
        self._lock = threading.Lock()
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
        logger.info(f"DatabaseManager initialized with database at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling and cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
            conn.execute('PRAGMA journal_mode = WAL')  # Enable WAL mode for better concurrency
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database with required tables and indexes."""
        try:
            with self._lock:
                with self.get_connection() as conn:
                    # Create all tables
                    for table_name, schema in self.SCHEMA.items():
                        conn.execute(schema)
                        logger.debug(f"Created/verified table: {table_name}")
                    
                    # Create all indexes
                    for index_name, index_sql in self.INDEXES.items():
                        conn.execute(index_sql)
                        logger.debug(f"Created/verified index: {index_name}")
                    
                    conn.commit()
                    logger.info("Database schema initialization completed successfully")
                    
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def validate_price_data(self, data: Union[PriceData, Dict]) -> bool:
        """Validate price data before insertion."""
        if isinstance(data, dict):
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(field in data for field in required_fields):
                return False
            
            # Validate OHLC relationships
            if not (data['high'] >= data['low'] and 
                   data['high'] >= data['open'] and 
                   data['high'] >= data['close'] and
                   data['low'] <= data['open'] and 
                   data['low'] <= data['close']):
                return False
            
            # Validate positive values
            if not all(data[field] > 0 for field in ['open', 'high', 'low', 'close']) or data['volume'] < 0:
                return False
        
        return True
    
    def store_price_data(self, data: Union[List[Dict], List[PriceData], Dict, PriceData]) -> bool:
        """Store price data in database with validation."""
        try:
            # Normalize input to list of dicts
            if isinstance(data, (dict, PriceData)):
                data = [data]
            
            price_records = []
            for item in data:
                if isinstance(item, PriceData):
                    record = {
                        'timestamp': item.timestamp,
                        'open': item.open,
                        'high': item.high,
                        'low': item.low,
                        'close': item.close,
                        'volume': item.volume
                    }
                else:
                    record = item
                
                if not self.validate_price_data(record):
                    logger.warning(f"Invalid price data skipped: {record}")
                    continue
                
                price_records.append(record)
            
            if not price_records:
                logger.warning("No valid price data to store")
                return False
            
            with self._lock:
                with self.get_connection() as conn:
                    conn.executemany('''
                        INSERT OR REPLACE INTO btc_prices 
                        (timestamp, open, high, low, close, volume)
                        VALUES (:timestamp, :open, :high, :low, :close, :volume)
                    ''', price_records)
                    conn.commit()
                    
            logger.info(f"Stored {len(price_records)} price records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store price data: {str(e)}")
            return False
    
    def store_prediction(self, prediction_data: Union[PredictionData, Dict]) -> Optional[int]:
        """Store model prediction in database."""
        try:
            if isinstance(prediction_data, PredictionData):
                data = {
                    'timestamp': prediction_data.timestamp,
                    'predicted_price': prediction_data.predicted_price,
                    'confidence_lower': prediction_data.confidence_lower,
                    'confidence_upper': prediction_data.confidence_upper,
                    'model_version': prediction_data.model_version
                }
            else:
                data = prediction_data
            
            # Validate prediction data
            required_fields = ['timestamp', 'predicted_price', 'confidence_lower', 'confidence_upper', 'model_version']
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in prediction data: {data}")
                return None
            
            if not (data['confidence_upper'] >= data['confidence_lower'] and
                   data['predicted_price'] >= data['confidence_lower'] and
                   data['predicted_price'] <= data['confidence_upper']):
                logger.error(f"Invalid confidence interval in prediction data: {data}")
                return None
            
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute('''
                        INSERT INTO predictions 
                        (timestamp, predicted_price, confidence_lower, confidence_upper, model_version)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (data['timestamp'], data['predicted_price'], data['confidence_lower'], 
                         data['confidence_upper'], data['model_version']))
                    conn.commit()
                    prediction_id = cursor.lastrowid
                    
            logger.info(f"Stored prediction with ID: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {str(e)}")
            return None
    
    def store_trade(self, trade_data: Union[TradeData, Dict]) -> Optional[int]:
        """Store trade information in database."""
        try:
            if isinstance(trade_data, TradeData):
                data = {
                    'entry_timestamp': trade_data.entry_timestamp,
                    'entry_price': trade_data.entry_price,
                    'position_size': trade_data.position_size,
                    'position_type': trade_data.position_type,
                    'confidence_score': trade_data.confidence_score,
                    'stop_loss': trade_data.stop_loss,
                    'take_profit': trade_data.take_profit,
                    'exit_timestamp': trade_data.exit_timestamp,
                    'exit_price': trade_data.exit_price,
                    'status': trade_data.status,
                    'pnl': trade_data.pnl
                }
            else:
                data = trade_data
            
            # Validate trade data
            required_fields = ['entry_timestamp', 'entry_price', 'position_size', 'position_type', 'confidence_score']
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in trade data: {data}")
                return None
            
            if data['position_type'] not in ['long', 'short']:
                logger.error(f"Invalid position type: {data['position_type']}")
                return None
            
            if not (0 <= data['confidence_score'] <= 1):
                logger.error(f"Invalid confidence score: {data['confidence_score']}")
                return None
            
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute('''
                        INSERT INTO trades 
                        (entry_timestamp, exit_timestamp, entry_price, exit_price, position_size, 
                         position_type, confidence_score, stop_loss, take_profit, status, pnl)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data['entry_timestamp'], data.get('exit_timestamp'), data['entry_price'],
                         data.get('exit_price'), data['position_size'], data['position_type'],
                         data['confidence_score'], data.get('stop_loss'), data.get('take_profit'),
                         data.get('status', 'open'), data.get('pnl')))
                    conn.commit()
                    trade_id = cursor.lastrowid
                    
            logger.info(f"Stored trade with ID: {trade_id}")
            return trade_id
            
        except Exception as e:
            logger.error(f"Failed to store trade: {str(e)}")
            return None
    
    def store_model_metrics(self, metrics_data: Union[ModelMetrics, Dict]) -> Optional[int]:
        """Store model performance metrics in database."""
        try:
            if isinstance(metrics_data, ModelMetrics):
                data = {
                    'timestamp': metrics_data.timestamp,
                    'mse': metrics_data.mse,
                    'mae': metrics_data.mae,
                    'directional_accuracy': metrics_data.directional_accuracy,
                    'model_version': metrics_data.model_version,
                    'sharpe_ratio': metrics_data.sharpe_ratio
                }
            else:
                data = metrics_data
            
            # Validate metrics data
            required_fields = ['timestamp', 'mse', 'mae', 'directional_accuracy', 'model_version']
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in metrics data: {data}")
                return None
            
            if not (0 <= data['directional_accuracy'] <= 1):
                logger.error(f"Invalid directional accuracy: {data['directional_accuracy']}")
                return None
            
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute('''
                        INSERT INTO model_metrics 
                        (timestamp, mse, mae, directional_accuracy, sharpe_ratio, model_version)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (data['timestamp'], data['mse'], data['mae'], data['directional_accuracy'],
                         data.get('sharpe_ratio'), data['model_version']))
                    conn.commit()
                    metrics_id = cursor.lastrowid
                    
            logger.info(f"Stored model metrics with ID: {metrics_id}")
            return metrics_id
            
        except Exception as e:
            logger.error(f"Failed to store model metrics: {str(e)}")
            return None
    
    def get_recent_prices(self, hours: int = 60) -> List[Dict]:
        """Retrieve recent price data."""
        try:
            cutoff_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT timestamp, open, high, low, close, volume, created_at
                    FROM btc_prices 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (cutoff_timestamp,))
                
                results = [dict(row) for row in cursor.fetchall()]
                
            logger.debug(f"Retrieved {len(results)} price records from last {hours} hours")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent prices: {str(e)}")
            return []
    
    def get_recent_predictions(self, hours: int = 24, model_version: Optional[str] = None) -> List[Dict]:
        """Retrieve recent predictions."""
        try:
            cutoff_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            if model_version:
                query = '''
                    SELECT id, timestamp, predicted_price, confidence_lower, confidence_upper, 
                           model_version, created_at
                    FROM predictions 
                    WHERE timestamp >= ? AND model_version = ?
                    ORDER BY timestamp DESC
                '''
                params = (cutoff_timestamp, model_version)
            else:
                query = '''
                    SELECT id, timestamp, predicted_price, confidence_lower, confidence_upper, 
                           model_version, created_at
                    FROM predictions 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                '''
                params = (cutoff_timestamp,)
            
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
            logger.debug(f"Retrieved {len(results)} predictions from last {hours} hours")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent predictions: {str(e)}")
            return []
    
    def get_active_trades(self) -> List[Dict]:
        """Retrieve all active trades."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT id, entry_timestamp, exit_timestamp, entry_price, exit_price, 
                           position_size, position_type, confidence_score, stop_loss, 
                           take_profit, status, pnl, created_at
                    FROM trades 
                    WHERE status = 'open'
                    ORDER BY entry_timestamp DESC
                ''')
                
                results = [dict(row) for row in cursor.fetchall()]
                
            logger.debug(f"Retrieved {len(results)} active trades")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve active trades: {str(e)}")
            return []
    
    def update_trade(self, trade_id: int, updates: Dict) -> bool:
        """Update existing trade record."""
        try:
            # Build dynamic update query
            update_fields = []
            values = []
            
            allowed_updates = ['exit_timestamp', 'exit_price', 'status', 'pnl']
            for field in allowed_updates:
                if field in updates:
                    update_fields.append(f"{field} = ?")
                    values.append(updates[field])
            
            if not update_fields:
                logger.warning("No valid update fields provided")
                return False
            
            values.append(trade_id)
            query = f"UPDATE trades SET {', '.join(update_fields)} WHERE id = ?"
            
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute(query, values)
                    conn.commit()
                    
                    if cursor.rowcount == 0:
                        logger.warning(f"No trade found with ID: {trade_id}")
                        return False
                    
            logger.info(f"Updated trade {trade_id} with fields: {list(updates.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update trade {trade_id}: {str(e)}")
            return False
    
    def get_trade_history(self, days: int = 30, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve trade history."""
        try:
            cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
            
            query = '''
                SELECT id, entry_timestamp, exit_timestamp, entry_price, exit_price, 
                       position_size, position_type, confidence_score, stop_loss, 
                       take_profit, status, pnl, created_at
                FROM trades 
                WHERE entry_timestamp >= ?
                ORDER BY entry_timestamp DESC
            '''
            
            params = [cutoff_timestamp]
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
            logger.debug(f"Retrieved {len(results)} trade history records")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve trade history: {str(e)}")
            return []
    
    def get_model_performance(self, days: int = 30, model_version: Optional[str] = None) -> List[Dict]:
        """Retrieve model performance metrics."""
        try:
            cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
            
            if model_version:
                query = '''
                    SELECT id, timestamp, mse, mae, directional_accuracy, sharpe_ratio, 
                           model_version, created_at
                    FROM model_metrics 
                    WHERE timestamp >= ? AND model_version = ?
                    ORDER BY timestamp DESC
                '''
                params = (cutoff_timestamp, model_version)
            else:
                query = '''
                    SELECT id, timestamp, mse, mae, directional_accuracy, sharpe_ratio, 
                           model_version, created_at
                    FROM model_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                '''
                params = (cutoff_timestamp,)
            
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
            logger.debug(f"Retrieved {len(results)} model performance records")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve model performance: {str(e)}")
            return []
    
    def store_performance_snapshot(self, snapshot_data: Union[PerformanceSnapshot, Dict]) -> Optional[int]:
        """Store performance snapshot in database."""
        try:
            if isinstance(snapshot_data, PerformanceSnapshot):
                data = {
                    'timestamp': snapshot_data.timestamp,
                    'total_return_pct': snapshot_data.total_return_pct,
                    'win_rate_pct': snapshot_data.win_rate_pct,
                    'avg_confidence': snapshot_data.avg_confidence,
                    'avg_uncertainty_pct': snapshot_data.avg_uncertainty_pct,
                    'trades_per_day': snapshot_data.trades_per_day,
                    'max_drawdown_pct': snapshot_data.max_drawdown_pct,
                    'portfolio_value': snapshot_data.portfolio_value,
                    'active_positions': snapshot_data.active_positions,
                    'sharpe_ratio': snapshot_data.sharpe_ratio
                }
            else:
                data = snapshot_data
            
            # Validate snapshot data
            required_fields = ['timestamp', 'total_return_pct', 'win_rate_pct', 'avg_confidence', 
                             'avg_uncertainty_pct', 'trades_per_day', 'max_drawdown_pct', 
                             'portfolio_value', 'active_positions']
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in performance snapshot: {data}")
                return None
            
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute('''
                        INSERT INTO performance_snapshots 
                        (timestamp, total_return_pct, win_rate_pct, avg_confidence, avg_uncertainty_pct,
                         trades_per_day, max_drawdown_pct, sharpe_ratio, portfolio_value, active_positions)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data['timestamp'], data['total_return_pct'], data['win_rate_pct'],
                         data['avg_confidence'], data['avg_uncertainty_pct'], data['trades_per_day'],
                         data['max_drawdown_pct'], data.get('sharpe_ratio'), data['portfolio_value'],
                         data['active_positions']))
                    conn.commit()
                    snapshot_id = cursor.lastrowid
                    
            logger.info(f"Stored performance snapshot with ID: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to store performance snapshot: {str(e)}")
            return None
    
    def store_parameter_adjustment(self, adjustment_data: Union[ParameterAdjustment, Dict]) -> Optional[int]:
        """Store parameter adjustment record in database."""
        try:
            if isinstance(adjustment_data, ParameterAdjustment):
                data = {
                    'timestamp': adjustment_data.timestamp,
                    'parameter_name': adjustment_data.parameter_name,
                    'old_value': adjustment_data.old_value,
                    'new_value': adjustment_data.new_value,
                    'adjustment_reason': adjustment_data.adjustment_reason,
                    'trigger_metric': adjustment_data.trigger_metric,
                    'trigger_value': adjustment_data.trigger_value,
                    'performance_before': adjustment_data.performance_before
                }
            else:
                data = adjustment_data
            
            # Validate adjustment data
            required_fields = ['timestamp', 'parameter_name', 'old_value', 'new_value', 
                             'adjustment_reason', 'trigger_metric', 'trigger_value']
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in parameter adjustment: {data}")
                return None
            
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute('''
                        INSERT INTO parameter_adjustments 
                        (timestamp, parameter_name, old_value, new_value, adjustment_reason,
                         trigger_metric, trigger_value, performance_before)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data['timestamp'], data['parameter_name'], data['old_value'],
                         data['new_value'], data['adjustment_reason'], data['trigger_metric'],
                         data['trigger_value'], data.get('performance_before')))
                    conn.commit()
                    adjustment_id = cursor.lastrowid
                    
            logger.info(f"Stored parameter adjustment with ID: {adjustment_id}")
            return adjustment_id
            
        except Exception as e:
            logger.error(f"Failed to store parameter adjustment: {str(e)}")
            return None
    
    def store_performance_alert(self, alert_data: Union[PerformanceAlert, Dict]) -> Optional[int]:
        """Store performance alert in database."""
        try:
            if isinstance(alert_data, PerformanceAlert):
                data = {
                    'timestamp': alert_data.timestamp,
                    'alert_type': alert_data.alert_type,
                    'severity': alert_data.severity,
                    'message': alert_data.message,
                    'threshold_value': alert_data.threshold_value,
                    'actual_value': alert_data.actual_value,
                    'acknowledged': alert_data.acknowledged,
                    'acknowledged_at': alert_data.acknowledged_at,
                    'resolved': alert_data.resolved,
                    'resolved_at': alert_data.resolved_at
                }
            else:
                data = alert_data
            
            # Validate alert data
            required_fields = ['timestamp', 'alert_type', 'severity', 'message', 
                             'threshold_value', 'actual_value']
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in performance alert: {data}")
                return None
            
            # Validate alert_type and severity
            valid_alert_types = ['win_rate', 'drawdown', 'confidence', 'uncertainty', 'trading_frequency', 'overtrading']
            valid_severities = ['low', 'medium', 'high', 'critical']
            
            if data['alert_type'] not in valid_alert_types:
                logger.error(f"Invalid alert_type: {data['alert_type']}")
                return None
                
            if data['severity'] not in valid_severities:
                logger.error(f"Invalid severity: {data['severity']}")
                return None
            
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute('''
                        INSERT INTO performance_alerts 
                        (timestamp, alert_type, severity, message, threshold_value, actual_value,
                         acknowledged, acknowledged_at, resolved, resolved_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data['timestamp'], data['alert_type'], data['severity'], data['message'],
                         data['threshold_value'], data['actual_value'], data.get('acknowledged', False),
                         data.get('acknowledged_at'), data.get('resolved', False), data.get('resolved_at')))
                    conn.commit()
                    alert_id = cursor.lastrowid
                    
            logger.info(f"Stored performance alert with ID: {alert_id}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Failed to store performance alert: {str(e)}")
            return None
    
    def get_recent_performance_snapshots(self, hours: int = 24, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve recent performance snapshots."""
        try:
            cutoff_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            query = '''
                SELECT id, timestamp, total_return_pct, win_rate_pct, avg_confidence, 
                       avg_uncertainty_pct, trades_per_day, max_drawdown_pct, sharpe_ratio,
                       portfolio_value, active_positions, created_at
                FROM performance_snapshots 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            
            params = [cutoff_timestamp]
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
            logger.debug(f"Retrieved {len(results)} performance snapshots from last {hours} hours")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent performance snapshots: {str(e)}")
            return []
    
    def get_parameter_adjustment_history(self, parameter_name: Optional[str] = None, days: int = 30) -> List[Dict]:
        """Retrieve parameter adjustment history."""
        try:
            cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
            
            if parameter_name:
                query = '''
                    SELECT id, timestamp, parameter_name, old_value, new_value, adjustment_reason,
                           trigger_metric, trigger_value, performance_before, created_at
                    FROM parameter_adjustments 
                    WHERE timestamp >= ? AND parameter_name = ?
                    ORDER BY timestamp DESC
                '''
                params = (cutoff_timestamp, parameter_name)
            else:
                query = '''
                    SELECT id, timestamp, parameter_name, old_value, new_value, adjustment_reason,
                           trigger_metric, trigger_value, performance_before, created_at
                    FROM parameter_adjustments 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                '''
                params = (cutoff_timestamp,)
            
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
            logger.debug(f"Retrieved {len(results)} parameter adjustments")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve parameter adjustment history: {str(e)}")
            return []
    
    def get_active_alerts(self, alert_type: Optional[str] = None) -> List[Dict]:
        """Retrieve active (unresolved) performance alerts."""
        try:
            if alert_type:
                query = '''
                    SELECT id, timestamp, alert_type, severity, message, threshold_value, 
                           actual_value, acknowledged, acknowledged_at, resolved, resolved_at, created_at
                    FROM performance_alerts 
                    WHERE resolved = FALSE AND alert_type = ?
                    ORDER BY severity DESC, timestamp DESC
                '''
                params = (alert_type,)
            else:
                query = '''
                    SELECT id, timestamp, alert_type, severity, message, threshold_value, 
                           actual_value, acknowledged, acknowledged_at, resolved, resolved_at, created_at
                    FROM performance_alerts 
                    WHERE resolved = FALSE
                    ORDER BY severity DESC, timestamp DESC
                '''
                params = ()
            
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
            logger.debug(f"Retrieved {len(results)} active alerts")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve active alerts: {str(e)}")
            return []
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark a performance alert as acknowledged."""
        try:
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute('''
                        UPDATE performance_alerts 
                        SET acknowledged = TRUE, acknowledged_at = ?
                        WHERE id = ? AND acknowledged = FALSE
                    ''', (int(datetime.now().timestamp()), alert_id))
                    conn.commit()
                    
                    if cursor.rowcount == 0:
                        logger.warning(f"No unacknowledged alert found with ID: {alert_id}")
                        return False
                    
            logger.info(f"Acknowledged alert {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
            return False
    
    def resolve_alert(self, alert_id: int) -> bool:
        """Mark a performance alert as resolved."""
        try:
            with self._lock:
                with self.get_connection() as conn:
                    cursor = conn.execute('''
                        UPDATE performance_alerts 
                        SET resolved = TRUE, resolved_at = ?
                        WHERE id = ? AND resolved = FALSE
                    ''', (int(datetime.now().timestamp()), alert_id))
                    conn.commit()
                    
                    if cursor.rowcount == 0:
                        logger.warning(f"No unresolved alert found with ID: {alert_id}")
                        return False
                    
            logger.info(f"Resolved alert {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
            return False
    
    def create_backup(self, backup_name: Optional[str] = None) -> bool:
        """Create database backup."""
        try:
            if backup_name is None:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
            backup_file = self.backup_path / backup_name
            
            with self._lock:
                shutil.copy2(self.db_path, backup_file)
            
            logger.info(f"Database backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            return False
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore database from backup."""
        try:
            backup_file = self.backup_path / backup_name
            
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False
            
            # Create current backup before restore
            self.create_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            
            with self._lock:
                shutil.copy2(backup_file, self.db_path)
            
            logger.info(f"Database restored from backup: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {str(e)}")
            return False
    
    def cleanup_old_backups(self, keep_days: int = 30) -> int:
        """Remove old backup files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            removed_count = 0
            
            for backup_file in self.backup_path.glob("*.db"):
                if backup_file.stat().st_ctime < cutoff_date.timestamp():
                    backup_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old backup: {backup_file}")
            
            logger.info(f"Cleaned up {removed_count} old backup files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {str(e)}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            with self.get_connection() as conn:
                # Table row counts
                for table in ['btc_prices', 'predictions', 'trades', 'model_metrics', 
                             'performance_snapshots', 'parameter_adjustments', 'performance_alerts']:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Database size
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                stats['database_size_bytes'] = page_size * page_count
                
                # Active trades
                cursor = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
                stats['active_trades'] = cursor.fetchone()[0]
                
                # Latest data timestamps
                for table, ts_field in [('btc_prices', 'timestamp'), ('predictions', 'timestamp'), 
                                       ('trades', 'entry_timestamp'), ('model_metrics', 'timestamp'),
                                       ('performance_snapshots', 'timestamp'), ('parameter_adjustments', 'timestamp'),
                                       ('performance_alerts', 'timestamp')]:
                    cursor = conn.execute(f"SELECT MAX({ts_field}) FROM {table}")
                    max_ts = cursor.fetchone()[0]
                    if max_ts:
                        stats[f"latest_{table}_timestamp"] = max_ts
                        stats[f"latest_{table}_datetime"] = datetime.fromtimestamp(max_ts).isoformat()
                
                # Additional performance-specific stats
                cursor = conn.execute("SELECT COUNT(*) FROM performance_alerts WHERE resolved = FALSE")
                stats['unresolved_alerts'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM performance_alerts WHERE acknowledged = FALSE")
                stats['unacknowledged_alerts'] = cursor.fetchone()[0]
            
            logger.debug(f"Database statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {}
    
    def vacuum_database(self) -> bool:
        """Optimize database by running VACUUM."""
        try:
            with self._lock:
                with self.get_connection() as conn:
                    conn.execute("VACUUM")
                    conn.commit()
            
            logger.info("Database vacuum completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to vacuum database: {str(e)}")
            return False
    
    def close(self):
        """Close database manager (cleanup method)."""
        logger.info("DatabaseManager closed")
        # SQLite connections are closed automatically in context manager 