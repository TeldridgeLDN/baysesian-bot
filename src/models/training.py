"""
Model Training Pipeline for Bayesian LSTM.
Handles training, validation, and model management operations.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

# Import project modules
from ..utils.config import ConfigManager, ModelConfig
from ..data.preprocessors import PriceDataPreprocessor
from .bayesian_lstm import BayesianLSTM, ModelManager
from .uncertainty import UncertaintyQuantifier

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Complete training pipeline for Bayesian LSTM model.
    
    Handles:
    - Data preparation and validation
    - Model training with callbacks
    - Performance monitoring
    - Model versioning and persistence
    - Training history tracking
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize training pipeline.
        
        Args:
            config: Configuration manager object
        """
        self.config = config
        self.model_config = config.get_model_config()
        self.data_config = config.get_data_config()
        
        # Initialize components
        self.preprocessor = PriceDataPreprocessor(config)
        self.model_manager = ModelManager()
        self.uncertainty_quantifier = UncertaintyQuantifier(
            confidence_level=self.model_config.confidence_interval
        )
        
        # Training state
        self.current_model = None
        self.training_history = []
        self.performance_metrics = {}
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs/training', exist_ok=True)
        
        logger.info("TrainingPipeline initialized")
    
    def prepare_training_data(self, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data.
        
        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        logger.info("Preparing training data")
        
        # Load data from database
        db_path = self.data_config['db_path']
        
        # Build query
        query = "SELECT * FROM btc_prices ORDER BY timestamp"
        params = []
        
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(pd.Timestamp(start_date).timestamp())
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(pd.Timestamp(end_date).timestamp())
            
            query = f"SELECT * FROM btc_prices WHERE {' AND '.join(conditions)} ORDER BY timestamp"
        
        # Load data
        with sqlite3.connect(db_path) as conn:
            raw_data = pd.read_sql_query(query, conn, params=params)
        
        if len(raw_data) == 0:
            raise ValueError("No data found for the specified date range")
        
        logger.info(f"Loaded {len(raw_data)} data points from {raw_data['timestamp'].min()} to {raw_data['timestamp'].max()}")
        
        # Process data through pipeline
        X, y, processing_report = self.preprocessor.process_pipeline(raw_data)
        
        # Split data
        X_train, X_val, y_train, y_val = self.preprocessor.split_data(X, y)
        
        # Validate data shapes
        assert X_train.shape[1] == self.model_config.sequence_length, \
            f"Sequence length mismatch: {X_train.shape[1]} vs {self.model_config.sequence_length}"
        
        # Update model config with actual feature count
        self.model_config.feature_count = X_train.shape[2]
        
        logger.info(f"Training data prepared:")
        logger.info(f"  Training: {X_train.shape} -> {y_train.shape}")
        logger.info(f"  Validation: {X_val.shape} -> {y_val.shape}")
        logger.info(f"  Features: {X_train.shape[2]}")
        logger.info(f"  Quality score: {processing_report['quality_score']:.3f}")
        
        return X_train, X_val, y_train, y_val
    
    def create_model(self) -> BayesianLSTM:
        """
        Create a new Bayesian LSTM model.
        
        Returns:
            BayesianLSTM instance
        """
        model_config = {
            'sequence_length': self.model_config.sequence_length,
            'feature_count': getattr(self.model_config, 'feature_count', 47),
            'lstm_units': self.model_config.lstm_units,
            'dropout_rate': self.model_config.dropout_rate,
            'recurrent_dropout_rate': self.model_config.recurrent_dropout_rate,
            'dense_dropout_rate': self.model_config.dense_dropout_rate,
            'learning_rate': self.model_config.learning_rate,
            'monte_carlo_samples': self.model_config.monte_carlo_samples,
            'confidence_interval': self.model_config.confidence_interval
        }
        
        self.current_model = self.model_manager.create_model(model_config)
        logger.info(f"Created new model: {self.current_model.model_version}")
        
        return self.current_model
    
    def train_model(self, 
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   save_model: bool = True) -> Dict[str, Any]:
        """
        Train the Bayesian LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        if self.current_model is None:
            self.create_model()
        
        logger.info("Starting model training")
        
        # Train model
        training_stats = self.current_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.model_config.epochs,
            batch_size=self.model_config.batch_size,
            early_stopping_patience=self.model_config.early_stopping_patience,
            verbose=1
        )
        
        # Evaluate model
        evaluation_metrics = self.current_model.evaluate_model(X_val, y_val)
        
        # Generate uncertainty analysis
        uncertainty_results = self.current_model.predict_with_uncertainty(X_val)
        uncertainty_report = self.uncertainty_quantifier.generate_uncertainty_report(
            uncertainty_results['all_samples'],
            y_val
        )
        
        # Combine results
        training_results = {
            'training_stats': training_stats,
            'evaluation_metrics': evaluation_metrics,
            'uncertainty_report': uncertainty_report,
            'model_version': self.current_model.model_version,
            'training_completed_at': datetime.now().isoformat()
        }
        
        # Save model if requested
        if save_model:
            model_path = self.model_manager.save_current_model()
            training_results['model_path'] = model_path
            logger.info(f"Model saved to {model_path}")
        
        # Update training history
        self.training_history.append(training_results)
        self.performance_metrics = evaluation_metrics
        
        # Log training summary
        logger.info("Training completed successfully:")
        logger.info(f"  Final validation loss: {training_stats['final_val_loss']:.6f}")
        logger.info(f"  Directional accuracy: {evaluation_metrics['directional_accuracy']:.3f}")
        logger.info(f"  Mean uncertainty: {uncertainty_report['summary']['mean_uncertainty']:.4f}")
        logger.info(f"  Interval coverage: {evaluation_metrics['interval_coverage']:.3f}")
        
        return training_results
    
    def full_training_pipeline(self, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute complete training pipeline.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Complete training results
        """
        logger.info("Executing full training pipeline")
        
        try:
            # Prepare data
            X_train, X_val, y_train, y_val = self.prepare_training_data(start_date, end_date)
            
            # Train model
            training_results = self.train_model(X_train, y_train, X_val, y_val)
            
            # Store training metadata
            self.save_training_metadata(training_results)
            
            logger.info("Full training pipeline completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def retrain_model(self, 
                     performance_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Retrain model with latest data.
        
        Args:
            performance_threshold: Minimum performance threshold for keeping model
            
        Returns:
            Retraining results
        """
        logger.info("Starting model retraining")
        
        # Calculate training window (last 180 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.data_config.get('historical_days', 180))
        
        # Execute training pipeline
        training_results = self.full_training_pipeline(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Check if new model is better
        current_performance = training_results['evaluation_metrics']['directional_accuracy']
        
        if current_performance < performance_threshold:
            logger.warning(f"New model performance ({current_performance:.3f}) below threshold ({performance_threshold})")
            training_results['performance_warning'] = True
        else:
            logger.info(f"Retraining successful: performance = {current_performance:.3f}")
            training_results['performance_warning'] = False
        
        # Update model metrics in database
        self.store_model_metrics(training_results)
        
        return training_results
    
    def validate_model_performance(self, 
                                 test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate current model performance.
        
        Args:
            test_data: Optional test data (if None, uses recent data)
            
        Returns:
            Validation results
        """
        if self.current_model is None:
            raise ValueError("No model available for validation")
        
        logger.info("Validating model performance")
        
        if test_data is None:
            # Use recent data for validation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            
            # Load test data
            db_path = self.data_config['db_path']
            query = """
                SELECT * FROM btc_prices 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """
            
            with sqlite3.connect(db_path) as conn:
                test_data = pd.read_sql_query(
                    query, conn, 
                    params=[start_date.timestamp(), end_date.timestamp()]
                )
        
        if len(test_data) < self.model_config.sequence_length:
            raise ValueError(f"Insufficient test data: {len(test_data)} < {self.model_config.sequence_length}")
        
        # Process test data
        X_test, y_test, _ = self.preprocessor.process_pipeline(test_data)
        
        # Evaluate model
        validation_metrics = self.current_model.evaluate_model(X_test, y_test)
        
        # Generate predictions with uncertainty
        prediction_results = self.current_model.predict_with_uncertainty(X_test)
        
        # Uncertainty analysis
        uncertainty_report = self.uncertainty_quantifier.generate_uncertainty_report(
            prediction_results['all_samples'],
            y_test
        )
        
        validation_results = {
            'validation_metrics': validation_metrics,
            'uncertainty_analysis': uncertainty_report,
            'test_samples': len(X_test),
            'model_version': self.current_model.model_version,
            'validation_date': datetime.now().isoformat()
        }
        
        logger.info(f"Model validation completed:")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  MAE: {validation_metrics['mae']:.6f}")
        logger.info(f"  Directional accuracy: {validation_metrics['directional_accuracy']:.3f}")
        
        return validation_results
    
    def save_training_metadata(self, training_results: Dict[str, Any]):
        """
        Save training metadata to file.
        
        Args:
            training_results: Training results dictionary
        """
        metadata_path = f"logs/training/training_{training_results['model_version']}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to {metadata_path}")
    
    def store_model_metrics(self, training_results: Dict[str, Any]):
        """
        Store model metrics in database.
        
        Args:
            training_results: Training results dictionary
        """
        db_path = self.data_config['db_path']
        
        metrics = training_results['evaluation_metrics']
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO model_metrics (
                    timestamp, mse, mae, directional_accuracy, 
                    sharpe_ratio, model_version
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                int(datetime.now().timestamp()),
                metrics['mse'],
                metrics['mae'],
                metrics['directional_accuracy'],
                metrics.get('sharpe_ratio', 0.0),
                training_results['model_version']
            ))
        
        logger.info("Model metrics stored in database")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get training history.
        
        Returns:
            List of training results
        """
        return self.training_history.copy()
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of current model performance.
        
        Returns:
            Performance summary dictionary
        """
        if not self.performance_metrics:
            return {'error': 'No performance metrics available'}
        
        summary = {
            'model_version': self.current_model.model_version if self.current_model else 'None',
            'performance_metrics': self.performance_metrics,
            'training_sessions': len(self.training_history),
            'last_training': self.training_history[-1]['training_completed_at'] if self.training_history else None,
            'summary_generated_at': datetime.now().isoformat()
        }
        
        return summary

class AutoTrainer:
    """
    Automated training scheduler for continuous model improvement.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize auto trainer.
        
        Args:
            config: Configuration manager object
        """
        self.config = config
        self.model_config = config.get_model_config()
        self.training_pipeline = TrainingPipeline(config)
        
        self.last_training_time = None
        self.training_interval = timedelta(hours=self.model_config.retrain_frequency_hours)
        
        logger.info(f"AutoTrainer initialized with {self.model_config.retrain_frequency_hours}h interval")
    
    def should_retrain(self) -> bool:
        """
        Check if model should be retrained.
        
        Returns:
            True if retraining is needed
        """
        if self.last_training_time is None:
            return True
        
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training >= self.training_interval
    
    def check_model_degradation(self) -> bool:
        """
        Check if current model has degraded.
        
        Returns:
            True if model degradation detected
        """
        if self.training_pipeline.current_model is None:
            return True
        
        try:
            # Validate on recent data
            validation_results = self.training_pipeline.validate_model_performance()
            
            # Check performance thresholds
            metrics = validation_results['validation_metrics']
            directional_accuracy = metrics['directional_accuracy']
            
            # Degradation if accuracy drops below 50%
            if directional_accuracy < 0.5:
                logger.warning(f"Model degradation detected: accuracy = {directional_accuracy:.3f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking model degradation: {str(e)}")
            return True  # Assume degradation if we can't check
    
    def auto_retrain(self) -> Dict[str, Any]:
        """
        Automatically retrain model if needed.
        
        Returns:
            Retraining results or status
        """
        if not self.should_retrain():
            return {
                'action': 'no_retraining_needed',
                'next_training_due': (self.last_training_time + self.training_interval).isoformat()
            }
        
        logger.info("Starting automatic retraining")
        
        try:
            # Check for degradation
            degradation_detected = self.check_model_degradation()
            
            # Retrain model
            training_results = self.training_pipeline.retrain_model()
            
            # Update last training time
            self.last_training_time = datetime.now()
            
            results = {
                'action': 'retrained',
                'degradation_detected': degradation_detected,
                'training_results': training_results,
                'next_training_due': (self.last_training_time + self.training_interval).isoformat()
            }
            
            logger.info("Automatic retraining completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Automatic retraining failed: {str(e)}")
            return {
                'action': 'retraining_failed',
                'error': str(e),
                'next_retry': (datetime.now() + timedelta(hours=1)).isoformat()
            }
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status.
        
        Returns:
            Training status dictionary
        """
        status = {
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_interval_hours': self.model_config.retrain_frequency_hours,
            'should_retrain': self.should_retrain(),
            'current_model_version': self.training_pipeline.current_model.model_version if self.training_pipeline.current_model else None,
            'training_history_count': len(self.training_pipeline.training_history),
            'status_generated_at': datetime.now().isoformat()
        }
        
        if self.last_training_time:
            next_training = self.last_training_time + self.training_interval
            status['next_training_due'] = next_training.isoformat()
            status['time_until_next_training'] = str(next_training - datetime.now())
        
        return status 