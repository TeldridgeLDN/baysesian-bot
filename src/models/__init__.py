"""
Models module for Bayesian Crypto Trading Bot.
Contains neural network models, uncertainty quantification, training pipelines, and production services.
"""

try:
    from .bayesian_lstm import BayesianLSTM, MonteCarloDropout, ModelManager
    from .uncertainty import UncertaintyQuantifier, ConfidenceScorer
    from .training import TrainingPipeline, AutoTrainer
    from .prediction_service import PredictionService
    from .model_factory import ModelFactory, TradingPredictor
    
    __all__ = [
        'BayesianLSTM',
        'MonteCarloDropout', 
        'ModelManager',
        'UncertaintyQuantifier',
        'ConfidenceScorer',
        'TrainingPipeline',
        'AutoTrainer',
        'PredictionService',
        'ModelFactory',
        'TradingPredictor'
    ]
    
except ImportError as e:
    print(f"Warning: TensorFlow models not available: {e}")
    print("Using mock implementations for testing")
    
    # Import mock implementations
    from .mock_predictor import MockBayesianPredictor, MockPredictionService, MockTradingPredictor
    
    __all__ = [
        'MockBayesianPredictor',
        'MockPredictionService', 
        'MockTradingPredictor'
    ]
