"""
Uncertainty Quantification Module for Bayesian LSTM.
Implements Monte Carlo methods and statistical analysis for prediction uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

class UncertaintyQuantifier:
    """
    Handles uncertainty quantification and analysis for Bayesian LSTM predictions.
    
    Provides methods for:
    - Monte Carlo sampling analysis
    - Confidence interval calculations
    - Uncertainty decomposition (epistemic vs aleatoric)
    - Calibration assessment
    - Risk metrics computation
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize uncertainty quantifier.
        
        Args:
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        logger.info(f"UncertaintyQuantifier initialized with {confidence_level*100}% confidence level")
    
    def analyze_mc_samples(self, mc_samples: np.ndarray) -> Dict[str, Any]:
        """
        Analyze Monte Carlo samples to extract uncertainty statistics.
        
        Args:
            mc_samples: Array of shape (n_samples, n_predictions)
            
        Returns:
            Dictionary with uncertainty statistics
        """
        n_samples, n_predictions = mc_samples.shape
        
        # Basic statistics
        mean_predictions = np.mean(mc_samples, axis=0)
        std_predictions = np.std(mc_samples, axis=0)
        var_predictions = np.var(mc_samples, axis=0)
        
        # Confidence intervals
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        confidence_lower = np.percentile(mc_samples, lower_percentile, axis=0)
        confidence_upper = np.percentile(mc_samples, upper_percentile, axis=0)
        interval_width = confidence_upper - confidence_lower
        
        # Distribution statistics
        skewness = stats.skew(mc_samples, axis=0)
        kurtosis = stats.kurtosis(mc_samples, axis=0)
        
        # Quantiles
        q25 = np.percentile(mc_samples, 25, axis=0)
        q75 = np.percentile(mc_samples, 75, axis=0)
        iqr = q75 - q25
        
        analysis = {
            'n_samples': n_samples,
            'n_predictions': n_predictions,
            'mean_predictions': mean_predictions,
            'std_predictions': std_predictions,
            'var_predictions': var_predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'interval_width': interval_width,
            'mean_interval_width': np.mean(interval_width),
            'std_interval_width': np.std(interval_width),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'confidence_level': self.confidence_level
        }
        
        logger.info(f"MC analysis: {n_samples} samples, mean uncertainty = {np.mean(std_predictions):.4f}")
        
        return analysis
    
    def calculate_prediction_intervals(self, 
                                     mc_samples: np.ndarray,
                                     method: str = 'percentile') -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals using different methods.
        
        Args:
            mc_samples: Monte Carlo samples (n_samples, n_predictions)
            method: Method for interval calculation ('percentile', 'gaussian')
            
        Returns:
            Dictionary with interval bounds
        """
        if method == 'percentile':
            # Use empirical percentiles
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            
            lower_bound = np.percentile(mc_samples, lower_percentile, axis=0)
            upper_bound = np.percentile(mc_samples, upper_percentile, axis=0)
            
        elif method == 'gaussian':
            # Assume Gaussian distribution
            mean_pred = np.mean(mc_samples, axis=0)
            std_pred = np.std(mc_samples, axis=0)
            
            z_score = stats.norm.ppf(1 - self.alpha / 2)
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        intervals = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'width': upper_bound - lower_bound,
            'method': method,
            'confidence_level': self.confidence_level
        }
        
        return intervals
    
    def assess_calibration(self, 
                          predictions: np.ndarray,
                          uncertainties: np.ndarray,
                          true_values: np.ndarray,
                          n_bins: int = 10) -> Dict[str, Any]:
        """
        Assess calibration of uncertainty estimates.
        
        Args:
            predictions: Point predictions
            uncertainties: Uncertainty estimates (standard deviations)
            true_values: Ground truth values
            n_bins: Number of bins for calibration curve
            
        Returns:
            Calibration assessment results
        """
        # Calculate prediction errors
        errors = np.abs(predictions - true_values)
        
        # Normalize errors by uncertainty
        normalized_errors = errors / (uncertainties + 1e-8)  # Avoid division by zero
        
        # Create bins based on uncertainty quantiles
        uncertainty_quantiles = np.linspace(0, 1, n_bins + 1)
        uncertainty_thresholds = np.percentile(uncertainties, uncertainty_quantiles * 100)
        
        calibration_results = {
            'bins': [],
            'expected_coverage': [],
            'observed_coverage': [],
            'bin_centers': [],
            'bin_counts': []
        }
        
        for i in range(n_bins):
            # Define bin
            lower_thresh = uncertainty_thresholds[i]
            upper_thresh = uncertainty_thresholds[i + 1]
            
            # Find predictions in this uncertainty bin
            in_bin = (uncertainties >= lower_thresh) & (uncertainties < upper_thresh)
            
            if i == n_bins - 1:  # Include upper bound in last bin
                in_bin = (uncertainties >= lower_thresh) & (uncertainties <= upper_thresh)
            
            if np.sum(in_bin) == 0:
                continue
            
            # Calculate expected coverage for this uncertainty level
            bin_uncertainties = uncertainties[in_bin]
            mean_uncertainty = np.mean(bin_uncertainties)
            
            # For Gaussian assumption, calculate expected coverage
            z_score = stats.norm.ppf(1 - self.alpha / 2)
            expected_coverage = 2 * stats.norm.cdf(z_score) - 1
            
            # Calculate observed coverage
            bin_errors = errors[in_bin]
            bin_predictions = predictions[in_bin]
            bin_true = true_values[in_bin]
            
            # Check if true values fall within confidence intervals
            lower_bound = bin_predictions - z_score * bin_uncertainties
            upper_bound = bin_predictions + z_score * bin_uncertainties
            
            in_interval = (bin_true >= lower_bound) & (bin_true <= upper_bound)
            observed_coverage = np.mean(in_interval)
            
            calibration_results['bins'].append((lower_thresh, upper_thresh))
            calibration_results['expected_coverage'].append(expected_coverage)
            calibration_results['observed_coverage'].append(observed_coverage)
            calibration_results['bin_centers'].append(mean_uncertainty)
            calibration_results['bin_counts'].append(np.sum(in_bin))
        
        # Calculate calibration metrics
        expected = np.array(calibration_results['expected_coverage'])
        observed = np.array(calibration_results['observed_coverage'])
        
        calibration_error = np.mean(np.abs(expected - observed))
        reliability = 1 - calibration_error
        
        calibration_results.update({
            'calibration_error': calibration_error,
            'reliability': reliability,
            'mean_expected_coverage': np.mean(expected),
            'mean_observed_coverage': np.mean(observed),
            'confidence_level': self.confidence_level
        })
        
        logger.info(f"Calibration assessment: reliability = {reliability:.3f}, error = {calibration_error:.3f}")
        
        return calibration_results
    
    def decompose_uncertainty(self, 
                            mc_samples: np.ndarray,
                            true_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Decompose uncertainty into epistemic (model) and aleatoric (data) components.
        
        Args:
            mc_samples: Monte Carlo samples (n_samples, n_predictions)
            true_values: Optional ground truth for validation
            
        Returns:
            Uncertainty decomposition results
        """
        # Epistemic uncertainty: variance across MC samples
        epistemic_uncertainty = np.var(mc_samples, axis=0)
        epistemic_std = np.sqrt(epistemic_uncertainty)
        
        # Total uncertainty
        total_variance = np.var(mc_samples, axis=0)
        total_std = np.sqrt(total_variance)
        
        # For LSTM, aleatoric uncertainty is harder to separate
        # We approximate it as the remaining variance
        mean_predictions = np.mean(mc_samples, axis=0)
        
        if true_values is not None:
            # Calculate prediction errors
            prediction_errors = (mean_predictions - true_values) ** 2
            
            # Approximate aleatoric uncertainty as irreducible error
            aleatoric_uncertainty = np.maximum(0, prediction_errors - epistemic_uncertainty)
            aleatoric_std = np.sqrt(aleatoric_uncertainty)
        else:
            # Without ground truth, assume aleatoric is a fraction of total
            aleatoric_uncertainty = total_variance * 0.3  # Rough approximation
            aleatoric_std = np.sqrt(aleatoric_uncertainty)
        
        decomposition = {
            'epistemic_uncertainty': epistemic_uncertainty,
            'epistemic_std': epistemic_std,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'aleatoric_std': aleatoric_std,
            'total_uncertainty': total_variance,
            'total_std': total_std,
            'epistemic_ratio': epistemic_uncertainty / (total_variance + 1e-8),
            'aleatoric_ratio': aleatoric_uncertainty / (total_variance + 1e-8),
            'mean_epistemic': np.mean(epistemic_std),
            'mean_aleatoric': np.mean(aleatoric_std),
            'mean_total': np.mean(total_std)
        }
        
        logger.info(f"Uncertainty decomposition: epistemic = {np.mean(epistemic_std):.4f}, "
                   f"aleatoric = {np.mean(aleatoric_std):.4f}")
        
        return decomposition
    
    def calculate_risk_metrics(self, 
                             predictions: np.ndarray,
                             uncertainties: np.ndarray,
                             confidence_lower: np.ndarray,
                             confidence_upper: np.ndarray,
                             price_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Calculate risk metrics based on predictions and uncertainties.
        
        Args:
            predictions: Point predictions
            uncertainties: Uncertainty estimates
            confidence_lower: Lower confidence bounds
            confidence_upper: Upper confidence bounds
            price_threshold: Minimum price change threshold for trading
            
        Returns:
            Risk metrics dictionary
        """
        # Relative uncertainty (uncertainty as fraction of prediction)
        relative_uncertainty = uncertainties / (np.abs(predictions) + 1e-8)
        
        # Confidence interval width as percentage
        interval_width_pct = ((confidence_upper - confidence_lower) / 
                             (np.abs(predictions) + 1e-8)) * 100
        
        # Risk categories based on uncertainty
        low_risk = relative_uncertainty < 0.01  # < 1%
        medium_risk = (relative_uncertainty >= 0.01) & (relative_uncertainty < 0.03)  # 1-3%
        high_risk = relative_uncertainty >= 0.03  # > 3%
        
        # Trading signal confidence
        price_change_pct = np.abs(predictions) / np.abs(predictions + 1e-8) * 100
        tradeable_signals = (price_change_pct > price_threshold) & (interval_width_pct < 3.0)
        
        # Uncertainty-based position sizing
        # Higher uncertainty = smaller position
        max_position_size = 0.10  # 10% max
        uncertainty_penalty = np.minimum(relative_uncertainty * 5, 0.8)  # Cap at 80% reduction
        position_sizes = max_position_size * (1 - uncertainty_penalty)
        
        risk_metrics = {
            'relative_uncertainty': relative_uncertainty,
            'mean_relative_uncertainty': np.mean(relative_uncertainty),
            'interval_width_pct': interval_width_pct,
            'mean_interval_width_pct': np.mean(interval_width_pct),
            'low_risk_count': np.sum(low_risk),
            'medium_risk_count': np.sum(medium_risk),
            'high_risk_count': np.sum(high_risk),
            'low_risk_pct': np.mean(low_risk) * 100,
            'medium_risk_pct': np.mean(medium_risk) * 100,
            'high_risk_pct': np.mean(high_risk) * 100,
            'tradeable_signals': tradeable_signals,
            'tradeable_count': np.sum(tradeable_signals),
            'tradeable_pct': np.mean(tradeable_signals) * 100,
            'position_sizes': position_sizes,
            'mean_position_size': np.mean(position_sizes),
            'max_position_size': np.max(position_sizes),
            'min_position_size': np.min(position_sizes)
        }
        
        logger.info(f"Risk metrics: {np.mean(tradeable_signals)*100:.1f}% tradeable signals, "
                   f"mean position size: {np.mean(position_sizes)*100:.1f}%")
        
        return risk_metrics
    
    def generate_uncertainty_report(self, 
                                  mc_samples: np.ndarray,
                                  true_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive uncertainty analysis report.
        
        Args:
            mc_samples: Monte Carlo samples
            true_values: Optional ground truth values
            
        Returns:
            Comprehensive uncertainty report
        """
        logger.info("Generating comprehensive uncertainty report")
        
        # Basic MC analysis
        mc_analysis = self.analyze_mc_samples(mc_samples)
        
        # Prediction intervals
        intervals_percentile = self.calculate_prediction_intervals(mc_samples, 'percentile')
        intervals_gaussian = self.calculate_prediction_intervals(mc_samples, 'gaussian')
        
        # Uncertainty decomposition
        uncertainty_decomp = self.decompose_uncertainty(mc_samples, true_values)
        
        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(
            mc_analysis['mean_predictions'],
            mc_analysis['std_predictions'],
            intervals_percentile['lower_bound'],
            intervals_percentile['upper_bound']
        )
        
        # Calibration assessment (if ground truth available)
        calibration = None
        if true_values is not None:
            calibration = self.assess_calibration(
                mc_analysis['mean_predictions'],
                mc_analysis['std_predictions'],
                true_values
            )
        
        report = {
            'summary': {
                'n_predictions': mc_analysis['n_predictions'],
                'n_mc_samples': mc_analysis['n_samples'],
                'confidence_level': self.confidence_level,
                'mean_uncertainty': mc_analysis['std_predictions'].mean(),
                'mean_interval_width': mc_analysis['mean_interval_width'],
                'tradeable_signals_pct': risk_metrics['tradeable_pct'],
                'mean_position_size_pct': risk_metrics['mean_position_size'] * 100
            },
            'monte_carlo_analysis': mc_analysis,
            'prediction_intervals': {
                'percentile_method': intervals_percentile,
                'gaussian_method': intervals_gaussian
            },
            'uncertainty_decomposition': uncertainty_decomp,
            'risk_metrics': risk_metrics,
            'calibration_assessment': calibration,
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        logger.info("Uncertainty report generated successfully")
        
        return report

class ConfidenceScorer:
    """
    Calculates confidence scores for trading decisions based on prediction uncertainty.
    """
    
    def __init__(self, 
                 base_threshold: float = 0.6,
                 uncertainty_penalty: float = 2.0,
                 interval_width_penalty: float = 1.5):
        """
        Initialize confidence scorer.
        
        Args:
            base_threshold: Base confidence threshold
            uncertainty_penalty: Penalty factor for high uncertainty
            interval_width_penalty: Penalty factor for wide intervals
        """
        self.base_threshold = base_threshold
        self.uncertainty_penalty = uncertainty_penalty
        self.interval_width_penalty = interval_width_penalty
        
    def calculate_confidence_score(self, 
                                 prediction: float,
                                 uncertainty: float,
                                 confidence_lower: float,
                                 confidence_upper: float) -> Dict[str, float]:
        """
        Calculate confidence score for a single prediction.
        
        Args:
            prediction: Point prediction
            uncertainty: Prediction uncertainty (std)
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound
            
        Returns:
            Confidence scoring results
        """
        # Relative uncertainty
        relative_uncertainty = uncertainty / (abs(prediction) + 1e-8)
        
        # Interval width as percentage
        interval_width = confidence_upper - confidence_lower
        interval_width_pct = interval_width / (abs(prediction) + 1e-8)
        
        # Base confidence (higher for larger predicted changes)
        base_confidence = min(abs(prediction) / 1000, 1.0)  # Normalize by typical price range
        
        # Uncertainty penalty
        uncertainty_factor = max(0.1, 1.0 - self.uncertainty_penalty * relative_uncertainty)
        
        # Interval width penalty
        width_factor = max(0.1, 1.0 - self.interval_width_penalty * interval_width_pct)
        
        # Combined confidence score
        confidence_score = base_confidence * uncertainty_factor * width_factor
        confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]
        
        # Trading recommendation
        if confidence_score >= self.base_threshold and interval_width_pct < 0.03:
            recommendation = 'TRADE'
        elif confidence_score >= 0.4:
            recommendation = 'MONITOR'
        else:
            recommendation = 'HOLD'
        
        return {
            'confidence_score': confidence_score,
            'relative_uncertainty': relative_uncertainty,
            'interval_width_pct': interval_width_pct,
            'uncertainty_factor': uncertainty_factor,
            'width_factor': width_factor,
            'recommendation': recommendation,
            'is_tradeable': recommendation == 'TRADE'
        }
    
    def batch_confidence_scores(self, 
                              predictions: np.ndarray,
                              uncertainties: np.ndarray,
                              confidence_lower: np.ndarray,
                              confidence_upper: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate confidence scores for batch of predictions.
        
        Args:
            predictions: Array of predictions
            uncertainties: Array of uncertainties
            confidence_lower: Array of lower bounds
            confidence_upper: Array of upper bounds
            
        Returns:
            Batch confidence scoring results
        """
        n_predictions = len(predictions)
        
        confidence_scores = np.zeros(n_predictions)
        relative_uncertainties = np.zeros(n_predictions)
        interval_widths_pct = np.zeros(n_predictions)
        recommendations = []
        
        for i in range(n_predictions):
            result = self.calculate_confidence_score(
                predictions[i],
                uncertainties[i],
                confidence_lower[i],
                confidence_upper[i]
            )
            
            confidence_scores[i] = result['confidence_score']
            relative_uncertainties[i] = result['relative_uncertainty']
            interval_widths_pct[i] = result['interval_width_pct']
            recommendations.append(result['recommendation'])
        
        # Summary statistics
        tradeable_mask = confidence_scores >= self.base_threshold
        tradeable_count = np.sum(tradeable_mask)
        
        return {
            'confidence_scores': confidence_scores,
            'relative_uncertainties': relative_uncertainties,
            'interval_widths_pct': interval_widths_pct,
            'recommendations': recommendations,
            'tradeable_mask': tradeable_mask,
            'tradeable_count': tradeable_count,
            'tradeable_pct': (tradeable_count / n_predictions) * 100,
            'mean_confidence': np.mean(confidence_scores),
            'high_confidence_count': np.sum(confidence_scores >= 0.8),
            'low_confidence_count': np.sum(confidence_scores < 0.4)
        } 