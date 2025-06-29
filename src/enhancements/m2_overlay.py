"""
M2 Money Supply Overlay for Trading Strategy
Implements optimized M2 regime filtering with backtest-validated parameters
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.m2_data_provider import M2DataProvider

logger = logging.getLogger(__name__)

class M2TradingOverlay:
    """
    M2 Money Supply overlay for trading strategy enhancement
    Uses backtest-optimized parameters for regime detection and position sizing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize M2 trading overlay with optimized parameters
        
        Args:
            config: Configuration dictionary with M2 parameters
        """
        # Load optimized parameters from backtest results
        self.config = config or self._get_default_config()
        
        # Initialize M2 data provider
        self.m2_provider = M2DataProvider()
        
        # Optimization results from backtest
        self.expansion_threshold = self.config.get('expansion_threshold', 8.0)  # % M2 YoY growth
        self.contraction_threshold = self.config.get('contraction_threshold', 2.0)  # % M2 YoY growth
        self.expansion_multiplier = self.config.get('expansion_multiplier', 1.1)  # Position boost
        self.contraction_multiplier = self.config.get('contraction_multiplier', 0.4)  # Position reduction
        self.stable_multiplier = self.config.get('stable_multiplier', 1.0)  # Normal position
        
        # State tracking
        self.current_regime = 'unknown'
        self.current_multiplier = 1.0
        self.last_update = None
        self.confidence = 0.0
        
        # Performance tracking
        self.regime_history = []
        self.multiplier_history = []
        
        logger.info(f"M2TradingOverlay initialized with optimized parameters:")
        logger.info(f"  Expansion: >{self.expansion_threshold}% (×{self.expansion_multiplier})")
        logger.info(f"  Contraction: <{self.contraction_threshold}% (×{self.contraction_multiplier})")
        logger.info(f"  Stable: {self.contraction_threshold}-{self.expansion_threshold}% (×{self.stable_multiplier})")
    
    def _get_default_config(self) -> Dict:
        """Get default optimized configuration from backtest results"""
        return {
            # Optimized parameters from backtest
            'expansion_threshold': 8.0,     # M2 YoY growth > 8% = expansion
            'contraction_threshold': 2.0,   # M2 YoY growth < 2% = contraction
            'expansion_multiplier': 1.1,    # 10% position boost during expansion
            'contraction_multiplier': 0.4,  # 60% position reduction during contraction
            'stable_multiplier': 1.0,       # Normal position during stable periods
            
            # Data management
            'update_frequency_hours': 24,   # Update M2 data daily
            'data_staleness_threshold': 14, # Alert if M2 data older than 14 days
            
            # Risk management
            'max_multiplier': 1.5,          # Cap position boosts
            'min_multiplier': 0.2,          # Floor for position reductions
            'regime_confidence_threshold': 0.3,  # Minimum confidence for regime changes
            
            # Logging and monitoring
            'log_regime_changes': True,
            'track_performance': True
        }
    
    def initialize_m2_data(self, start_date: str = None) -> bool:
        """
        Initialize M2 data for the overlay
        
        Args:
            start_date: Start date for M2 data (defaults to 2 years ago)
            
        Returns:
            True if successful, False otherwise
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        logger.info(f"Initializing M2 data from {start_date}")
        
        if self.m2_provider.fetch_m2_data(start_date=start_date):
            # Get initial regime
            self._update_current_regime()
            logger.info(f"✅ M2 overlay initialized - Current regime: {self.current_regime}")
            return True
        else:
            logger.error("❌ Failed to initialize M2 data")
            return False
    
    def _update_current_regime(self):
        """Update current M2 regime and position multiplier"""
        try:
            # Get current M2 metrics
            m2_metrics = self.m2_provider.get_current_m2_regime()
            
            if m2_metrics['regime'] == 'unknown':
                logger.warning("M2 regime unknown - using stable multiplier")
                self.current_regime = 'stable'
                self.current_multiplier = self.stable_multiplier
                self.confidence = 0.0
                return
            
            # Extract metrics
            growth_yoy = m2_metrics['growth_yoy']
            confidence = m2_metrics['confidence']
            
            # Determine regime based on optimized thresholds
            if growth_yoy > self.expansion_threshold:
                new_regime = 'expansion'
                new_multiplier = self.expansion_multiplier
            elif growth_yoy < self.contraction_threshold:
                new_regime = 'contraction'
                new_multiplier = self.contraction_multiplier
            else:
                new_regime = 'stable'
                new_multiplier = self.stable_multiplier
            
            # Apply confidence threshold for regime changes
            if (self.current_regime != 'unknown' and 
                new_regime != self.current_regime and 
                confidence < self.config['regime_confidence_threshold']):
                
                logger.info(f"M2 regime change pending (low confidence: {confidence:.2f})")
                return  # Keep current regime
            
            # Update regime if changed
            if new_regime != self.current_regime:
                old_regime = self.current_regime
                self.current_regime = new_regime
                self.current_multiplier = new_multiplier
                self.confidence = confidence
                self.last_update = datetime.now()
                
                # Log regime change
                if self.config['log_regime_changes']:
                    logger.info(f"🔄 M2 regime change: {old_regime} → {new_regime}")
                    logger.info(f"   M2 YoY Growth: {growth_yoy:.1f}%")
                    logger.info(f"   Position Multiplier: {self.current_multiplier:.1f}x")
                    logger.info(f"   Confidence: {confidence:.2f}")
                
                # Track history
                if self.config['track_performance']:
                    self.regime_history.append({
                        'timestamp': self.last_update,
                        'regime': new_regime,
                        'growth_yoy': growth_yoy,
                        'confidence': confidence
                    })
                    self.multiplier_history.append({
                        'timestamp': self.last_update,
                        'multiplier': new_multiplier,
                        'regime': new_regime
                    })
            else:
                # Update confidence even if regime unchanged
                self.confidence = confidence
                self.last_update = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating M2 regime: {e}")
            # Fallback to stable regime on error
            self.current_regime = 'stable'
            self.current_multiplier = self.stable_multiplier
            self.confidence = 0.0
    
    def get_position_multiplier(self, force_update: bool = False) -> Dict[str, float]:
        """
        Get current position multiplier based on M2 regime
        
        Args:
            force_update: Force update of M2 data and regime
            
        Returns:
            Dictionary with multiplier information
        """
        # Check if we need to update M2 data
        update_needed = (
            force_update or
            self.last_update is None or
            (datetime.now() - self.last_update).total_seconds() > 
            self.config['update_frequency_hours'] * 3600
        )
        
        if update_needed:
            self._update_current_regime()
        
        # Apply safety bounds
        safe_multiplier = max(
            self.config['min_multiplier'],
            min(self.config['max_multiplier'], self.current_multiplier)
        )
        
        return {
            'multiplier': safe_multiplier,
            'raw_multiplier': self.current_multiplier,
            'regime': self.current_regime,
            'confidence': self.confidence,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'reasoning': self._get_regime_reasoning()
        }
    
    def _get_regime_reasoning(self) -> str:
        """Get human-readable reasoning for current regime"""
        try:
            m2_metrics = self.m2_provider.get_current_m2_regime()
            growth = m2_metrics.get('growth_yoy', 0)
            
            if self.current_regime == 'expansion':
                return f"M2 expansion ({growth:.1f}% > {self.expansion_threshold}%) → Boost positions {self.expansion_multiplier:.1f}x"
            elif self.current_regime == 'contraction':
                return f"M2 contraction ({growth:.1f}% < {self.contraction_threshold}%) → Reduce positions {self.contraction_multiplier:.1f}x"
            else:
                return f"M2 stable ({growth:.1f}%) → Normal positions {self.stable_multiplier:.1f}x"
        except:
            return f"M2 {self.current_regime} → {self.current_multiplier:.1f}x position sizing"
    
    def enhance_trading_signal(self, base_signal: float, base_confidence: float) -> Dict[str, float]:
        """
        Enhance trading signal with M2 overlay
        
        Args:
            base_signal: Base trading signal (-1 to 1)
            base_confidence: Base signal confidence (0 to 1)
            
        Returns:
            Enhanced signal information
        """
        # Get current M2 position multiplier
        m2_info = self.get_position_multiplier()
        position_multiplier = m2_info['multiplier']
        
        # Calculate enhanced signal
        enhanced_signal = base_signal * position_multiplier
        
        # Combine confidences (conservative approach)
        combined_confidence = (base_confidence + m2_info['confidence']) / 2
        
        # Risk adjustment based on regime uncertainty
        if m2_info['confidence'] < 0.5:
            # Reduce position size during uncertain periods
            uncertainty_penalty = 1 - (0.5 - m2_info['confidence'])
            enhanced_signal *= uncertainty_penalty
        
        return {
            'enhanced_signal': enhanced_signal,
            'base_signal': base_signal,
            'position_multiplier': position_multiplier,
            'combined_confidence': combined_confidence,
            'regime': m2_info['regime'],
            'reasoning': m2_info['reasoning'],
            'risk_adjustment': uncertainty_penalty if m2_info['confidence'] < 0.5 else 1.0
        }
    
    def get_regime_status(self) -> Dict:
        """Get comprehensive M2 regime status for monitoring"""
        try:
            m2_metrics = self.m2_provider.get_current_m2_regime()
            multiplier_info = self.get_position_multiplier()
            
            # Data freshness check
            is_fresh = self.m2_provider.is_data_fresh(self.config['data_staleness_threshold'])
            
            return {
                'regime': self.current_regime,
                'position_multiplier': self.current_multiplier,
                'confidence': self.confidence,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                
                # M2 metrics
                'growth_yoy': m2_metrics.get('growth_yoy', 0),
                'trend_12w': m2_metrics.get('trend_12w', 0),
                'acceleration': m2_metrics.get('acceleration', 0),
                
                # Thresholds
                'expansion_threshold': self.expansion_threshold,
                'contraction_threshold': self.contraction_threshold,
                
                # Data quality
                'data_fresh': is_fresh,
                'data_age_days': (datetime.now() - self.last_update).days if self.last_update else None,
                
                # Performance tracking
                'regime_changes_count': len(self.regime_history),
                'reasoning': self._get_regime_reasoning()
            }
            
        except Exception as e:
            logger.error(f"Error getting regime status: {e}")
            return {
                'regime': 'error',
                'position_multiplier': 1.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of M2 overlay"""
        if not self.regime_history:
            return {'error': 'No performance data available'}
        
        try:
            # Regime distribution
            regimes = [r['regime'] for r in self.regime_history]
            regime_counts = pd.Series(regimes).value_counts()
            
            # Average confidence by regime
            regime_df = pd.DataFrame(self.regime_history)
            avg_confidence = regime_df.groupby('regime')['confidence'].mean()
            
            # Recent activity
            recent_changes = [r for r in self.regime_history 
                            if (datetime.now() - r['timestamp']).days <= 30]
            
            return {
                'total_regime_changes': len(self.regime_history),
                'regime_distribution': regime_counts.to_dict(),
                'average_confidence_by_regime': avg_confidence.to_dict(),
                'recent_changes_30d': len(recent_changes),
                'current_regime': self.current_regime,
                'current_confidence': self.confidence,
                'tracking_start': self.regime_history[0]['timestamp'].isoformat(),
                'last_change': self.regime_history[-1]['timestamp'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {'error': str(e)}

class M2OverlayManager:
    """Manager class for M2 overlay integration with existing trading systems"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize M2 overlay manager"""
        self.overlay = M2TradingOverlay(config)
        self.enabled = False
        self.initialization_attempts = 0
        self.max_init_attempts = 3
        
    def initialize(self, start_date: str = None) -> bool:
        """
        Initialize M2 overlay with retry logic
        
        Args:
            start_date: Start date for M2 data
            
        Returns:
            True if successful, False otherwise
        """
        self.initialization_attempts += 1
        
        logger.info(f"Initializing M2 overlay (attempt {self.initialization_attempts})")
        
        if self.overlay.initialize_m2_data(start_date):
            self.enabled = True
            logger.info("✅ M2 overlay enabled and ready")
            return True
        else:
            if self.initialization_attempts < self.max_init_attempts:
                logger.warning(f"M2 initialization failed, will retry ({self.initialization_attempts}/{self.max_init_attempts})")
            else:
                logger.error("M2 initialization failed after maximum attempts - disabling overlay")
                self.enabled = False
            return False
    
    def process_trading_signal(self, base_signal: float, base_confidence: float) -> Dict:
        """
        Process trading signal through M2 overlay if enabled
        
        Args:
            base_signal: Base trading signal
            base_confidence: Base signal confidence
            
        Returns:
            Processed signal information
        """
        if not self.enabled:
            # Return unmodified signal if M2 overlay disabled
            return {
                'enhanced_signal': base_signal,
                'base_signal': base_signal,
                'position_multiplier': 1.0,
                'combined_confidence': base_confidence,
                'regime': 'm2_disabled',
                'reasoning': 'M2 overlay disabled or not initialized'
            }
        
        try:
            return self.overlay.enhance_trading_signal(base_signal, base_confidence)
        except Exception as e:
            logger.error(f"Error processing signal through M2 overlay: {e}")
            # Return unmodified signal on error
            return {
                'enhanced_signal': base_signal,
                'base_signal': base_signal,
                'position_multiplier': 1.0,
                'combined_confidence': base_confidence,
                'regime': 'error',
                'reasoning': f'M2 overlay error: {str(e)}'
            }
    
    def get_status(self) -> Dict:
        """Get comprehensive M2 overlay status"""
        return {
            'enabled': self.enabled,
            'initialization_attempts': self.initialization_attempts,
            'regime_status': self.overlay.get_regime_status() if self.enabled else None,
            'performance_summary': self.overlay.get_performance_summary() if self.enabled else None
        }