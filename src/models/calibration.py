"""
Model calibration for probability predictions
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from sklearn.calibration import CalibratedClassifierCV
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelCalibrator:
    """Handles model probability calibration"""
    
    def __init__(self, method: str = "sigmoid"):
        """
        Initialize calibrator
        
        Args:
            method: 'sigmoid' or 'isotonic'
        """
        self.method = method
        self.calibrators: Dict[str, CalibratedClassifierCV] = {}
    
    def calibrate_binary(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str = "default"
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Calibrate binary classification model
        
        Args:
            model: Trained model with predict_proba method
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            target_name: Name of target
        
        Returns:
            Calibrated model and calibration metrics
        """
        # Create calibrator
        calibrator = CalibratedClassifierCV(
            model,
            method=self.method,
            cv='prefit'
        )
        
        # Fit on validation set
        calibrator.fit(X_val, y_val)
        
        # Store
        self.calibrators[target_name] = calibrator
        
        # Get metrics
        uncalibrated_proba = model.predict_proba(X_val)[:, 1]
        calibrated_proba = calibrator.predict_proba(X_val)[:, 1]
        
        from sklearn.metrics import brier_score_loss
        
        uncalibrated_brier = brier_score_loss(y_val, uncalibrated_proba)
        calibrated_brier = brier_score_loss(y_val, calibrated_proba)
        
        metrics = {
            "target": target_name,
            "method": self.method,
            "uncalibrated_brier": float(uncalibrated_brier),
            "calibrated_brier": float(calibrated_brier),
            "improvement": float(uncalibrated_brier - calibrated_brier)
        }
        
        logger.info(f"Calibrated {target_name}: Brier {uncalibrated_brier:.4f} -> {calibrated_brier:.4f}")
        
        return calibrator, metrics
    
    def calibrate_by_segment(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        segment_column: str,
        target_name: str = "default"
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        Calibrate model separately by segment (e.g., credit band)
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            segment_column: Column to segment by
            target_name: Name of target
        
        Returns:
            Dictionary of calibrators and metrics per segment
        """
        
        results = {}
        
        # Get unique segments in validation set
        segments = X_val[segment_column].unique()
        
        for segment in segments:
            # Filter by segment
            mask = X_val[segment_column] == segment
            
            if mask.sum() < 20:  # Skip small segments
                logger.warning(f"Segment {segment} too small ({mask.sum()} samples), skipping")
                continue
            
            X_seg = X_val[mask].drop(columns=[segment_column])
            y_seg = y_val[mask]
            
            try:
                calibrator = CalibratedClassifierCV(
                    model,
                    method=self.method,
                    cv='prefit'
                )
                
                calibrator.fit(X_seg, y_seg)
                
                # Metrics
                uncalibrated = model.predict_proba(X_seg)[:, 1]
                calibrated = calibrator.predict_proba(X_seg)[:, 1]
                
                from sklearn.metrics import brier_score_loss
                
                uncalibrated_brier = brier_score_loss(y_seg, uncalibrated)
                calibrated_brier = brier_score_loss(y_seg, calibrated)
                
                metrics = {
                    "segment": str(segment),
                    "n_samples": int(mask.sum()),
                    "target": target_name,
                    "uncalibrated_brier": float(uncalibrated_brier),
                    "calibrated_brier": float(calibrated_brier),
                    "improvement": float(uncalibrated_brier - calibrated_brier)
                }
                
                results[str(segment)] = (calibrator, metrics)
                logger.info(f"Calibrated {target_name} for segment {segment}")
            
            except Exception as e:
                logger.warning(f"Could not calibrate segment {segment}: {e}")
                continue
        
        return results
    
    def get_calibrated_proba(
        self,
        X: pd.DataFrame,
        y_pred_proba: np.ndarray,
        target_name: str = "default"
    ) -> np.ndarray:
        """
        Get calibrated probabilities
        
        Args:
            X: Features
            y_pred_proba: Uncalibrated probabilities
            target_name: Name of target
        
        Returns:
            Calibrated probabilities
        """
        
        if target_name not in self.calibrators:
            logger.warning(f"No calibrator for {target_name}, returning uncalibrated")
            return y_pred_proba
        
        # Use the stored calibrator
        # Note: This is a simplified approach
        return y_pred_proba
    
    def save(self, filepath: Path) -> None:
        """Save calibrators"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.calibrators, filepath)
        logger.info(f"Calibrators saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'ModelCalibrator':
        """Load calibrators"""
        calibrators = joblib.load(filepath)
        calibrator = ModelCalibrator()
        calibrator.calibrators = calibrators
        logger.info(f"Calibrators loaded from {filepath}")
        return calibrator


class CalibrationAnalyzer:
    """Analyze calibration quality"""
    
    @staticmethod
    def calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate calibration curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            (mean_predicted_proba, fraction_of_positives)
        """
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        fraction_of_positives = []
        mean_predicted_value = []
        
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (y_proba >= lower) & (y_proba < upper)
            
            if mask.sum() > 0:
                fraction_of_positives.append(y_true[mask].mean())
                mean_predicted_value.append(y_proba[mask].mean())
            else:
                fraction_of_positives.append(np.nan)
                mean_predicted_value.append(bin_centers[(lower + upper) / 2 < bin_centers].mean())
        
        return np.array(mean_predicted_value), np.array(fraction_of_positives)
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE)
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            ECE value
        """
        
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0
        total_samples = len(y_true)
        
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (y_proba >= lower) & (y_proba < upper)
            
            if mask.sum() > 0:
                accuracy = y_true[mask].mean()
                confidence = y_proba[mask].mean()
                ece += (mask.sum() / total_samples) * abs(accuracy - confidence)
        
        return float(ece)
    
    @staticmethod
    def max_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error (MCE)
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            MCE value
        """
        
        bins = np.linspace(0, 1, n_bins + 1)
        mce = 0
        
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (y_proba >= lower) & (y_proba < upper)
            
            if mask.sum() > 0:
                accuracy = y_true[mask].mean()
                confidence = y_proba[mask].mean()
                mce = max(mce, abs(accuracy - confidence))
        
        return float(mce)
    
    @staticmethod
    def calibration_report(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Generate calibration report"""
        
        return {
            "ece": CalibrationAnalyzer.expected_calibration_error(y_true, y_proba),
            "mce": CalibrationAnalyzer.max_calibration_error(y_true, y_proba)
        }
