"""
Class imbalance handling strategies
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImbalanceHandler:
    """Handle class imbalance in training data"""
    
    @staticmethod
    def calculate_class_weights(y: pd.Series) -> dict:
        """
        Calculate class weights for imbalanced data
        
        Args:
            y: Target series
        
        Returns:
            Dictionary mapping class to weight
        """
        
        n_samples = len(y)
        n_classes = len(y.unique())
        
        weights = {}
        
        for class_val in y.unique():
            count = (y == class_val).sum()
            weight = n_samples / (n_classes * count)
            weights[int(class_val)] = float(weight)
        
        logger.info(f"Class weights: {weights}")
        
        return weights
    
    @staticmethod
    def apply_class_weights_to_model(model, class_weights: dict) -> None:
        """
        Apply class weights to CatBoost or XGBoost model
        
        Args:
            model: Model object
            class_weights: Dictionary of class weights
        """
        
        # For CatBoost
        if hasattr(model, 'set_params'):
            if 'class_weights' in model.get_params():
                model.set_params(class_weights=class_weights)
                logger.info("Class weights applied to model")
    
    @staticmethod
    def get_sample_weights(y: pd.Series) -> np.ndarray:
        """
        Get sample weights for class imbalance
        
        Args:
            y: Target series
        
        Returns:
            Array of sample weights
        """
        
        class_weights = ImbalanceHandler.calculate_class_weights(y)
        
        sample_weights = np.array([class_weights[val] for val in y])
        
        # Normalize to sum to n_samples
        sample_weights = sample_weights * len(y) / sample_weights.sum()
        
        return sample_weights
    
    @staticmethod
    def report_class_distribution(y: pd.Series, name: str = "Dataset") -> dict:
        """
        Report class distribution
        
        Args:
            y: Target series
            name: Name of dataset
        
        Returns:
            Distribution statistics
        """
        
        distribution = y.value_counts().to_dict()
        total = len(y)
        
        report = {
            "name": name,
            "total_samples": int(total),
            "distribution": {str(k): int(v) for k, v in distribution.items()},
            "percentages": {str(k): float(v / total * 100) for k, v in distribution.items()}
        }
        
        # Imbalance ratio
        if len(distribution) == 2:
            counts = sorted(distribution.values())
            report["imbalance_ratio"] = float(counts[1] / counts[0])
        
        logger.info(f"{name} class distribution: {report['distribution']}")
        
        return report
    
    @staticmethod
    def optimize_threshold_for_recall(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_recall: float = 0.9
    ) -> Tuple[float, dict]:
        """
        Find optimal threshold to achieve target recall
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            target_recall: Target recall (e.g., 0.9 = 90%)
        
        Returns:
            (optimal_threshold, metrics_at_threshold)
        """
        
        from sklearn.metrics import precision_recall_curve
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find threshold closest to target recall
        idx = np.argmin(np.abs(recalls - target_recall))
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        optimal_threshold = float(optimal_threshold)
        
        # Get metrics at this threshold
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            "threshold": optimal_threshold,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        logger.info(f"Optimal threshold for recall {target_recall}: {optimal_threshold:.4f}")
        
        return optimal_threshold, metrics
    
    @staticmethod
    def optimize_threshold_for_precision(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_precision: float = 0.9
    ) -> Tuple[float, dict]:
        """
        Find optimal threshold to achieve target precision
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            target_precision: Target precision (e.g., 0.9 = 90%)
        
        Returns:
            (optimal_threshold, metrics_at_threshold)
        """
        
        from sklearn.metrics import precision_recall_curve
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find threshold closest to target precision
        idx = np.argmin(np.abs(precisions - target_precision))
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        optimal_threshold = float(optimal_threshold)
        
        # Get metrics at this threshold
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            "threshold": optimal_threshold,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        logger.info(f"Optimal threshold for precision {target_precision}: {optimal_threshold:.4f}")
        
        return optimal_threshold, metrics
