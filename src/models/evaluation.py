"""
Model evaluation and comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare multiple models for the same target"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        target: str,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        training_time: Optional[float] = None
    ) -> None:
        """Add model evaluation result"""
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "target": target,
            "model_name": model_name,
            "model_type": model_type,
            "metrics": metrics,
            "training_time": training_time
        }
        
        self.results.append(result)
    
    def get_comparison_table(self, target: Optional[str] = None) -> pd.DataFrame:
        """
        Get comparison table for models
        
        Args:
            target: Filter by specific target (optional)
        
        Returns:
            DataFrame with model comparison
        """
        
        filtered_results = self.results
        
        if target:
            filtered_results = [r for r in self.results if r['target'] == target]
        
        if not filtered_results:
            return pd.DataFrame()
        
        # Build comparison table
        rows = []
        
        for result in filtered_results:
            row = {
                "Model": result['model_name'],
                "Type": result['model_type'],
                "Target": result['target'],
                "ROC-AUC": result['metrics'].get('roc_auc'),
                "PR-AUC": result['metrics'].get('pr_auc'),
                "F1": result['metrics'].get('f1'),
                "Recall": result['metrics'].get('recall'),
                "Precision": result['metrics'].get('precision'),
                "Brier": result['metrics'].get('brier_score'),
                "Accuracy": result['metrics'].get('accuracy'),
                "Training Time (s)": result.get('training_time')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Round metrics
        metric_cols = ['ROC-AUC', 'PR-AUC', 'F1', 'Recall', 'Precision', 'Brier', 'Accuracy']
        for col in metric_cols:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        return df
    
    def get_best_model(self, target: str, metric: str = 'roc_auc') -> Dict[str, Any]:
        """
        Get best model for a target based on metric
        
        Args:
            target: Target name
            metric: Metric to rank by (default: roc_auc)
        
        Returns:
            Best model result
        """
        
        filtered = [r for r in self.results if r['target'] == target]
        
        if not filtered:
            return None
        
        # Sort by metric
        best = max(
            filtered,
            key=lambda x: x['metrics'].get(metric, -1)
        )
        
        logger.info(f"Best model for {target}: {best['model_name']} ({metric}={best['metrics'].get(metric)})")
        
        return best
    
    def save_comparison(self, filepath: Path) -> None:
        """Save comparison to file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Comparison saved to {filepath}")
    
    def export_to_csv(self, filepath: Path, target: Optional[str] = None) -> None:
        """Export comparison to CSV"""
        df = self.get_comparison_table(target)
        df.to_csv(filepath, index=False)
        logger.info(f"Comparison exported to {filepath}")


class ErrorAnalysis:
    """Analyze model errors"""
    
    @staticmethod
    def false_positives(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[pd.DataFrame] = None,
        sample_size: int = 20
    ) -> pd.DataFrame:
        """
        Get false positive examples
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Features (optional)
            sample_size: Number of examples to return
        
        Returns:
            DataFrame with false positive examples
        """
        
        fp_mask = (y_pred == 1) & (y_true == 0)
        fp_indices = np.where(fp_mask)[0]
        
        if len(fp_indices) == 0:
            logger.warning("No false positives found")
            return pd.DataFrame()
        
        # Sample
        sample_indices = np.random.choice(
            fp_indices,
            size=min(sample_size, len(fp_indices)),
            replace=False
        )
        
        if X is not None:
            return X.iloc[sample_indices].copy()
        else:
            return pd.DataFrame({
                'index': sample_indices,
                'y_true': y_true[sample_indices],
                'y_pred': y_pred[sample_indices]
            })
    
    @staticmethod
    def false_negatives(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[pd.DataFrame] = None,
        sample_size: int = 20
    ) -> pd.DataFrame:
        """
        Get false negative examples
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Features (optional)
            sample_size: Number of examples to return
        
        Returns:
            DataFrame with false negative examples
        """
        
        fn_mask = (y_pred == 0) & (y_true == 1)
        fn_indices = np.where(fn_mask)[0]
        
        if len(fn_indices) == 0:
            logger.warning("No false negatives found")
            return pd.DataFrame()
        
        # Sample
        sample_indices = np.random.choice(
            fn_indices,
            size=min(sample_size, len(fn_indices)),
            replace=False
        )
        
        if X is not None:
            return X.iloc[sample_indices].copy()
        else:
            return pd.DataFrame({
                'index': sample_indices,
                'y_true': y_true[sample_indices],
                'y_pred': y_pred[sample_indices]
            })
    
    @staticmethod
    def error_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate error analysis report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
        
        Returns:
            Error analysis report
        """
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total = len(y_true)
        accuracy = (tp + tn) / total
        error_rate = (fp + fn) / total
        
        report = {
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "total": int(total)
            },
            "rates": {
                "accuracy": float(accuracy),
                "error_rate": float(error_rate),
                "fp_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
                "fn_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
                "tp_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                "tn_rate": float(tn / (tn + fp)) if (tn + fp) > 0 else 0
            }
        }
        
        # If probabilities available, add confidence analysis
        if y_proba is not None:
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            
            fp_proba = y_proba[(y_pred == 1) & (y_true == 0)]
            fn_proba = y_proba[(y_pred == 0) & (y_true == 1)]
            tp_proba = y_proba[(y_pred == 1) & (y_true == 1)]
            tn_proba = y_proba[(y_pred == 0) & (y_true == 0)]
            
            report["confidence_analysis"] = {
                "fp_mean_confidence": float(fp_proba.mean()) if len(fp_proba) > 0 else None,
                "fn_mean_confidence": float(fn_proba.mean()) if len(fn_proba) > 0 else None,
                "tp_mean_confidence": float(tp_proba.mean()) if len(tp_proba) > 0 else None,
                "tn_mean_confidence": float(tn_proba.mean()) if len(tn_proba) > 0 else None
            }
        
        return report
