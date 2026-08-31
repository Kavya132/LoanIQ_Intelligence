"""
Base model classes and interfaces for Loan Performance Intelligence Engine
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, brier_score_loss,
    confusion_matrix, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics"""
    target: str
    model_type: str
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    brier_score: Optional[float] = None
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    threshold: float = 0.5
    class_distribution: Optional[Dict[str, int]] = None
    note: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        cm = self.confusion_matrix
        if cm is not None:
            cm = [list(row) for row in cm]
        
        return {
            "target": self.target,
            "model_type": self.model_type,
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "pr_auc": float(self.pr_auc) if self.pr_auc is not None else None,
            "f1": float(self.f1) if self.f1 is not None else None,
            "precision": float(self.precision) if self.precision is not None else None,
            "recall": float(self.recall) if self.recall is not None else None,
            "brier_score": float(self.brier_score) if self.brier_score is not None else None,
            "accuracy": float(self.accuracy) if self.accuracy is not None else None,
            "macro_f1": float(self.macro_f1) if self.macro_f1 is not None else None,
            "confusion_matrix": cm,
            "threshold": float(self.threshold),
            "class_distribution": self.class_distribution,
            "note": self.note
        }


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names: List[str] = []
        self.training_config: Dict[str, Any] = {}
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (binary class or class labels)"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    def evaluate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str = "default",
        threshold: float = 0.5
    ) -> ModelMetrics:
        """Evaluate model on validation set"""
        
        y_pred_proba = self.predict_proba(X_val)
        
        # Handle multi-class vs binary
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 2:
            # Multi-class: use accuracy and macro-F1
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            metrics = ModelMetrics(
                target=target_name,
                model_type=self.model_type,
                accuracy=float(accuracy_score(y_val, y_pred)),
                macro_f1=float(f1_score(y_val, y_pred, average='macro', zero_division=0))
            )
            
            # Add per-class metrics if possible
            cm = confusion_matrix(y_val, y_pred)
            metrics.confusion_matrix = cm
            
            return metrics
        
        # Binary classification
        if y_pred_proba.ndim == 2:
            y_proba_pos = y_pred_proba[:, 1]
        else:
            y_proba_pos = y_pred_proba
        
        y_pred = (y_proba_pos >= threshold).astype(int)
        
        # Calculate metrics
        try:
            roc_auc = float(roc_auc_score(y_val, y_proba_pos))
        except:
            roc_auc = None
        
        try:
            precision, recall, _ = precision_recall_curve(y_val, y_proba_pos)
            pr_auc = float(auc(recall, precision))
        except:
            pr_auc = None
        
        try:
            f1 = float(f1_score(y_val, y_pred, zero_division=0))
        except:
            f1 = None
        
        try:
            prec = float(precision_score(y_val, y_pred, zero_division=0))
        except:
            prec = None
        
        try:
            rec = float(recall_score(y_val, y_pred, zero_division=0))
        except:
            rec = None
        
        try:
            bs = float(brier_score_loss(y_val, y_proba_pos))
        except:
            bs = None
        
        try:
            acc = float(accuracy_score(y_val, y_pred))
        except:
            acc = None
        
        cm = confusion_matrix(y_val, y_pred)
        
        # Class distribution
        class_dist = {
            "0": int((y_val == 0).sum()),
            "1": int((y_val == 1).sum()),
            "total": int(len(y_val))
        }
        
        metrics = ModelMetrics(
            target=target_name,
            model_type=self.model_type,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            f1=f1,
            precision=prec,
            recall=rec,
            brier_score=bs,
            accuracy=acc,
            confusion_matrix=cm,
            threshold=threshold,
            class_distribution=class_dist
        )
        
        return metrics
    
    def save(self, filepath: Path) -> None:
        """Save model to disk"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'BaseModel':
        """Load model from disk"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance (to be overridden by subclasses)"""
        raise NotImplementedError("Subclass must implement get_feature_importance")


class BaselineModel(BaseModel):
    """Baseline model - majority class or simple rules"""
    
    def __init__(self, strategy: str = "majority"):
        """
        Initialize baseline model
        
        Args:
            strategy: 'majority' (always predict majority class) or 'random' (random prediction)
        """
        super().__init__("Baseline", "baseline")
        self.strategy = strategy
        self.majority_class = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train baseline"""
        self.feature_names = X_train.columns.tolist()
        self.majority_class = y_train.mode()[0]
        self.is_trained = True
        self.training_config = {"strategy": self.strategy}
        logger.info(f"Baseline trained. Majority class: {self.majority_class}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict majority class"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if self.strategy == "majority":
            return np.full(len(X), self.majority_class)
        else:
            return np.random.randint(0, 2, len(X))
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities - all to majority class"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        proba = np.zeros((len(X), 2))
        if self.majority_class == 1:
            proba[:, 1] = 1.0
        else:
            proba[:, 0] = 1.0
        
        return proba
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Baseline has no feature importance"""
        return pd.DataFrame({
            "feature": ["N/A"],
            "importance": [0.0]
        })
