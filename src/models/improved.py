"""
CatBoost and XGBoost models for binary and multi-class classification
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from catboost import CatBoostClassifier, Pool, cv
from sklearn.preprocessing import LabelEncoder
from .base import BaseModel, ModelMetrics

logger = logging.getLogger(__name__)


class CatBoostBinaryModel(BaseModel):
    """CatBoost for binary classification"""
    
    def __init__(self, model_name: str = "CatBoost_Binary", **kwargs):
        """
        Initialize CatBoost binary model
        
        Args:
            model_name: Model identifier
            **kwargs: CatBoost parameters
        """
        super().__init__(model_name, "catboost")
        
        # Default parameters
        default_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 7,
            "verbose": False,
            "task_type": "CPU",
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_state": 42,
            "bootstrap_type": "Bayesian",
            "od_type": "Iter",
            "od_wait": 20
        }
        
        # Override with provided kwargs
        default_params.update(kwargs)
        self.training_config = default_params
        
        self.model = CatBoostClassifier(**default_params)
        self.categorical_features: List[int] = []
        self.cat_feature_names: List[str] = []
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> None:
        """
        Train CatBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        self.feature_names = X_train.columns.tolist()
        
        # Identify categorical features
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        self.cat_feature_names = cat_cols
        
        # Create CatBoost Pool
        train_pool = Pool(
            X_train,
            label=y_train,
            cat_features=cat_cols if cat_cols else None
        )
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(
                X_val,
                label=y_val,
                cat_features=cat_cols if cat_cols else None
            )
        
        # Train
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            verbose=self.training_config.get('verbose', False)
        )
        
        self.is_trained = True
        logger.info(f"CatBoost model trained. Features: {len(self.feature_names)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importances = self.model.get_feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n).reset_index(drop=True)


class CatBoostMulticlassModel(BaseModel):
    """CatBoost for multi-class classification (e.g., next_state prediction)"""
    
    def __init__(self, model_name: str = "CatBoost_Multiclass", **kwargs):
        """
        Initialize CatBoost multiclass model
        
        Args:
            model_name: Model identifier
            **kwargs: CatBoost parameters
        """
        super().__init__(model_name, "catboost")
        
        default_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 7,
            "verbose": False,
            "task_type": "CPU",
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "random_state": 42,
            "bootstrap_type": "Bayesian",
            "od_type": "Iter",
            "od_wait": 20
        }
        
        default_params.update(kwargs)
        self.training_config = default_params
        
        self.model = CatBoostClassifier(**default_params)
        self.cat_feature_names: List[str] = []
        self.classes_ = None
        self.le = LabelEncoder()
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> None:
        """Train multiclass model"""
        self.feature_names = X_train.columns.tolist()
        
        # Encode labels if necessary
        if y_train.dtype == 'object':
            y_train_encoded = self.le.fit_transform(y_train)
            self.classes_ = self.le.classes_
        else:
            y_train_encoded = y_train
            self.classes_ = np.unique(y_train)
        
        # Identify categorical features
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        self.cat_feature_names = cat_cols
        
        # Create pools
        train_pool = Pool(
            X_train,
            label=y_train_encoded,
            cat_features=cat_cols if cat_cols else None
        )
        
        eval_set = None
        if X_val is not None and y_val is not None:
            if self.le.classes_ is not None:
                y_val_encoded = self.le.transform(y_val)
            else:
                y_val_encoded = y_val
            
            eval_set = Pool(
                X_val,
                label=y_val_encoded,
                cat_features=cat_cols if cat_cols else None
            )
        
        # Train
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            verbose=self.training_config.get('verbose', False)
        )
        
        self.is_trained = True
        logger.info(f"CatBoost multiclass model trained. Classes: {len(self.classes_)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        pred = self.model.predict(X).astype(int)
        
        # Decode if necessary
        if self.le.classes_ is not None:
            pred = self.le.inverse_transform(pred)
        
        return pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importances = self.model.get_feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n).reset_index(drop=True)


class XGBoostBinaryModel(BaseModel):
    """XGBoost for binary classification (fallback if available)"""
    
    def __init__(self, model_name: str = "XGBoost_Binary", **kwargs):
        """Initialize XGBoost binary model"""
        super().__init__(model_name, "xgboost")
        
        try:
            import xgboost as xgb
            
            default_params = {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 7,
                "random_state": 42,
                "eval_metric": "logloss",
                "use_label_encoder": False,
                "verbose": False
            }
            
            default_params.update(kwargs)
            self.training_config = default_params
            
            self.model = xgb.XGBClassifier(**default_params)
            self.xgb_available = True
            
        except ImportError:
            logger.warning("XGBoost not available, using CatBoost fallback")
            self.xgb_available = False
            self.model = CatBoostBinaryModel(**kwargs).model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> None:
        """Train XGBoost model"""
        self.feature_names = X_train.columns.tolist()
        
        if self.xgb_available:
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=self.training_config.get('verbose', False)
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            # Use CatBoost
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info("XGBoost model trained")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if self.xgb_available:
            import xgboost as xgb
            importances = self.model.feature_importances_
        else:
            importances = self.model.get_feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n).reset_index(drop=True)
