"""
Model Training Orchestration - PHASE 4
Handles baseline and improved model training for all targets
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    BaselineModel, CatBoostBinaryModel, CatBoostMulticlassModel,
    ModelMetrics, ModelComparator, ImbalanceHandler, 
    ModelCalibrator, CalibrationAnalyzer
)
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Orchestrate model training for all targets"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.comparator = ModelComparator()
        self.metrics_report: List[Dict[str, Any]] = []
    
    def prepare_train_val_features(
        self,
        train_split: pd.DataFrame,
        val_split: pd.DataFrame,
        feature_df: pd.DataFrame,
        target_names: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Prepare train/val feature sets for modeling
        
        Returns:
            (X_train, y_train, X_val, y_val, split_metadata)
        """
        
        logger.info("Preparing train/val feature matrices...")
        
        # Get indices from splits
        train_indices = train_split.index
        val_indices = val_split.index
        
        # Align features with splits
        X_train = feature_df.loc[train_indices].copy()
        X_val = feature_df.loc[val_indices].copy()
        
        # Drop any NaN features
        X_train = X_train.dropna(axis=1, how='all')
        X_val = X_val[X_train.columns]
        
        logger.info(f"Train features: {X_train.shape}")
        logger.info(f"Validation features: {X_val.shape}")
        
        # Check for target columns in features (leakage detection)
        target_in_features = [t for t in target_names if t in X_train.columns]
        if target_in_features:
            logger.warning(f"TARGET LEAKAGE: {target_in_features} found in features!")
            # Remove them
            X_train = X_train.drop(columns=target_in_features)
            X_val = X_val.drop(columns=target_in_features)
        
        metadata = {
            "X_train_shape": X_train.shape,
            "X_val_shape": X_val.shape,
            "feature_count": X_train.shape[1],
            "feature_names": X_train.columns.tolist()
        }
        
        return X_train, X_val, metadata
    
    def train_baseline_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_name: str
    ) -> BaselineModel:
        """Train baseline model"""
        
        logger.info(f"\nTraining BASELINE model for {target_name}...")
        
        # Create and train
        baseline = BaselineModel(strategy="majority")
        baseline.train(X_train, y_train)
        
        self.models.setdefault(target_name, {})['baseline'] = baseline
        
        logger.info(f"Baseline trained for {target_name}")
        
        return baseline
    
    def train_catboost_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str,
        is_multiclass: bool = False,
        **kwargs
    ) -> Optional[Any]:
        """
        Train CatBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            target_name: Name of target
            is_multiclass: If True, use multiclass model
            **kwargs: CatBoost hyperparameters
        
        Returns:
            Trained model or None if fails
        """
        
        logger.info(f"\nTraining CATBOOST model for {target_name} (multiclass={is_multiclass})...")
        
        try:
            # Apply class weights for binary if imbalanced
            if not is_multiclass:
                class_dist = ImbalanceHandler.report_class_distribution(y_train, target_name)
                
                # Initialize model
                model = CatBoostBinaryModel(
                    model_name=f"CatBoost_{target_name}",
                    **kwargs
                )
                
                # Train
                start_time = time.time()
                model.train(X_train, y_train, X_val, y_val)
                elapsed = time.time() - start_time
                
            else:
                # Multiclass
                model = CatBoostMulticlassModel(
                    model_name=f"CatBoost_{target_name}",
                    **kwargs
                )
                
                start_time = time.time()
                model.train(X_train, y_train, X_val, y_val)
                elapsed = time.time() - start_time
            
            self.models.setdefault(target_name, {})['catboost'] = model
            
            logger.info(f"CatBoost trained for {target_name} in {elapsed:.2f}s")
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to train CatBoost for {target_name}: {e}")
            return None
    
    def evaluate_models_for_target(
        self,
        target_name: str,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        threshold: float = 0.5
    ) -> List[ModelMetrics]:
        """
        Evaluate all models trained for a target
        
        Args:
            target_name: Name of target
            X_val: Validation features
            y_val: Validation target
            threshold: Decision threshold for binary
        
        Returns:
            List of ModelMetrics objects
        """
        
        logger.info(f"\nEvaluating models for {target_name}...")
        
        metrics_list = []
        
        if target_name not in self.models:
            logger.warning(f"No models found for {target_name}")
            return metrics_list
        
        for model_type, model in self.models[target_name].items():
            try:
                metrics = model.evaluate(X_val, y_val, target_name, threshold)
                metrics_list.append(metrics)
                
                # Log key metrics
                if metrics.roc_auc is not None:
                    logger.info(f"{model_type}: ROC-AUC={metrics.roc_auc:.4f}, F1={metrics.f1:.4f}")
                elif metrics.accuracy is not None:
                    logger.info(f"{model_type}: Accuracy={metrics.accuracy:.4f}, Macro-F1={metrics.macro_f1:.4f}")
                
                # Add to comparison
                metrics_dict = metrics.to_dict()
                self.comparator.add_result(
                    target=target_name,
                    model_name=model.model_name,
                    model_type=model.model_type,
                    metrics={k: v for k, v in metrics_dict.items() if k not in ['target', 'model_type']}
                )
            
            except Exception as e:
                logger.error(f"Error evaluating {model_type} for {target_name}: {e}", exc_info=True)
        
        return metrics_list
    
    def calibrate_models_for_target(
        self,
        target_name: str,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Calibrate probability predictions for binary targets
        
        Args:
            target_name: Name of target
            X_val: Validation features
            y_val: Validation target
        
        Returns:
            Calibration results
        """
        
        logger.info(f"\nCalibrating models for {target_name}...")
        
        if target_name not in self.models:
            return {}
        
        calibrator = ModelCalibrator(method="sigmoid")
        calibration_results = {}
        
        for model_type, model in self.models[target_name].items():
            try:
                if model_type == 'baseline':
                    continue  # Skip baseline
                
                calib_model, metrics = calibrator.calibrate_binary(
                    model.model,
                    X_val,
                    y_val,
                    X_val,
                    y_val,
                    target_name
                )
                
                calibration_results[model_type] = metrics
                logger.info(f"{model_type} calibration: {metrics}")
            
            except Exception as e:
                logger.warning(f"Could not calibrate {model_type} for {target_name}: {e}")
        
        return calibration_results
    
    def save_models(self, output_dir: Path) -> None:
        """Save all trained models"""
        
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for target_name, target_models in self.models.items():
            for model_type, model in target_models.items():
                filepath = models_dir / f"{target_name}_{model_type}.joblib"
                model.save(filepath)
                logger.info(f"Saved {model_type} for {target_name}")
    
    def save_metrics_report(self, output_dir: Path) -> None:
        """Save model metrics comparison"""
        
        import numpy as np
        
        # Save as JSON
        json_path = output_dir / "model_metrics.json"
        results = []
        
        for result in self.comparator.results:
            result_copy = result.copy()
            result_copy['timestamp'] = str(result_copy['timestamp'])
            
            # Convert numpy types to native Python types
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(i) for i in obj]
                else:
                    return obj
            
            result_copy = convert_types(result_copy)
            results.append(result_copy)
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Metrics saved to {json_path}")
        
        # Export comparison table
        csv_path = output_dir / "model_comparison.csv"
        self.comparator.export_to_csv(csv_path)
        logger.info(f"Comparison table saved to {csv_path}")


def train_phase_4(
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    feature_df: pd.DataFrame,
    settings=None
) -> Tuple[Dict[str, Any], ModelComparator]:
    """
    Execute Phase 4: Baseline & Improved Model Training
    
    Args:
        train_split: Training split from time-aware split
        val_split: Validation split from time-aware split
        feature_df: Feature matrix for all data
        settings: Configuration settings
    
    Returns:
        (models_dict, model_comparator)
    """
    
    settings = settings or get_settings()
    trainer = ModelTrainer(settings)
    
    # Identify target columns
    targets = [
        'next_3m_delinquency_flag',
        'next_6m_delinquency_flag',
        'next_12m_default_flag',
        'next_12m_prepayment_flag',
        'next_state',
        'exception_required',
        'exception_type'
    ]
    
    # Filter to available targets
    available_targets = [t for t in targets if t in train_split.columns]
    
    if not available_targets:
        logger.warning("No target columns found in data!")
        return trainer.models, trainer.comparator
    
    logger.info(f"\nAvailable targets: {available_targets}")
    
    # Prepare features
    X_train, X_val, split_metadata = trainer.prepare_train_val_features(
        train_split, val_split, feature_df, available_targets
    )
    
    # Get targets
    targets_data = {}
    for target in available_targets:
        if target in train_split.columns and target in val_split.columns:
            targets_data[target] = {
                'y_train': train_split[target],
                'y_val': val_split[target]
            }
    
    # Train models for each target
    for target_name, target_info in targets_data.items():
        y_train = target_info['y_train']
        y_val = target_info['y_val']
        
        # Skip if no positive examples
        if y_train.nunique() < 2:
            logger.warning(f"Target {target_name} has no variation, skipping")
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING MODELS FOR: {target_name}")
        logger.info(f"Train distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"{'='*70}")
        
        # Train baseline
        trainer.train_baseline_model(X_train, y_train, target_name)
        
        # Train CatBoost
        is_multiclass = y_train.nunique() > 2
        
        if not is_multiclass:
            trainer.train_catboost_model(
                X_train, y_train, X_val, y_val,
                target_name,
                is_multiclass=False,
                iterations=settings.get_model_config().get('catboost_iterations', 500)
            )
        else:
            trainer.train_catboost_model(
                X_train, y_train, X_val, y_val,
                target_name,
                is_multiclass=True,
                iterations=settings.get_model_config().get('catboost_iterations', 500)
            )
        
        # Evaluate
        metrics = trainer.evaluate_models_for_target(target_name, X_val, y_val)
        
        # Calibrate if binary
        if not is_multiclass:
            trainer.calibrate_models_for_target(target_name, X_val, y_val)
    
    # Save models and metrics
    trainer.save_models(settings.outputs_dir)
    trainer.save_metrics_report(settings.outputs_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 4 COMPLETE: MODEL TRAINING")
    logger.info(f"Models trained: {len(trainer.models)}")
    logger.info(f"{'='*70}")
    
    return trainer.models, trainer.comparator
