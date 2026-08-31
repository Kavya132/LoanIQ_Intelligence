"""
Phase 5: Survival and Transition Model Training
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional
import json
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.survival.survival_model import SurvivalModel, TransitionModel, CompetingRisksModel
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SurvivalPhaseTrainer:
    """Orchestrates Phase 5: Survival and Transition Model Training"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.survival_models: Dict[str, SurvivalModel] = {}
        self.transition_models: Dict[str, TransitionModel] = {}
        self.competing_risks: Optional[CompetingRisksModel] = None
        self.results = []
    
    def prepare_survival_data(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare durations and event indicators"""
        
        # Ensure no missing values
        valid_mask = df[[duration_col, event_col]].notna().all(axis=1)
        
        durations = df.loc[valid_mask, duration_col].values
        events = df.loc[valid_mask, event_col].astype(int).values
        
        # Handle negative or zero durations
        durations = np.maximum(durations, 0.1)
        
        logger.info(f"Survival data: {len(durations)} records, {events.sum()} events")
        
        return durations, events
    
    def train_kaplan_meier(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        event_name: str,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train Kaplan-Meier model for a specific event
        
        Args:
            df: DataFrame with survival data
            duration_col: Name of duration column
            event_col: Name of event indicator column
            event_name: Name of event for identification
            save_path: Path to save model
        
        Returns:
            Results dictionary
        """
        
        logger.info(f"\nTraining Kaplan-Meier model for {event_name}...")
        
        try:
            durations, events = self.prepare_survival_data(df, duration_col, event_col)
            
            # Train model
            model = SurvivalModel()
            results = model.fit_kaplan_meier(durations, events, event_name)
            
            self.survival_models[event_name] = model
            
            if save_path:
                model.save(save_path)
            
            return results
        
        except Exception as e:
            logger.error(f"Error training Kaplan-Meier for {event_name}: {e}")
            return {}
    
    def train_transition_model(
        self,
        df: pd.DataFrame,
        current_state_col: str,
        next_state_col: str,
        model_name: str = "state_transition",
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train transition model from state sequences
        
        Args:
            df: DataFrame with state columns
            current_state_col: Name of current state column
            next_state_col: Name of next state column
            model_name: Name of model
            save_path: Path to save model
        
        Returns:
            Results dictionary
        """
        
        logger.info(f"\nTraining Transition model: {model_name}...")
        
        try:
            model = TransitionModel()
            
            # Discover states
            model.discover_states(df[current_state_col])
            
            # Build transition matrix
            transition_matrix = model.build_transition_matrix(
                df[current_state_col],
                df[next_state_col],
                normalize=True
            )
            
            self.transition_models[model_name] = model
            
            if save_path:
                model.save(save_path)
            
            # Convert to JSON-serializable format
            results = {
                "model_name": model_name,
                "states": model.states,
                "transition_matrix": transition_matrix.to_dict(),
                "n_records": len(df),
                "n_states": len(model.states)
            }
            
            return results
        
        except Exception as e:
            logger.error(f"Error training transition model: {e}")
            return {}
    
    def project_future_states(
        self,
        model_name: str,
        current_distribution: Dict[str, float],
        n_periods: int = 6
    ) -> Dict[str, Any]:
        """
        Project state distribution forward in time
        
        Args:
            model_name: Name of transition model
            current_distribution: Current state probabilities
            n_periods: Number of periods to project
        
        Returns:
            Projection results
        """
        
        if model_name not in self.transition_models:
            logger.warning(f"Model {model_name} not found")
            return {}
        
        model = self.transition_models[model_name]
        
        try:
            projections = model.project_state_distribution(
                current_distribution,
                n_periods
            )
            
            # Convert to JSON-serializable format
            results = {
                "model_name": model_name,
                "n_periods": n_periods,
                "projections": [
                    {str(k): float(v) for k, v in proj.items()}
                    for proj in projections
                ]
            }
            
            logger.info(f"Projected states for {model_name} over {n_periods} periods")
            
            return results
        
        except Exception as e:
            logger.error(f"Error projecting states: {e}")
            return {}
    
    def train_competing_risks(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_type_col: str,
        event_types: list,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train competing risks model
        
        Args:
            df: DataFrame with event data
            duration_col: Name of duration column
            event_type_col: Name of event type column
            event_types: List of possible event types
            save_path: Path to save model
        
        Returns:
            Results dictionary
        """
        
        logger.info(f"\nTraining Competing Risks model for events: {event_types}...")
        
        try:
            model = CompetingRisksModel()
            
            results = model.fit_competing_risks(
                df,
                duration_col,
                event_type_col,
                event_types
            )
            
            self.competing_risks = model
            
            if save_path:
                model.save(save_path)
            
            return {
                "model_type": "competing_risks",
                "event_types": event_types,
                "results": results
            }
        
        except Exception as e:
            logger.error(f"Error training competing risks model: {e}")
            return {}
    
    def save_models(self, output_dir: Path) -> None:
        """Save all survival models"""
        
        models_dir = output_dir / "models" / "survival"
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Save Kaplan-Meier models
        for event_name, model in self.survival_models.items():
            filepath = models_dir / f"km_{event_name}.joblib"
            model.save(filepath)
        
        # Save transition models
        for model_name, model in self.transition_models.items():
            filepath = models_dir / f"transition_{model_name}.joblib"
            model.save(filepath)
        
        # Save competing risks
        if self.competing_risks:
            filepath = models_dir / "competing_risks.joblib"
            self.competing_risks.save(filepath)
        
        logger.info(f"Survival models saved to {models_dir}")
    
    def save_results(self, output_dir: Path) -> None:
        """Save survival analysis results"""
        
        results_path = output_dir / "survival_analysis_results.json"
        
        # Prepare results
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "kaplan_meier_models": list(self.survival_models.keys()),
            "transition_models": list(self.transition_models.keys()),
            "competing_risks_available": self.competing_risks is not None,
            "model_results": self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Survival results saved to {results_path}")


def train_phase_5(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    settings=None
) -> Tuple[Dict[str, SurvivalModel], Dict[str, TransitionModel], Optional[CompetingRisksModel]]:
    """
    Execute Phase 5: Survival and Transition Model Training
    
    Args:
        train_df: Training data with targets
        val_df: Validation data with targets
        settings: Configuration settings
    
    Returns:
        (survival_models, transition_models, competing_risks_model)
    """
    
    settings = settings or get_settings()
    trainer = SurvivalPhaseTrainer(settings)
    
    logger.info("\n" + "="*70)
    logger.info("PHASE 5: SURVIVAL & TRANSITION MODELING")
    logger.info("="*70)
    
    # Combine train and val for model training
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # 1. Kaplan-Meier for default events
    if 'next_12m_default_flag' in full_df.columns and 'loan_age_months' in full_df.columns:
        km_results = trainer.train_kaplan_meier(
            full_df,
            duration_col='loan_age_months',
            event_col='next_12m_default_flag',
            event_name='default_12m',
            save_path=settings.outputs_dir / "models" / "survival" / "km_default.joblib"
        )
        if km_results:
            trainer.results.append(km_results)
            logger.info(f"Kaplan-Meier for Default: {km_results}")
    
    # 2. Kaplan-Meier for prepayment events
    if 'next_12m_prepayment_flag' in full_df.columns and 'loan_age_months' in full_df.columns:
        km_results = trainer.train_kaplan_meier(
            full_df,
            duration_col='loan_age_months',
            event_col='next_12m_prepayment_flag',
            event_name='prepayment_12m',
            save_path=settings.outputs_dir / "models" / "survival" / "km_prepayment.joblib"
        )
        if km_results:
            trainer.results.append(km_results)
            logger.info(f"Kaplan-Meier for Prepayment: {km_results}")
    
    # 3. Transition Model based on next_state
    if 'current_status' in full_df.columns and 'next_state' in full_df.columns:
        transition_results = trainer.train_transition_model(
            full_df,
            current_state_col='current_status',
            next_state_col='next_state',
            model_name='loan_status_transitions',
            save_path=settings.outputs_dir / "models" / "survival" / "transition_status.joblib"
        )
        if transition_results:
            trainer.results.append(transition_results)
            logger.info(f"Transition model trained with {transition_results.get('n_states', 0)} states")
    
    # 4. Competing Risks (simplified: default vs prepayment)
    if 'next_state' in full_df.columns and 'loan_age_months' in full_df.columns:
        # Create event type indicator
        competing_events = []
        if 'next_12m_default_flag' in full_df.columns and 'next_12m_prepayment_flag' in full_df.columns:
            # Map states to events
            event_type = 'none'
            for idx, row in full_df.iterrows():
                if row.get('next_12m_default_flag', False):
                    event_type = 'default'
                elif row.get('next_12m_prepayment_flag', False):
                    event_type = 'prepayment'
                competing_events.append(event_type)
            
            full_df['competing_event'] = competing_events
            
            competing_results = trainer.train_competing_risks(
                full_df,
                duration_col='loan_age_months',
                event_type_col='competing_event',
                event_types=['default', 'prepayment', 'none'],
                save_path=settings.outputs_dir / "models" / "survival" / "competing_risks.joblib"
            )
            if competing_results:
                trainer.results.append(competing_results)
                logger.info(f"Competing risks model trained")
    
    # Save all models and results
    trainer.save_models(settings.outputs_dir)
    trainer.save_results(settings.outputs_dir)
    
    logger.info("\n" + "="*70)
    logger.info(f"PHASE 5 COMPLETE: {len(trainer.survival_models)} KM models, {len(trainer.transition_models)} transition models")
    logger.info("="*70)
    
    return trainer.survival_models, trainer.transition_models, trainer.competing_risks
