"""
Survival and Time-to-Event Modeling using lifelines
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import logging
from pathlib import Path
import joblib
from datetime import datetime

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
    from lifelines.utils import median_survival_times
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

logger = logging.getLogger(__name__)


class SurvivalModel:
    """Handles survival analysis using lifelines"""
    
    def __init__(self):
        self.km_fitter = None if LIFELINES_AVAILABLE else None
        self.cox_model = None if LIFELINES_AVAILABLE else None
        self.event_name = ""
        self.duration_col = ""
        self.is_trained = False
    
    def fit_kaplan_meier(
        self,
        durations: np.ndarray,
        event_observed: np.ndarray,
        event_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Fit Kaplan-Meier survival curve
        
        Args:
            durations: Time to event or censoring
            event_observed: Binary indicator of event occurrence
            event_name: Name of event
        
        Returns:
            Dictionary with survival statistics
        """
        
        if not LIFELINES_AVAILABLE:
            logger.warning("lifelines not available, skipping Kaplan-Meier")
            return {}
        
        try:
            self.km_fitter = KaplanMeierFitter()
            self.km_fitter.fit(durations, event_observed, label=event_name)
            self.event_name = event_name
            self.duration_col = "duration"
            self.is_trained = True
            
            # Get median survival time
            median_survival = self.km_fitter.median_survival_time_
            
            # Get survival at key timepoints
            survival_at_6m = self.km_fitter.predict(6) if len(self.km_fitter.survival_function_) > 6 else None
            survival_at_12m = self.km_fitter.predict(12) if len(self.km_fitter.survival_function_) > 12 else None
            
            results = {
                "model_type": "kaplan_meier",
                "event_name": event_name,
                "median_survival": float(median_survival) if median_survival is not None else None,
                "survival_at_6m": float(survival_at_6m) if survival_at_6m is not None else None,
                "survival_at_12m": float(survival_at_12m) if survival_at_12m is not None else None,
                "n_events": int(event_observed.sum()),
                "n_total": int(len(event_observed)),
                "event_rate": float(event_observed.sum() / len(event_observed))
            }
            
            logger.info(f"Kaplan-Meier fit for {event_name}: {results}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error fitting Kaplan-Meier: {e}")
            return {}
    
    def fit_cox(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        covariates: List[str],
        event_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Fit Cox Proportional Hazards model
        
        Args:
            df: DataFrame with duration, event, and covariates
            duration_col: Name of duration column
            event_col: Name of event column
            covariates: List of covariate column names
            event_name: Name of event
        
        Returns:
            Dictionary with Cox model results
        """
        
        if not LIFELINES_AVAILABLE:
            logger.warning("lifelines not available, skipping Cox model")
            return {}
        
        try:
            # Prepare data
            df_cox = df[[duration_col, event_col] + covariates].copy()
            df_cox = df_cox.dropna()
            
            # Fit Cox model
            self.cox_model = CoxPHFitter()
            self.cox_model.fit(df_cox, duration_col=duration_col, event_col=event_col)
            
            self.event_name = event_name
            self.duration_col = duration_col
            self.is_trained = True
            
            # Get concordance index
            concordance = self.cox_model.concordance_index_
            
            # Get hazard ratios
            hazard_ratios = np.exp(self.cox_model.params_)
            
            results = {
                "model_type": "cox_proportional_hazards",
                "event_name": event_name,
                "concordance_index": float(concordance),
                "n_observations": len(df_cox),
                "n_events": int(df_cox[event_col].sum()),
                "hazard_ratios": {col: float(hr) for col, hr in hazard_ratios.items()},
                "log_partial_likelihood": float(self.cox_model.log_likelihood_)
            }
            
            logger.info(f"Cox model fit for {event_name}: Concordance={concordance:.4f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error fitting Cox model: {e}")
            return {}
    
    def predict_survival_probability(
        self,
        time_point: int,
        covariates: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Predict survival probability at timepoint
        
        Args:
            time_point: Time point to predict
            covariates: Covariate values for Cox model
        
        Returns:
            Array of survival probabilities
        """
        
        if not self.is_trained:
            logger.warning("Model not trained")
            return np.array([])
        
        if self.km_fitter is not None:
            # Use Kaplan-Meier
            try:
                return np.array([self.km_fitter.predict(time_point)])
            except:
                return np.array([])
        
        elif self.cox_model is not None and covariates is not None:
            # Use Cox model
            try:
                return self.cox_model.predict_survival_function(covariates).iloc[time_point].values
            except:
                return np.array([])
        
        return np.array([])
    
    def save(self, filepath: Path) -> None:
        """Save survival model"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Survival model saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'SurvivalModel':
        """Load survival model"""
        model = joblib.load(filepath)
        logger.info(f"Survival model loaded from {filepath}")
        return model


class TransitionModel:
    """Models state transitions over time"""
    
    def __init__(self):
        self.transition_matrix: Optional[pd.DataFrame] = None
        self.states: List[str] = []
        self.is_trained = False
    
    def discover_states(self, status_column: pd.Series) -> List[str]:
        """Discover unique states from data"""
        self.states = sorted(status_column.unique().astype(str).tolist())
        logger.info(f"Discovered states: {self.states}")
        return self.states
    
    def build_transition_matrix(
        self,
        current_state: pd.Series,
        next_state: pd.Series,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Build transition matrix from current to next states
        
        Args:
            current_state: Current state observations
            next_state: Next state observations
            normalize: If True, normalize rows to probabilities
        
        Returns:
            Transition matrix
        """
        
        # Discover states if not already done
        if not self.states:
            all_states = pd.concat([current_state, next_state]).unique()
            self.states = sorted(all_states.astype(str).tolist())
        
        # Create contingency table
        transitions = pd.crosstab(
            current_state.astype(str),
            next_state.astype(str),
            margins=False
        )
        
        # Ensure all states are represented
        for state in self.states:
            if state not in transitions.index:
                transitions.loc[state] = 0
            if state not in transitions.columns:
                transitions[state] = 0
        
        # Reorder
        transitions = transitions.loc[self.states, self.states]
        
        # Normalize if requested
        if normalize:
            transitions = transitions.div(transitions.sum(axis=1), axis=0).fillna(0)
        
        self.transition_matrix = transitions
        self.is_trained = True
        
        logger.info(f"Transition matrix built:\n{transitions}")
        
        return transitions
    
    def get_transition_matrix(self, normalize: bool = True) -> Optional[pd.DataFrame]:
        """Get transition matrix"""
        if self.transition_matrix is None:
            return None
        
        if normalize:
            return self.transition_matrix.div(self.transition_matrix.sum(axis=1), axis=0).fillna(0)
        else:
            return self.transition_matrix
    
    def project_state_distribution(
        self,
        initial_distribution: Dict[str, float],
        n_steps: int
    ) -> List[Dict[str, float]]:
        """
        Project state distribution forward in time
        
        Args:
            initial_distribution: Initial state probabilities
            n_steps: Number of steps to project
        
        Returns:
            List of state distributions over time
        """
        
        if self.transition_matrix is None:
            logger.warning("Transition matrix not available")
            return []
        
        # Convert initial distribution to array
        state_dist = np.array([initial_distribution.get(s, 0) for s in self.states])
        
        projections = [dict(zip(self.states, state_dist))]
        
        # Project forward
        transition_array = self.transition_matrix.values
        
        for _ in range(n_steps):
            state_dist = state_dist @ transition_array
            projections.append(dict(zip(self.states, state_dist)))
        
        return projections
    
    def save(self, filepath: Path) -> None:
        """Save transition model"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Transition model saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'TransitionModel':
        """Load transition model"""
        model = joblib.load(filepath)
        logger.info(f"Transition model loaded from {filepath}")
        return model


class CompetingRisksModel:
    """Approximation of competing risks framework"""
    
    def __init__(self):
        self.risk_models: Dict[str, SurvivalModel] = {}
        self.event_names: List[str] = []
        self.is_trained = False
    
    def fit_competing_risks(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_type_col: str,
        event_types: List[str],
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fit competing risks model (simplified approach)
        
        For each competing event, fit a separate survival model
        treating other events as censoring
        
        Args:
            df: DataFrame with duration, event type, and covariates
            duration_col: Name of duration column
            event_type_col: Name of event type column
            event_types: List of competing event types
            covariates: Optional list of covariate names
        
        Returns:
            Dictionary of results for each event type
        """
        
        if not LIFELINES_AVAILABLE:
            logger.warning("lifelines not available")
            return {}
        
        results = {}
        self.event_names = event_types
        
        for event_type in event_types:
            try:
                # Create binary indicator for this event
                event_occurred = (df[event_type_col] == event_type).astype(int)
                
                # Fit Kaplan-Meier
                km_fitter = KaplanMeierFitter()
                km_fitter.fit(
                    df[duration_col],
                    event_occurred,
                    label=event_type
                )
                
                model = SurvivalModel()
                model.km_fitter = km_fitter
                model.event_name = event_type
                model.is_trained = True
                
                self.risk_models[event_type] = model
                
                results[event_type] = {
                    "event_type": event_type,
                    "median_survival": float(km_fitter.median_survival_time_) if km_fitter.median_survival_time_ is not None else None,
                    "n_events": int(event_occurred.sum()),
                    "event_rate": float(event_occurred.sum() / len(event_occurred))
                }
                
                logger.info(f"Competing risk model fit for {event_type}")
            
            except Exception as e:
                logger.error(f"Error fitting competing risk model for {event_type}: {e}")
        
        self.is_trained = len(self.risk_models) > 0
        
        return results
    
    def get_cumulative_incidence(self, time_point: int) -> Dict[str, float]:
        """Get cumulative incidence for each competing event"""
        
        results = {}
        
        for event_type, model in self.risk_models.items():
            try:
                if model.km_fitter is not None:
                    # Cumulative incidence = 1 - survival probability
                    survival_prob = model.km_fitter.predict(time_point)
                    cumulative_incidence = 1 - float(survival_prob)
                    results[event_type] = cumulative_incidence
            except:
                pass
        
        return results
    
    def save(self, filepath: Path) -> None:
        """Save competing risks model"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Competing risks model saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'CompetingRisksModel':
        """Load competing risks model"""
        model = joblib.load(filepath)
        logger.info(f"Competing risks model loaded from {filepath}")
        return model
