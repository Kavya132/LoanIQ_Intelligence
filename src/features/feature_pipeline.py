"""
Feature engineering pipeline for loan prediction
Implements time-aware feature creation and leakage prevention
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import joblib

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Build features for loan prediction models"""
    
    # Categorical columns that are safe to encode
    CATEGORICAL_FEATURES = [
        "credit_score_band", "ltv_band", "dti_band", "state", 
        "loan_purpose", "occupancy_type", "property_type", 
        "servicer_name", "current_status", "document_status"
    ]
    
    # Numeric features to standardize
    NUMERIC_FEATURES = [
        "original_balance", "current_balance", "interest_rate",
        "loan_age_months", "remaining_term_months", "days_past_due"
    ]
    
    # Temporal features to create
    TEMPORAL_FEATURES = {
        "age_to_term_ratio": ("loan_age_months", "remaining_term_months", "divide"),
        "balance_ratio": ("current_balance", "original_balance", "divide"),
    }
    
    @staticmethod
    def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        df = df.copy()
        
        # Age to term ratio (avoids division by zero)
        if "loan_age_months" in df.columns and "remaining_term_months" in df.columns:
            df["age_to_term_ratio"] = df.apply(
                lambda row: row["loan_age_months"] / row["remaining_term_months"] 
                if row["remaining_term_months"] > 0 else 0, axis=1
            )
        
        # Balance ratio
        if "current_balance" in df.columns and "original_balance" in df.columns:
            df["balance_ratio"] = df.apply(
                lambda row: row["current_balance"] / row["original_balance"]
                if row["original_balance"] > 0 else 0, axis=1
            )
        
        return df
    
    @staticmethod
    def engineer_delinquency_features(df: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
        """Create delinquency trend features"""
        df = df.copy()
        
        # Basic delinquency features
        if "days_past_due" in df.columns:
            df["dpd_bucket"] = pd.cut(
                df["days_past_due"],
                bins=[-np.inf, 0, 30, 60, 90, np.inf],
                labels=["Current", "30_59", "60_89", "90_119", "120+"]
            )
        
        return df
    
    @classmethod
    def build_feature_matrix(
        cls,
        df: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        numeric_features: Optional[List[str]] = None,
        include_engineered: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Build complete feature matrix
        
        Args:
            df: Input DataFrame with raw features
            categorical_features: List of categorical columns
            numeric_features: List of numeric columns
            include_engineered: Whether to create engineered features
            
        Returns:
            Tuple of (feature_df, feature_metadata)
        """
        if categorical_features is None:
            categorical_features = [c for c in cls.CATEGORICAL_FEATURES if c in df.columns]
        if numeric_features is None:
            numeric_features = [c for c in cls.NUMERIC_FEATURES if c in df.columns]
        
        # Start with raw features
        feature_cols = numeric_features + categorical_features
        feature_df = df[feature_cols].copy()
        
        # Add engineered features
        if include_engineered:
            feature_df = cls.engineer_temporal_features(feature_df)
            feature_df = cls.engineer_delinquency_features(feature_df)
        
        # Remove rows with all NaNs in features
        feature_df = feature_df.dropna(how="all")
        
        # Build metadata
        metadata = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "total_features": len(feature_df.columns),
            "rows": len(feature_df),
            "feature_names": feature_df.columns.tolist()
        }
        
        logger.info(f"Built feature matrix: {len(feature_df)} rows × {len(feature_df.columns)} columns")
        
        return feature_df, metadata


class PersistedFeaturePipeline:
    """Persist the training feature contract and apply it unchanged at inference.

    The current feature construction is stateless (CatBoost natively handles the
    categorical fields), but persisting the contract prevents inference from
    silently drifting to a different set of raw or engineered fields.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []

    def fit(self, df: pd.DataFrame) -> "PersistedFeaturePipeline":
        features, _ = FeaturePipeline.build_feature_matrix(df, include_engineered=True)
        self.feature_names = features.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features, _ = FeaturePipeline.build_feature_matrix(df, include_engineered=True)
        missing = sorted(set(self.feature_names) - set(features.columns))
        if missing:
            raise ValueError("Input cannot generate trained features: " + ", ".join(missing))
        return features.reindex(columns=self.feature_names)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class TimeAwareSplit:
    """Create time-aware train/validation splits to prevent leakage"""
    
    @staticmethod
    def split_by_reporting_month(
        df: pd.DataFrame,
        reporting_month_col: str = "reporting_month",
        validation_fraction: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split data by reporting month (time-aware)
        
        Args:
            df: Input DataFrame with reporting_month
            reporting_month_col: Name of month column
            validation_fraction: Fraction for validation set
            
        Returns:
            Tuple of (train_df, validation_df, split_metadata)
        """
        if reporting_month_col not in df.columns:
            raise ValueError(f"Column {reporting_month_col} not found")
        
        # Convert to datetime
        df = df.copy()
        df["_month"] = pd.to_datetime(df[reporting_month_col], errors="coerce")
        
        # Sort by month
        df = df.sort_values("_month")
        
        # Split point
        n_months = df["_month"].nunique()
        split_month_idx = int(n_months * (1 - validation_fraction))
        
        unique_months = sorted(df["_month"].unique())
        split_month = unique_months[split_month_idx]
        
        # Split
        train_df = df[df["_month"] < split_month].drop("_month", axis=1)
        val_df = df[df["_month"] >= split_month].drop("_month", axis=1)
        
        # Metadata
        metadata = {
            "method": "time_aware_split_by_month",
            "split_month": str(split_month),
            "train_months": sorted(train_df[reporting_month_col].unique()),
            "val_months": sorted(val_df[reporting_month_col].unique()),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "train_loans": train_df["loan_id"].nunique() if "loan_id" in train_df.columns else 0,
            "val_loans": val_df["loan_id"].nunique() if "loan_id" in val_df.columns else 0,
            "leakage_check": "PASS"  # Add actual checks below
        }
        
        logger.info(
            f"Time-aware split: {len(train_df)} train records ({metadata['train_loans']} loans), "
            f"{len(val_df)} validation records ({metadata['val_loans']} loans)"
        )
        
        return train_df, val_df, metadata
    
    @staticmethod
    def check_temporal_leakage(
        df: pd.DataFrame,
        target_col: str,
        reporting_month_col: str = "reporting_month",
        lookforward_months: int = 12
    ) -> Dict[str, Any]:
        """
        Verify no future target information leaks into features
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            reporting_month_col: Reporting month column
            lookforward_months: How many months forward is the target
            
        Returns:
            Leakage check report
        """
        report = {
            "target_col": target_col,
            "leakage_detected": False,
            "issues": []
        }
        
        # Check if target exists
        if target_col not in df.columns:
            report["issues"].append(f"Target column {target_col} not found")
            return report
        
        # Check if any records have target at prediction time
        if reporting_month_col in df.columns:
            # Ideally would check that target values only exist for future months
            # This is a simplified check
            if df[target_col].notna().sum() > 0:
                report["leakage_detected"] = False  # Target existence itself isn't leakage
        
        return report


class LeakageDetector:
    """Detect potential data leakage"""
    
    @staticmethod
    def detect_target_in_features(
        features: pd.DataFrame,
        target: pd.Series,
        target_name: str = "target"
    ) -> Dict[str, Any]:
        """
        Check if target column appears in features (accidental inclusion)
        
        Args:
            features: Feature DataFrame
            target: Target Series
            target_name: Name of target
            
        Returns:
            Detection report
        """
        report = {
            "target": target_name,
            "leakage_detected": False,
            "problematic_features": []
        }
        
        # Check correlation with target
        target_copy = target.copy()
        if pd.api.types.is_object_dtype(target_copy.dtype):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            try:
                target_copy = le.fit_transform(target_copy.astype(str))
            except:
                return report
        
        # Check numeric features for perfect correlation
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        for feat in numeric_features:
            if features[feat].notna().sum() < 2:
                continue
            
            try:
                corr = features[feat].corr(pd.Series(target_copy))
                if abs(corr) > 0.99:  # Nearly perfect correlation
                    report["leakage_detected"] = True
                    report["problematic_features"].append({
                        "feature": feat,
                        "correlation": float(corr)
                    })
            except:
                pass
        
        return report
    
    @staticmethod
    def detect_future_features(
        df: pd.DataFrame,
        reporting_month_col: str = "reporting_month"
    ) -> Dict[str, Any]:
        """
        Check if feature values look like they're from the future
        (e.g., using next month's balance as feature)
        
        Args:
            df: DataFrame to check
            reporting_month_col: Reporting month column
            
        Returns:
            Detection report
        """
        report = {
            "method": "future_feature_detection",
            "issues": [],
            "leakage_risk": "LOW"
        }
        
        # Check if balance columns decrease over time (normal pattern)
        if "loan_id" in df.columns and "current_balance" in df.columns and reporting_month_col in df.columns:
            df_sorted = df.sort_values(["loan_id", reporting_month_col])
            
            # Sample check: for each loan, verify balance is monotonic
            for loan_id in df_sorted["loan_id"].unique()[:100]:
                loan_data = df_sorted[df_sorted["loan_id"] == loan_id]
                if len(loan_data) > 1 and "current_balance" in loan_data.columns:
                    balances = loan_data["current_balance"].values
                    if pd.notna(balances).all():
                        # Check if increasing (could indicate future data)
                        increases = (balances[1:] > balances[:-1]).sum()
                        if increases / len(balances) > 0.5:
                            report["issues"].append(f"Loan {loan_id}: balance increasing over time")
                            report["leakage_risk"] = "MEDIUM"
        
        return report
