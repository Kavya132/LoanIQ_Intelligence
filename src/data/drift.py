"""
Train/test data drift detection
Implements drift monitoring for Task 1: Data Intelligence
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.stats import ks_2samp, chi2_contingency
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect data drift between train and test datasets
    """
    
    @staticmethod
    def calculate_psi(
        series_train: pd.Series,
        series_test: pd.Series,
        buckets: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            series_train: Training data
            series_test: Test data
            buckets: Number of buckets for binning
            
        Returns:
            PSI value (0 = no drift, >0.25 = significant drift)
        """
        # Remove nulls
        train = series_train.dropna()
        test = series_test.dropna()
        
        if len(train) == 0 or len(test) == 0:
            return 0.0
        
        # Bin based on training data
        try:
            # For numeric data, use quantile binning
            if pd.api.types.is_numeric_dtype(train.dtype):
                bins = pd.qcut(train, q=buckets, duplicates="drop")
                bin_edges = bins.cat.categories
                
                train_pcts = pd.cut(train, bins=bin_edges, include_lowest=True).value_counts(normalize=True).sort_index()
                test_pcts = pd.cut(test, bins=bin_edges, include_lowest=True, right=False).value_counts(normalize=True).sort_index()
            else:
                # For categorical, use value counts
                train_pcts = train.value_counts(normalize=True).head(buckets)
                test_pcts = test.value_counts(normalize=True)[train_pcts.index]
                
                # Handle missing categories
                missing_cats = set(train_pcts.index) - set(test_pcts.index)
                for cat in missing_cats:
                    test_pcts[cat] = 0.0001  # Small epsilon to avoid log(0)
            
            # Ensure valid percentages
            train_pcts = train_pcts + 0.0001
            test_pcts = test_pcts + 0.0001
            
            # Calculate PSI
            psi = np.sum((test_pcts - train_pcts) * np.log(test_pcts / train_pcts))
            return float(psi)
        
        except Exception as e:
            logger.debug(f"PSI calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_ks_statistic(
        series_train: pd.Series,
        series_test: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov test statistic
        
        Args:
            series_train: Training data
            series_test: Test data
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        try:
            train = pd.to_numeric(series_train, errors="coerce").dropna()
            test = pd.to_numeric(series_test, errors="coerce").dropna()
            
            if len(train) < 2 or len(test) < 2:
                return 0.0, 1.0
            
            ks_stat, p_value = ks_2samp(train, test)
            return float(ks_stat), float(p_value)
        
        except Exception as e:
            logger.debug(f"KS test failed: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def calculate_chi_square(
        series_train: pd.Series,
        series_test: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate chi-square statistic for categorical data
        
        Args:
            series_train: Training data
            series_test: Test data
            
        Returns:
            Tuple of (chi-square statistic, p-value)
        """
        try:
            # Create contingency table
            train_counts = series_train.value_counts()
            test_counts = series_test.value_counts()
            
            # Align indices
            all_cats = set(train_counts.index) | set(test_counts.index)
            train_counts = train_counts.reindex(all_cats, fill_value=1)  # Add 1 to avoid zero cells
            test_counts = test_counts.reindex(all_cats, fill_value=1)
            
            # Chi-square test
            chi2, p_value, _, _ = chi2_contingency(np.array([train_counts, test_counts]))
            return float(chi2), float(p_value)
        
        except Exception as e:
            logger.debug(f"Chi-square test failed: {e}")
            return 0.0, 1.0
    
    @classmethod
    def detect_column_drift(
        cls,
        col_name: str,
        series_train: pd.Series,
        series_test: pd.Series,
        psi_threshold: float = 0.25,
        ks_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Detect drift for a single column
        
        Args:
            col_name: Column name
            series_train: Training data
            series_test: Test data
            psi_threshold: PSI threshold for significance
            ks_threshold: KS statistic threshold
            
        Returns:
            Drift report for column
        """
        result = {
            "column": col_name,
            "dtype": str(series_train.dtype),
            "train_count": len(series_train),
            "test_count": len(series_test),
            "train_null_pct": float(series_train.isna().sum() / len(series_train)),
            "test_null_pct": float(series_test.isna().sum() / len(series_test)),
            "drift_detected": False,
            "drift_score": 0.0,
            "metrics": {}
        }
        
        # Remove nulls for drift calculation
        train_clean = series_train.dropna()
        test_clean = series_test.dropna()
        
        if len(train_clean) < 2 or len(test_clean) < 2:
            return result
        
        # Numeric column
        if pd.api.types.is_numeric_dtype(series_train.dtype):
            # KS test
            ks_stat, ks_pval = cls.calculate_ks_statistic(train_clean, test_clean)
            result["metrics"]["ks_statistic"] = ks_stat
            result["metrics"]["ks_pvalue"] = ks_pval
            
            # PSI
            psi = cls.calculate_psi(train_clean, test_clean)
            result["metrics"]["psi"] = psi
            
            # Distribution comparison
            result["metrics"]["train_mean"] = float(train_clean.mean())
            result["metrics"]["test_mean"] = float(test_clean.mean())
            result["metrics"]["train_std"] = float(train_clean.std())
            result["metrics"]["test_std"] = float(test_clean.std())
            result["metrics"]["train_median"] = float(train_clean.median())
            result["metrics"]["test_median"] = float(test_clean.median())
            
            # Drift detection
            if ks_stat > ks_threshold or psi > psi_threshold:
                result["drift_detected"] = True
                result["drift_score"] = max(ks_stat / ks_threshold, psi / psi_threshold)
        
        # Categorical column
        else:
            # Chi-square
            chi2, chi_pval = cls.calculate_chi_square(train_clean, test_clean)
            result["metrics"]["chi_square"] = chi2
            result["metrics"]["chi_pvalue"] = chi_pval
            
            # Distribution divergence
            train_dist = train_clean.value_counts(normalize=True)
            test_dist = test_clean.value_counts(normalize=True)
            
            # Top categories
            top_cats = set(train_dist.head(5).index) | set(test_dist.head(5).index)
            for cat in top_cats:
                result["metrics"][f"train_{cat}_pct"] = float(train_dist.get(cat, 0) * 100)
                result["metrics"][f"test_{cat}_pct"] = float(test_dist.get(cat, 0) * 100)
            
            # Drift detection based on distribution change
            if chi_pval < 0.05 or chi2 > 10:
                result["drift_detected"] = True
                result["drift_score"] = min(chi2 / 10, 1.0)
        
        return result
    
    @classmethod
    def detect_dataset_drift(
        cls,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        exclude_cols: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Detect drift across entire dataset
        
        Args:
            df_train: Training dataset
            df_test: Test dataset
            exclude_cols: Columns to exclude from drift detection
            
        Returns:
            Drift report for entire dataset
        """
        if exclude_cols is None:
            exclude_cols = []
        
        logger.info(f"Detecting drift between train ({len(df_train)}) and test ({len(df_test)}) datasets")
        
        # Find common columns
        common_cols = set(df_train.columns) & set(df_test.columns)
        common_cols = [c for c in common_cols if c not in exclude_cols]
        
        column_drifts = []
        drift_flags = []
        
        for col in sorted(common_cols):
            drift_result = cls.detect_column_drift(
                col,
                df_train[col],
                df_test[col]
            )
            column_drifts.append(drift_result)
            
            if drift_result["drift_detected"]:
                drift_flags.append(col)
        
        # Sort by drift score
        column_drifts.sort(key=lambda x: x["drift_score"], reverse=True)
        
        # Overall assessment
        report = {
            "train_records": len(df_train),
            "test_records": len(df_test),
            "columns_analyzed": len(common_cols),
            "columns_with_drift": len(drift_flags),
            "overall_drift_risk": "LOW" if len(drift_flags) == 0 else (
                "MEDIUM" if len(drift_flags) < len(common_cols) * 0.1 else "HIGH"
            ),
            "top_drifting_columns": [c["column"] for c in column_drifts[:5]],
            "column_drifts": column_drifts
        }
        
        logger.info(f"Drift detection complete: {len(drift_flags)} columns with drift")
        
        return report
    
    @staticmethod
    def save_drift_report(
        report: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Save drift detection report"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_dir / "drift_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Drift report saved to {json_path}")
        
        # Save CSV with column-level drift
        if "column_drifts" in report:
            df_drifts = pd.DataFrame(report["column_drifts"])
            csv_path = output_dir / "column_drifts.csv"
            df_drifts.to_csv(csv_path, index=False)
            logger.info(f"Column drift details saved to {csv_path}")
