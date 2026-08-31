"""
Data profiling and quality scoring for loan data
Implements Task 1: Data Intelligence and Profiling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Statistical profile of a single column"""
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    q25: Optional[float]
    q75: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    mode: Optional[str]
    value_counts: Dict[str, int]  # Top values
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecordQualityScore:
    """Quality score for a single loan record"""
    loan_id: Any
    month_index: int
    quality_score: float  # 0-1
    issues: List[str]
    severity: str  # LOW, MEDIUM, HIGH
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataProfiler:
    """Profile and analyze loan data distributions and quality"""
    
    @staticmethod
    def profile_column(series: pd.Series, top_n: int = 10) -> ColumnProfile:
        """
        Create statistical profile of a column
        
        Args:
            series: Pandas Series
            top_n: Number of top values to capture
            
        Returns:
            ColumnProfile with statistics
        """
        name = series.name
        missing_count = series.isna().sum()
        missing_pct = missing_count / len(series)
        unique_count = series.nunique()
        unique_pct = unique_count / len(series)
        
        # Try numeric statistics
        mean = None
        median = None
        std = None
        min_val = None
        max_val = None
        q25 = None
        q75 = None
        skewness = None
        kurtosis = None
        
        try:
            numeric_data = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric_data) > 0:
                mean = float(numeric_data.mean())
                median = float(numeric_data.median())
                std = float(numeric_data.std())
                min_val = float(numeric_data.min())
                max_val = float(numeric_data.max())
                q25 = float(numeric_data.quantile(0.25))
                q75 = float(numeric_data.quantile(0.75))
                skewness = float(numeric_data.skew())
                kurtosis = float(numeric_data.kurtosis())
        except Exception as e:
            logger.debug(f"Could not compute numeric stats for {name}: {e}")
        
        # Mode
        mode_val = None
        try:
            mode_series = series.dropna().mode()
            if len(mode_series) > 0:
                mode_val = str(mode_series.iloc[0])
        except:
            pass
        
        # Value counts
        value_counts_dict = series.value_counts().head(top_n).to_dict()
        value_counts = {str(k): int(v) for k, v in value_counts_dict.items()}
        
        return ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            missing_count=int(missing_count),
            missing_pct=float(missing_pct),
            unique_count=int(unique_count),
            unique_pct=float(unique_pct),
            mean=mean,
            median=median,
            std=std,
            min_val=min_val,
            max_val=max_val,
            q25=q25,
            q75=q75,
            skewness=skewness,
            kurtosis=kurtosis,
            mode=mode_val,
            value_counts=value_counts
        )
    
    @classmethod
    def profile_dataset(cls, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """
        Profile entire dataset
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Dict mapping column names to profiles
        """
        profiles = {}
        for col in df.columns:
            try:
                profiles[col] = cls.profile_column(df[col])
            except Exception as e:
                logger.error(f"Error profiling column {col}: {e}")
        
        return profiles
    
    @staticmethod
    def calculate_correlations(
        df: pd.DataFrame,
        numeric_only: bool = True
    ) -> pd.DataFrame:
        """
        Calculate correlations between numeric columns
        
        Args:
            df: DataFrame
            numeric_only: Only numeric columns
            
        Returns:
            Correlation matrix
        """
        try:
            return df.corr(numeric_only=numeric_only)
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Dict[str, List[int]]:
        """
        Detect outliers using IQR or Z-score method
        
        Args:
            df: DataFrame
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
            
        Returns:
            Dict mapping column names to outlier indices
        """
        outliers = {}
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if len(col_data) < 2:
                continue
            
            if method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower) | (df[col] > upper)
            else:  # zscore
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    outlier_mask = z_scores > threshold
                else:
                    outlier_mask = pd.Series(False, index=df.index)
            
            outlier_indices = df[outlier_mask].index.tolist()
            if outlier_indices:
                outliers[col] = outlier_indices
        
        return outliers
    
    @staticmethod
    def calculate_record_quality_score(
        record: pd.Series,
        df: pd.DataFrame,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> RecordQualityScore:
        """
        Calculate quality score for a single record
        
        Args:
            record: Single row from DataFrame
            df: Full DataFrame for context
            validation_rules: Dict of validation rules
            
        Returns:
            RecordQualityScore
        """
        issues = []
        severity_score = 0
        
        # Check for missing required fields
        required_fields = ["loan_id", "reporting_month", "current_status"]
        for field in required_fields:
            if field in record.index and pd.isna(record[field]):
                issues.append(f"Missing required field: {field}")
                severity_score += 0.3
        
        # Check balance consistency
        if "original_balance" in record.index and "current_balance" in record.index:
            orig = record["original_balance"]
            curr = record["current_balance"]
            if pd.notna(orig) and pd.notna(curr):
                if curr > orig * 1.1:  # Current > 110% of original
                    issues.append("current_balance exceeds 110% of original_balance")
                    severity_score += 0.2
                elif curr < 0:
                    issues.append("current_balance is negative")
                    severity_score += 0.3
        
        # Check days past due consistency
        if "days_past_due" in record.index and "current_status" in record.index:
            dpd = record["days_past_due"]
            status = record["current_status"]
            if pd.notna(dpd) and pd.notna(status):
                if dpd > 0 and status == "CURRENT":
                    issues.append("DPD > 0 but status is CURRENT")
                    severity_score += 0.15
                elif dpd == 0 and status == "DELINQUENT":
                    issues.append("DPD = 0 but status is DELINQUENT")
                    severity_score += 0.1
        
        # Check date consistency
        if "reporting_month" in record.index and "origination_month" in record.index:
            rep = record["reporting_month"]
            orig = record["origination_month"]
            if pd.notna(rep) and pd.notna(orig):
                try:
                    rep_date = pd.to_datetime(rep)
                    orig_date = pd.to_datetime(orig)
                    if rep_date < orig_date:
                        issues.append("reporting_month before origination_month")
                        severity_score += 0.25
                except:
                    pass
        
        # Normalize severity
        severity_score = min(0.99, severity_score)
        quality_score = 1.0 - severity_score
        
        # Determine severity level
        if severity_score >= 0.3:
            severity = "HIGH"
        elif severity_score >= 0.15:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        loan_id = record.get("loan_id", "UNKNOWN")
        month_idx = record.get("month_index", 0)
        
        return RecordQualityScore(
            loan_id=loan_id,
            month_index=int(month_idx) if pd.notna(month_idx) else 0,
            quality_score=float(quality_score),
            issues=issues,
            severity=severity
        )
    
    @classmethod
    def calculate_batch_quality_scores(
        cls,
        df: pd.DataFrame,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculate quality scores for entire dataset
        
        Args:
            df: Input DataFrame
            validation_rules: Validation rules
            
        Returns:
            Tuple of (DataFrame with quality scores, summary statistics)
        """
        logger.info(f"Calculating record-level quality scores for {len(df)} records")
        
        # Calculate score for each record
        scores = []
        for idx, row in df.iterrows():
            score = cls.calculate_record_quality_score(row, df, validation_rules)
            scores.append(score.to_dict())
        
        scores_df = pd.DataFrame(scores)
        
        # Batch-level statistics
        summary = {
            "total_records": len(df),
            "average_quality_score": float(scores_df["quality_score"].mean()),
            "median_quality_score": float(scores_df["quality_score"].median()),
            "min_quality_score": float(scores_df["quality_score"].min()),
            "max_quality_score": float(scores_df["quality_score"].max()),
            "std_quality_score": float(scores_df["quality_score"].std()),
            "records_high_severity": int((scores_df["severity"] == "HIGH").sum()),
            "records_medium_severity": int((scores_df["severity"] == "MEDIUM").sum()),
            "records_low_severity": int((scores_df["severity"] == "LOW").sum()),
            "pct_high_severity": float((scores_df["severity"] == "HIGH").sum() / len(scores_df)),
            "pct_medium_severity": float((scores_df["severity"] == "MEDIUM").sum() / len(scores_df)),
            "pct_low_severity": float((scores_df["severity"] == "LOW").sum() / len(scores_df)),
        }
        
        logger.info(
            f"Quality scores calculated. Mean: {summary['average_quality_score']:.3f}, "
            f"High severity: {summary['records_high_severity']} records"
        )
        
        return scores_df, summary


class DataQualityReporter:
    """Generate data quality reports"""
    
    @staticmethod
    def save_profile_report(
        profiles: Dict[str, ColumnProfile],
        output_dir: Path,
        filename: str = "data_profile.json"
    ) -> None:
        """Save column profiles to JSON"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        profiles_dict = {
            name: profile.to_dict()
            for name, profile in profiles.items()
        }
        
        with open(output_dir / filename, "w") as f:
            json.dump(profiles_dict, f, indent=2, default=str)
        
        logger.info(f"Profile report saved to {output_dir / filename}")
    
    @staticmethod
    def save_quality_report(
        scores_df: pd.DataFrame,
        summary: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Save quality scores and summary"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / "data_quality_report.csv"
        scores_df.to_csv(csv_path, index=False)
        logger.info(f"Quality scores saved to {csv_path}")
        
        # Save summary JSON
        summary_path = output_dir / "data_quality_summary.json"
        summary["timestamp"] = datetime.now().isoformat()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Quality summary saved to {summary_path}")
    
    @staticmethod
    def generate_html_profile_report(
        profiles: Dict[str, ColumnProfile],
        output_path: Path
    ) -> None:
        """Generate HTML data intelligence report"""
        html_content = """
        <html>
        <head>
            <title>Data Intelligence Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .summary { background-color: #e3f2fd; padding: 10px; border-radius: 4px; }
                .high-missing { background-color: #ffcccc; }
                .outliers { background-color: #ffffcc; }
            </style>
        </head>
        <body>
            <h1>Data Intelligence Report</h1>
            
            <h2>Column Profiles</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Data Type</th>
                    <th>Non-Null %</th>
                    <th>Unique</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Skewness</th>
                </tr>
        """
        
        for col_name in sorted(profiles.keys()):
            profile = profiles[col_name]
            mean_str = f"{profile.mean:.2f}" if profile.mean is not None else "N/A"
            median_str = f"{profile.median:.2f}" if profile.median is not None else "N/A"
            std_str = f"{profile.std:.2f}" if profile.std is not None else "N/A"
            min_str = f"{profile.min_val:.2f}" if profile.min_val is not None else "N/A"
            max_str = f"{profile.max_val:.2f}" if profile.max_val is not None else "N/A"
            skew_str = f"{profile.skewness:.2f}" if profile.skewness is not None else "N/A"
            
            html_content += f"""
                <tr>
                    <td><strong>{profile.name}</strong></td>
                    <td>{profile.dtype}</td>
                    <td>{(1 - profile.missing_pct):.1%}</td>
                    <td>{profile.unique_count}</td>
                    <td>{mean_str}</td>
                    <td>{median_str}</td>
                    <td>{std_str}</td>
                    <td>{min_str}</td>
                    <td>{max_str}</td>
                    <td>{skew_str}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"HTML profile report saved to {output_path}")
