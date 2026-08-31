"""
Servicer data reconciliation for conflict detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of reconciling two data sources"""
    loan_id: Any
    field: str
    primary_value: Any
    secondary_value: Any
    conflict: bool
    confidence: float  # How confident we are about the conflict
    recommendation: str


class ServicerReconciliation:
    """
    Reconcile data from primary source with servicer updates
    Detect conflicts and stale records
    """
    
    @staticmethod
    def detect_conflicts(
        primary_df: pd.DataFrame,
        servicer_df: pd.DataFrame,
        key_cols: List[str] = None
    ) -> List[ReconciliationResult]:
        """
        Detect conflicts between primary and servicer data
        
        Args:
            primary_df: Primary source data (loan_monthly_performance)
            servicer_df: Servicer updates data
            key_cols: Columns to check for conflicts
            
        Returns:
            List of conflicts
        """
        if key_cols is None:
            key_cols = ["loan_id", "reporting_month", "current_balance", "days_past_due", "current_status"]
        
        conflicts = []
        
        # Merge on loan_id and reporting_month if available
        merge_on = []
        if "loan_id" in primary_df.columns and "loan_id" in servicer_df.columns:
            merge_on.append("loan_id")
        if "reporting_month" in primary_df.columns and "reporting_month" in servicer_df.columns:
            merge_on.append("reporting_month")
        
        if not merge_on:
            logger.warning("Cannot merge: no common key columns")
            return conflicts
        
        # Merge dataframes
        try:
            merged = primary_df.merge(
                servicer_df,
                on=merge_on,
                how="inner",
                suffixes=("_primary", "_servicer")
            )
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return conflicts
        
        # Check for conflicts
        for col in key_cols:
            col_primary = f"{col}_primary" if f"{col}_primary" in merged.columns else None
            col_servicer = f"{col}_servicer" if f"{col}_servicer" in merged.columns else None
            
            if col_primary is None and col in merged.columns:
                col_primary = col
            if col_servicer is None and col in merged.columns:
                col_servicer = col
            
            if col_primary and col_servicer and col_primary != col_servicer:
                for idx, row in merged.iterrows():
                    prim_val = row[col_primary]
                    serv_val = row[col_servicer]
                    
                    # Check if values differ (allowing small numeric differences)
                    differs = False
                    if pd.isna(prim_val) != pd.isna(serv_val):
                        differs = True
                    elif pd.notna(prim_val) and pd.notna(serv_val):
                        if isinstance(prim_val, (int, float)) and isinstance(serv_val, (int, float)):
                            # Allow 1% tolerance for numeric values
                            if abs(prim_val - serv_val) > abs(prim_val) * 0.01:
                                differs = True
                        else:
                            if prim_val != serv_val:
                                differs = True
                    
                    if differs:
                        loan_id = row.get("loan_id", "UNKNOWN")
                        conflicts.append(ReconciliationResult(
                            loan_id=loan_id,
                            field=col,
                            primary_value=prim_val,
                            secondary_value=serv_val,
                            conflict=True,
                            confidence=0.8,
                            recommendation=f"Use primary value {prim_val} (servicer reported {serv_val})"
                        ))
        
        logger.info(f"Found {len(conflicts)} potential conflicts")
        return conflicts
    
    @staticmethod
    def detect_stale_records(
        df: pd.DataFrame,
        date_col: str = "last_updated_at",
        days_threshold: int = 90
    ) -> pd.DataFrame:
        """
        Identify stale records not updated recently
        
        Args:
            df: Input DataFrame
            date_col: Column with update timestamp
            days_threshold: Days since update to consider stale
            
        Returns:
            DataFrame with stale_record_flag
        """
        if date_col not in df.columns:
            logger.warning(f"Column {date_col} not found, cannot detect stale records")
            df["stale_record_flag"] = 0
            return df
        
        try:
            df["last_updated"] = pd.to_datetime(df[date_col], errors="coerce")
            now = pd.Timestamp.now()
            days_since = (now - df["last_updated"]).dt.days
            
            df["stale_record_flag"] = (days_since > days_threshold).astype(int)
            stale_count = df["stale_record_flag"].sum()
            logger.info(f"Detected {stale_count} stale records (not updated in {days_threshold} days)")
        
        except Exception as e:
            logger.error(f"Error detecting stale records: {e}")
            df["stale_record_flag"] = 0
        
        return df
    
    @staticmethod
    def detect_source_conflicts(
        df: pd.DataFrame,
        source_col: str = "source_system"
    ) -> pd.DataFrame:
        """
        Detect multiple conflicting sources for same loan
        
        Args:
            df: Input DataFrame
            source_col: Column indicating data source
            
        Returns:
            DataFrame with source_conflict_flag
        """
        if source_col not in df.columns:
            logger.warning(f"Column {source_col} not found, cannot detect source conflicts")
            df["source_conflict_flag"] = 0
            return df
        
        df["source_conflict_flag"] = 0
        
        try:
            # Group by loan_id and check for multiple sources
            if "loan_id" in df.columns:
                source_counts = df.groupby("loan_id")[source_col].nunique()
                multi_source_loans = source_counts[source_counts > 1].index
                
                df.loc[df["loan_id"].isin(multi_source_loans), "source_conflict_flag"] = 1
                
                conflict_count = df["source_conflict_flag"].sum()
                logger.info(f"Detected {conflict_count} records with source conflicts")
        
        except Exception as e:
            logger.error(f"Error detecting source conflicts: {e}")
        
        return df
    
    @staticmethod
    def add_reconciliation_flags(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add reconciliation-related flags to DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new reconciliation flags
        """
        # Initialize flags
        if "source_conflict_flag" not in df.columns:
            df["source_conflict_flag"] = 0
        if "stale_record_flag" not in df.columns:
            df["stale_record_flag"] = 0
        if "reconciliation_status" not in df.columns:
            df["reconciliation_status"] = "OK"
        
        # Update flags
        df = ServicerReconciliation.detect_source_conflicts(df)
        df = ServicerReconciliation.detect_stale_records(df)
        
        # Set reconciliation status
        df.loc[df["source_conflict_flag"] == 1, "reconciliation_status"] = "CONFLICT"
        df.loc[df["stale_record_flag"] == 1, "reconciliation_status"] = "STALE"
        df.loc[(df["source_conflict_flag"] == 1) & (df["stale_record_flag"] == 1), "reconciliation_status"] = "CONFLICT_STALE"
        
        return df
