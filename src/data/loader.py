"""
Data loading utilities with encoding detection and flexible schema matching
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import chardet
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Robust data loader for CSV files with flexible schema discovery
    """
    
    # Common date column patterns
    DATE_PATTERNS = [
        "month", "date", "reporting", "origination", "observation",
        "updated", "timestamp", "last_", "_at"
    ]
    
    @staticmethod
    def detect_encoding(filepath: Path, sample_size: int = 10000) -> str:
        """
        Detect file encoding using chardet
        
        Args:
            filepath: Path to file
            sample_size: Number of bytes to sample
            
        Returns:
            Detected encoding name
        """
        with open(filepath, "rb") as f:
            raw_data = f.read(sample_size)
        
        detection = chardet.detect(raw_data)
        encoding = detection.get("encoding", "utf-8")
        
        if encoding is None:
            encoding = "utf-8"
        
        logger.info(f"Detected encoding for {filepath.name}: {encoding}")
        return encoding
    
    @classmethod
    def load_csv(
        cls,
        filepath: Path,
        encoding: Optional[str] = None,
        parse_dates: bool = True,
        memory_efficient: bool = False,
        nrows: Optional[int] = None,
        sample_for_schema: int = 1000
    ) -> pd.DataFrame:
        """
        Load CSV file with robust error handling
        
        Args:
            filepath: Path to CSV file
            encoding: File encoding (auto-detected if None)
            parse_dates: Whether to parse date columns
            memory_efficient: Use chunking for large files
            nrows: Limit number of rows to load
            sample_for_schema: Number of rows to sample for schema detection
            
        Returns:
            Loaded DataFrame
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Detect encoding if not provided
        if encoding is None:
            encoding = cls.detect_encoding(filepath)
        
        logger.info(f"Loading CSV: {filepath.name} ({encoding})")
        
        try:
            # First pass: detect dtypes and date columns
            sample_df = pd.read_csv(
                filepath,
                encoding=encoding,
                nrows=min(sample_for_schema, 1000),
                low_memory=False
            )
            
            # Identify date columns
            date_cols = cls._identify_date_columns(sample_df)
            
            # Load full dataset
            df = pd.read_csv(
                filepath,
                encoding=encoding,
                parse_dates=date_cols if parse_dates else False,
                nrows=nrows,
                low_memory=False
            )
            
            logger.info(
                f"Loaded {len(df)} rows × {len(df.columns)} columns from {filepath.name}"
            )
            
            return df
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error with {encoding}: {e}")
            # Try UTF-8 as fallback
            if encoding.lower() != "utf-8":
                logger.info("Retrying with UTF-8 encoding")
                return cls.load_csv(
                    filepath,
                    encoding="utf-8",
                    parse_dates=parse_dates,
                    memory_efficient=memory_efficient,
                    nrows=nrows
                )
            raise
        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {e}")
            raise
    
    @classmethod
    def _identify_date_columns(cls, df: pd.DataFrame) -> List[str]:
        """
        Identify likely date columns based on name patterns and values
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of likely date column names
        """
        date_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check name patterns
            if any(pattern in col_lower for pattern in cls.DATE_PATTERNS):
                date_cols.append(col)
                continue
            
            # Check if values look like dates
            if df[col].dtype == object:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    try:
                        pd.to_datetime(sample)
                        date_cols.append(col)
                    except:
                        pass
        
        return date_cols
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = "keep",
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            df: Input DataFrame
            strategy: 'keep', 'drop_rows', 'drop_cols'
            threshold: Drop columns if missing % > threshold
            
        Returns:
            Cleaned DataFrame
        """
        missing_report = df.isna().sum()
        missing_pct = missing_report / len(df)
        
        logger.info(f"Missing values summary:\n{missing_pct[missing_pct > 0]}")
        
        if strategy == "drop_cols":
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            if cols_to_drop:
                logger.info(f"Dropping columns with >{threshold:.0%} missing: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
        
        elif strategy == "drop_rows":
            initial_len = len(df)
            df = df.dropna()
            logger.info(f"Dropped {initial_len - len(df)} rows with any missing values")
        
        return df
    
    @staticmethod
    def handle_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> pd.DataFrame:
        """
        Handle duplicate rows
        
        Args:
            df: Input DataFrame
            subset: Columns to check for duplicates (all if None)
            keep: 'first', 'last', or False (remove all)
            
        Returns:
            DataFrame with duplicates handled
        """
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        duplicates_removed = initial_len - len(df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df
    
    @staticmethod
    def convert_numeric_safe(
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Safely convert columns to numeric types
        
        Args:
            df: Input DataFrame
            numeric_cols: Columns to convert (all if None)
            
        Returns:
            DataFrame with numeric columns converted
        """
        if numeric_cols is None:
            numeric_cols = df.columns.tolist()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Log conversion issues
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Column {col}: {nan_count} values could not convert to numeric")
            except Exception as e:
                logger.debug(f"Could not convert {col} to numeric: {e}")
        
        return df
    
    @staticmethod
    def infer_bool_columns(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Identify columns that should be boolean
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict mapping column names to boolean conversion status
        """
        bool_map = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check name patterns
            if any(x in col_lower for x in ["flag", "indicator", "is_", "_flag"]):
                unique_vals = df[col].dropna().unique()
                # Check if only contains 0/1, True/False, yes/no, etc.
                if len(unique_vals) <= 2:
                    bool_map[col] = True
        
        return bool_map
    
    @classmethod
    def clean_dataset(
        cls,
        df: pd.DataFrame,
        handle_missing: bool = True,
        handle_duplicates: bool = True,
        convert_types: bool = True
    ) -> pd.DataFrame:
        """
        Apply full dataset cleaning pipeline
        
        Args:
            df: Input DataFrame
            handle_missing: Handle missing values
            handle_duplicates: Remove duplicates
            convert_types: Convert data types
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning: {len(df)} rows × {len(df.columns)} columns")
        
        # Handle duplicates first
        if handle_duplicates:
            df = cls.handle_duplicates(df)
        
        # Handle missing values
        if handle_missing:
            df = cls.handle_missing_values(df, strategy="keep")
        
        # Convert types
        if convert_types:
            # Infer bool columns
            bool_cols = cls.infer_bool_columns(df)
            if bool_cols:
                logger.info(f"Converting to bool: {list(bool_cols.keys())}")
                for col in bool_cols:
                    df[col] = df[col].astype("boolean")
            
            # Convert numeric columns
            numeric_cols = df.select_dtypes(include=["object"]).columns
            if len(numeric_cols) > 0:
                df = cls.convert_numeric_safe(df, numeric_cols.tolist())
        
        logger.info(f"Cleaning complete: {len(df)} rows × {len(df.columns)} columns")
        
        return df
    
    @staticmethod
    def load_all_from_directory(
        data_dir: Path,
        pattern: str = "*.csv"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from a directory
        
        Args:
            data_dir: Directory containing CSV files
            pattern: File pattern to match
            
        Returns:
            Dict mapping filenames to DataFrames
        """
        data_dir = Path(data_dir)
        datasets = {}
        
        for filepath in sorted(data_dir.glob(pattern)):
            key = filepath.stem  # filename without extension
            try:
                datasets[key] = DataLoader.load_csv(filepath)
                logger.info(f"Loaded {key}: {len(datasets[key])} rows")
            except Exception as e:
                logger.error(f"Failed to load {filepath.name}: {e}")
        
        return datasets
