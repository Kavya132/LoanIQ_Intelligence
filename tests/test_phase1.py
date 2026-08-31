"""
Basic tests to verify Phase 1 setup
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings, load_yaml_config
from src.utils.logging import setup_logging, get_logger
from src.data.schema import SchemaDiscovery
from src.data.loader import DataLoader
import pandas as pd
import numpy as np


class TestConfigSetup:
    """Test configuration loading"""
    
    def test_load_yaml_config(self):
        """Test YAML config loading"""
        config = load_yaml_config()
        assert isinstance(config, dict)
        assert "data" in config
    
    def test_get_settings(self):
        """Test settings initialization"""
        settings = get_settings()
        assert settings is not None
        assert settings.project_root.exists()
        assert settings.data_raw_dir.exists()
        assert settings.models_dir.exists()
        assert settings.outputs_dir.exists()


class TestLogging:
    """Test logging setup"""
    
    def test_setup_logging(self, tmp_path):
        """Test logging initialization"""
        setup_logging(log_level="INFO", log_dir=tmp_path)
        logger = get_logger(__name__)
        assert logger is not None
        # Log something
        logger.info("Test log message")
        # Check log file was created
        assert (tmp_path / "loan_intelligence.log").exists()


class TestSchemaDiscovery:
    """Test schema discovery"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe"""
        return pd.DataFrame({
            "loan_id": [1, 2, 3],
            "reporting_month": ["2023-01", "2023-02", "2023-03"],
            "current_balance": [100000.0, 95000.0, 90000.0],
            "days_past_due": [0, 30, 60],
            "next_state": ["CURRENT", "DELINQUENT", "DEFAULT"],
        })
    
    def test_infer_dtype(self, sample_df):
        """Test data type inference"""
        assert SchemaDiscovery.infer_dtype(sample_df["loan_id"]) == "int"
        assert SchemaDiscovery.infer_dtype(sample_df["current_balance"]) == "float"
        assert SchemaDiscovery.infer_dtype(sample_df["reporting_month"]) == "datetime"
        assert SchemaDiscovery.infer_dtype(sample_df["next_state"]) == "string"
    
    def test_discover_schema(self, sample_df):
        """Test schema discovery"""
        report = SchemaDiscovery.discover_schema(
            sample_df,
            "test_file.csv",
            expected_columns=["loan_id", "reporting_month", "current_balance"]
        )
        assert report.filename == "test_file.csv"
        assert report.row_count == 3
        assert report.column_count == 5
        assert "loan_id" in report.found_columns
        assert len(report.missing_columns) == 0
    
    def test_match_columns(self):
        """Test column name matching"""
        actual = ["loan_identifier", "Report_Month", "DPD", "loan_balance"]
        expected = ["loan_id", "reporting_month", "days_past_due", "current_balance"]
        
        matched = SchemaDiscovery._match_columns(actual, expected)
        
        # Should find loan_id via alias
        assert "loan_identifier" in matched
        # Should find reporting_month via case-insensitive
        assert "Report_Month" in matched
        # Should find days_past_due via alias
        assert "DPD" in matched


class TestDataLoader:
    """Test data loading"""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create sample CSV file"""
        df = pd.DataFrame({
            "loan_id": [1, 2, 3],
            "reporting_month": ["2023-01", "2023-02", "2023-03"],
            "balance": [100000.0, 95000.0, 90000.0],
        })
        filepath = tmp_path / "test.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    def test_detect_encoding(self, sample_csv):
        """Test encoding detection"""
        encoding = DataLoader.detect_encoding(sample_csv)
        assert encoding.lower() in ["utf-8", "ascii", "utf_8"]
    
    def test_load_csv(self, sample_csv):
        """Test CSV loading"""
        df = DataLoader.load_csv(sample_csv)
        assert len(df) == 3
        assert "loan_id" in df.columns
        assert df["loan_id"].dtype in [int, np.int64]
    
    def test_identify_date_columns(self, sample_csv):
        """Test date column identification"""
        df = pd.read_csv(sample_csv)
        date_cols = DataLoader._identify_date_columns(df)
        assert "reporting_month" in date_cols
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df = pd.DataFrame({
            "a": [1, 2, np.nan],
            "b": [np.nan, np.nan, np.nan],
            "c": [1, 1, 1]
        })
        
        # Keep strategy
        result = DataLoader.handle_missing_values(df, strategy="keep")
        assert len(result) == 3  # Keep all rows
        
        # Drop columns with >50% missing
        result = DataLoader.handle_missing_values(df, strategy="drop_cols", threshold=0.5)
        assert "b" not in result.columns  # 100% missing
    
    def test_handle_duplicates(self):
        """Test duplicate handling"""
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "b": [1, 1, 2]
        })
        
        result = DataLoader.handle_duplicates(df)
        assert len(result) == 2  # One duplicate removed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
