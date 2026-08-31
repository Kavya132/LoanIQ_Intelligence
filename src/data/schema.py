"""
Schema discovery and validation for loan data
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum


class DataType(str, Enum):
    """Data type enumeration"""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    DATETIME = "datetime"
    BOOL = "bool"
    CATEGORICAL = "categorical"


@dataclass
class ColumnSchema:
    """Schema information for a single column"""
    name: str
    dtype: str
    inferred_dtype: str
    nullable: bool
    missing_pct: float
    unique_count: int
    sample_values: List[Any]
    expected: bool
    status: str  # FOUND, MISSING, UNEXPECTED
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataSchemaReport:
    """Complete schema report for a dataset"""
    filename: str
    row_count: int
    column_count: int
    columns: List[ColumnSchema]
    expected_columns: List[str]
    found_columns: List[str]
    missing_columns: List[str]
    unexpected_columns: List[str]
    data_completeness: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "expected_columns": self.expected_columns,
            "found_columns": self.found_columns,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "data_completeness": self.data_completeness,
            "columns": [c.to_dict() for c in self.columns],
            "timestamp": self.timestamp,
        }


class SchemaDiscovery:
    """Discover and validate data schema"""
    
    # Expected training fields from specification
    EXPECTED_TRAINING_FIELDS = [
        "loan_id", "month_index", "reporting_month", "origination_month",
        "loan_age_months", "remaining_term_months", "original_balance",
        "current_balance", "interest_rate", "credit_score_band",
        "ltv_band", "dti_band", "state", "loan_purpose", "occupancy_type",
        "property_type", "servicer_name", "current_status", "days_past_due",
        "modification_flag", "prepayment_flag", "default_flag",
        "loss_severity_band", "last_updated_at", "source_system",
        "document_status"
    ]
    
    # Expected target fields
    EXPECTED_TARGETS = [
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
        "next_state",
        "exception_required",
        "exception_type"
    ]
    
    # Field aliases for flexible data discovery
    FIELD_ALIASES = {
        "loan_id": ["loan_id", "loan_identifier", "loanid"],
        "reporting_month": ["reporting_month", "month", "observation_month", "report_month"],
        "origination_month": ["origination_month", "orig_month"],
        "current_balance": ["current_balance", "balance", "current_principal"],
        "days_past_due": ["days_past_due", "dpd", "delinquency_days"],
        "current_status": ["current_status", "loan_status", "status"],
    }
    
    @staticmethod
    def infer_dtype(series: pd.Series) -> str:
        """Infer the data type of a pandas Series"""
        if series.dtype == object:
            # Try to determine if it's date
            try:
                pd.to_datetime(series.dropna())
                return "datetime"
            except:
                return "string"
        elif series.dtype == bool or series.name and any(x in str(series.name).lower() for x in ["flag", "indicator"]):
            return "bool"
        elif pd.api.types.is_integer_dtype(series.dtype):
            return "int"
        elif pd.api.types.is_float_dtype(series.dtype):
            return "float"
        elif pd.api.types.is_categorical_dtype(series.dtype):
            return "categorical"
        else:
            return str(series.dtype)
    
    @staticmethod
    def get_sample_values(series: pd.Series, n: int = 5) -> List[Any]:
        """Get sample non-null values from series"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return []
        # Convert to list, handling various types
        samples = non_null.head(n).tolist()
        return [str(x) if not isinstance(x, (int, float, bool)) else x for x in samples]
    
    @classmethod
    def discover_schema(
        cls,
        df: pd.DataFrame,
        filename: str,
        expected_columns: Optional[List[str]] = None
    ) -> DataSchemaReport:
        """
        Discover schema from a dataframe
        
        Args:
            df: Pandas DataFrame
            filename: Source filename
            expected_columns: List of expected columns (defaults to training fields)
            
        Returns:
            DataSchemaReport with schema information
        """
        from datetime import datetime
        
        if expected_columns is None:
            expected_columns = cls.EXPECTED_TRAINING_FIELDS
        
        # Find actual columns (handle case-insensitive and alias matching)
        found_columns = cls._match_columns(df.columns, expected_columns)
        missing_columns = [c for c in expected_columns if c not in found_columns]
        unexpected_columns = [c for c in df.columns if c not in found_columns]
        
        # Build column schemas
        column_schemas = []
        for col in df.columns:
            series = df[col]
            missing_pct = series.isna().sum() / len(series)
            
            schema = ColumnSchema(
                name=col,
                dtype=str(series.dtype),
                inferred_dtype=cls.infer_dtype(series),
                nullable=missing_pct > 0,
                missing_pct=missing_pct,
                unique_count=series.nunique(),
                sample_values=cls.get_sample_values(series),
                expected=col in found_columns,
                status="FOUND" if col in found_columns else (
                    "MISSING" if col in missing_columns else "UNEXPECTED"
                )
            )
            column_schemas.append(schema)
        
        # Calculate overall completeness
        data_completeness = (1 - (len(missing_columns) / len(expected_columns))) if expected_columns else 1.0
        
        report = DataSchemaReport(
            filename=filename,
            row_count=len(df),
            column_count=len(df.columns),
            columns=column_schemas,
            expected_columns=expected_columns,
            found_columns=found_columns,
            missing_columns=missing_columns,
            unexpected_columns=unexpected_columns,
            data_completeness=data_completeness,
            timestamp=datetime.now().isoformat()
        )
        
        return report
    
    @classmethod
    def _match_columns(cls, actual_columns, expected_columns: List[str]) -> List[str]:
        """
        Match actual columns to expected columns using exact match and aliases
        
        Args:
            actual_columns: Columns from dataframe
            expected_columns: Expected column names
            
        Returns:
            List of matched column names
        """
        matched = []
        actual_lower = {col.lower(): col for col in actual_columns}
        
        for expected in expected_columns:
            # Exact match
            if expected in actual_columns:
                matched.append(expected)
            # Case-insensitive match
            elif expected.lower() in actual_lower:
                matched.append(actual_lower[expected.lower()])
            # Alias match
            elif expected in cls.FIELD_ALIASES:
                for alias in cls.FIELD_ALIASES[expected]:
                    if alias in actual_columns:
                        matched.append(alias)
                        break
                    elif alias.lower() in actual_lower:
                        matched.append(actual_lower[alias.lower()])
                        break
        
        return matched
    
    @classmethod
    def save_report(cls, report: DataSchemaReport, output_dir: Path) -> None:
        """Save schema report to JSON and HTML"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / "schema_report.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        # Save HTML
        html_path = output_dir / "schema_report.html"
        html_content = cls._generate_html_report(report)
        with open(html_path, "w") as f:
            f.write(html_content)
    
    @staticmethod
    def _generate_html_report(report: DataSchemaReport) -> str:
        """Generate HTML report"""
        html = f"""
        <html>
        <head>
            <title>Schema Report: {report.filename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e3f2fd; padding: 10px; border-radius: 4px; }}
                .status-found {{ color: green; }}
                .status-missing {{ color: red; }}
                .status-unexpected {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>Schema Discovery Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>File:</strong> {report.filename}</p>
                <p><strong>Rows:</strong> {report.row_count:,}</p>
                <p><strong>Columns:</strong> {report.column_count}</p>
                <p><strong>Data Completeness:</strong> {report.data_completeness:.1%}</p>
                <p><strong>Missing Expected Columns:</strong> {len(report.missing_columns)}</p>
                <p><strong>Unexpected Columns:</strong> {len(report.unexpected_columns)}</p>
            </div>
            
            <h2>Column Schema</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Pandas Type</th>
                    <th>Inferred Type</th>
                    <th>Unique</th>
                    <th>Missing %</th>
                    <th>Status</th>
                    <th>Sample Values</th>
                </tr>
        """
        
        for col in report.columns:
            status_class = f"status-{col.status.lower()}"
            samples = ", ".join(str(s)[:20] for s in col.sample_values[:3])
            html += f"""
                <tr>
                    <td>{col.name}</td>
                    <td>{col.dtype}</td>
                    <td>{col.inferred_dtype}</td>
                    <td>{col.unique_count}</td>
                    <td>{col.missing_pct:.1%}</td>
                    <td class="{status_class}">{col.status}</td>
                    <td>{samples}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
