"""
Validation rules engine for loan data
Detects inconsistencies and invalid states
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationFailure:
    """A single validation rule violation"""
    loan_id: Any
    month_index: int
    field: str
    rule: str
    observed_value: Any
    expected_value: Optional[Any]
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ValidationRulesEngine:
    """
    Apply deterministic validation rules to loan data
    """
    
    # Default validation rules when rules.json is not available
    DEFAULT_RULES = [
        {
            "name": "non_negative_balance",
            "description": "Current balance must be non-negative",
            "field": "current_balance",
            "rule_type": "min_value",
            "min_value": 0,
            "severity": "HIGH"
        },
        {
            "name": "non_negative_original_balance",
            "description": "Original balance must be non-negative",
            "field": "original_balance",
            "rule_type": "min_value",
            "min_value": 0,
            "severity": "HIGH"
        },
        {
            "name": "balance_monotonicity",
            "description": "Current balance should not exceed original balance",
            "field": "current_balance",
            "rule_type": "comparison",
            "compare_to": "original_balance",
            "operator": "<=",
            "severity": "HIGH"
        },
        {
            "name": "non_negative_dpd",
            "description": "Days past due must be non-negative",
            "field": "days_past_due",
            "rule_type": "min_value",
            "min_value": 0,
            "severity": "MEDIUM"
        },
        {
            "name": "non_negative_age",
            "description": "Loan age must be non-negative",
            "field": "loan_age_months",
            "rule_type": "min_value",
            "min_value": 0,
            "severity": "MEDIUM"
        },
        {
            "name": "non_negative_remaining_term",
            "description": "Remaining term must be non-negative",
            "field": "remaining_term_months",
            "rule_type": "min_value",
            "min_value": 0,
            "severity": "MEDIUM"
        },
        {
            "name": "reporting_after_origination",
            "description": "Reporting month must be on or after origination month",
            "field": "reporting_month",
            "rule_type": "date_comparison",
            "compare_to": "origination_month",
            "operator": ">=",
            "severity": "CRITICAL"
        },
        {
            "name": "dpd_consistency_with_status",
            "description": "DPD > 0 requires delinquent status",
            "field": "days_past_due",
            "rule_type": "status_consistency",
            "status_field": "current_status",
            "severity": "HIGH"
        },
        {
            "name": "interest_rate_range",
            "description": "Interest rate should be between 0 and 15 percent",
            "field": "interest_rate",
            "rule_type": "range",
            "min_value": 0,
            "max_value": 15,
            "severity": "MEDIUM"
        }
    ]
    
    @classmethod
    def load_rules(cls, rules_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Load validation rules from JSON file
        
        Args:
            rules_path: Path to validation_rules.json
            
        Returns:
            List of rule dictionaries
        """
        if rules_path and rules_path.exists():
            try:
                with open(rules_path, "r") as f:
                    rules = json.load(f)
                logger.info(f"Loaded {len(rules)} validation rules from {rules_path}")
                return rules
            except Exception as e:
                logger.warning(f"Failed to load rules from {rules_path}: {e}")
        
        logger.info("Using default validation rules")
        return cls.DEFAULT_RULES
    
    @classmethod
    def validate_dataset(
        cls,
        df: pd.DataFrame,
        rules: Optional[List[Dict[str, Any]]] = None
    ) -> List[ValidationFailure]:
        """
        Validate entire dataset against rules
        
        Args:
            df: Input DataFrame
            rules: List of rule dictionaries
            
        Returns:
            List of ValidationFailure objects
        """
        if rules is None:
            rules = cls.DEFAULT_RULES
        
        failures = []
        logger.info(f"Validating {len(df)} records against {len(rules)} rules")
        
        for idx, row in df.iterrows():
            for rule in rules:
                try:
                    rule_failures = cls.apply_rule(row, rule)
                    failures.extend(rule_failures)
                except Exception as e:
                    logger.debug(f"Error applying rule {rule.get('name')}: {e}")
        
        logger.info(f"Validation complete: {len(failures)} violations found")
        return failures
    
    @staticmethod
    def apply_rule(record: pd.Series, rule: Dict[str, Any]) -> List[ValidationFailure]:
        """
        Apply a single rule to a record
        
        Args:
            record: Single row
            rule: Rule dictionary
            
        Returns:
            List of failures (empty if rule passes)
        """
        failures = []
        rule_type = rule.get("rule_type")
        field = rule.get("field")
        
        if field not in record.index or pd.isna(record[field]):
            return failures  # Skip if field missing
        
        loan_id = record.get("loan_id", "UNKNOWN")
        month_idx = record.get("month_index", 0)
        severity = rule.get("severity", "MEDIUM")
        
        try:
            if rule_type == "min_value":
                min_val = rule.get("min_value")
                if record[field] < min_val:
                    failures.append(ValidationFailure(
                        loan_id=loan_id,
                        month_index=int(month_idx) if pd.notna(month_idx) else 0,
                        field=field,
                        rule=rule.get("name"),
                        observed_value=record[field],
                        expected_value=f">= {min_val}",
                        severity=severity,
                        message=f"{field} = {record[field]} < {min_val}"
                    ))
            
            elif rule_type == "max_value":
                max_val = rule.get("max_value")
                if record[field] > max_val:
                    failures.append(ValidationFailure(
                        loan_id=loan_id,
                        month_index=int(month_idx) if pd.notna(month_idx) else 0,
                        field=field,
                        rule=rule.get("name"),
                        observed_value=record[field],
                        expected_value=f"<= {max_val}",
                        severity=severity,
                        message=f"{field} = {record[field]} > {max_val}"
                    ))
            
            elif rule_type == "range":
                min_val = rule.get("min_value")
                max_val = rule.get("max_value")
                if record[field] < min_val or record[field] > max_val:
                    failures.append(ValidationFailure(
                        loan_id=loan_id,
                        month_index=int(month_idx) if pd.notna(month_idx) else 0,
                        field=field,
                        rule=rule.get("name"),
                        observed_value=record[field],
                        expected_value=f"{min_val} - {max_val}",
                        severity=severity,
                        message=f"{field} = {record[field]} outside range [{min_val}, {max_val}]"
                    ))
            
            elif rule_type == "comparison":
                compare_to = rule.get("compare_to")
                operator = rule.get("operator", "<=")
                
                if compare_to in record.index and pd.notna(record[compare_to]):
                    comp_val = record[compare_to]
                    passes = False
                    
                    if operator == "<=":
                        passes = record[field] <= comp_val
                    elif operator == ">=":
                        passes = record[field] >= comp_val
                    elif operator == "<":
                        passes = record[field] < comp_val
                    elif operator == ">":
                        passes = record[field] > comp_val
                    elif operator == "==":
                        passes = record[field] == comp_val
                    
                    if not passes:
                        failures.append(ValidationFailure(
                            loan_id=loan_id,
                            month_index=int(month_idx) if pd.notna(month_idx) else 0,
                            field=field,
                            rule=rule.get("name"),
                            observed_value=record[field],
                            expected_value=f"{operator} {comp_val}",
                            severity=severity,
                            message=f"{field} = {record[field]} not {operator} {compare_to} = {comp_val}"
                        ))
            
            elif rule_type == "date_comparison":
                compare_to = rule.get("compare_to")
                operator = rule.get("operator", ">=")
                
                if compare_to in record.index and pd.notna(record[compare_to]):
                    try:
                        date1 = pd.to_datetime(record[field])
                        date2 = pd.to_datetime(record[compare_to])
                        
                        passes = False
                        if operator == ">=":
                            passes = date1 >= date2
                        elif operator == "<=":
                            passes = date1 <= date2
                        elif operator == ">":
                            passes = date1 > date2
                        elif operator == "<":
                            passes = date1 < date2
                        
                        if not passes:
                            failures.append(ValidationFailure(
                                loan_id=loan_id,
                                month_index=int(month_idx) if pd.notna(month_idx) else 0,
                                field=field,
                                rule=rule.get("name"),
                                observed_value=str(record[field]),
                                expected_value=f"{operator} {record[compare_to]}",
                                severity=severity,
                                message=f"{field} {operator} {compare_to} violated"
                            ))
                    except Exception as e:
                        logger.debug(f"Date comparison failed: {e}")
            
            elif rule_type == "status_consistency":
                status_field = rule.get("status_field")
                if status_field in record.index and pd.notna(record[status_field]):
                    # If DPD > 0, status should indicate delinquency
                    if record[field] > 0:
                        status = record[status_field]
                        if status in ["CURRENT"]:
                            failures.append(ValidationFailure(
                                loan_id=loan_id,
                                month_index=int(month_idx) if pd.notna(month_idx) else 0,
                                field=field,
                                rule=rule.get("name"),
                                observed_value=record[field],
                                expected_value="DPD > 0 requires delinquent status",
                                severity=severity,
                                message=f"DPD = {record[field]} but status = {status}"
                            ))
        
        except Exception as e:
            logger.debug(f"Error applying rule {rule.get('name')}: {e}")
        
        return failures
    
    @staticmethod
    def save_validation_report(
        failures: List[ValidationFailure],
        output_dir: Path
    ) -> None:
        """Save validation failures to CSV"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(failures) == 0:
            logger.info("No validation failures")
            return
        
        # Convert to DataFrame
        failures_data = [f.to_dict() for f in failures]
        df = pd.DataFrame(failures_data)
        
        # Save to CSV
        output_path = output_dir / "validation_failures.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Validation report saved to {output_path}")
        
        # Summary
        summary = {
            "total_failures": len(failures),
            "critical": len([f for f in failures if f.severity == "CRITICAL"]),
            "high": len([f for f in failures if f.severity == "HIGH"]),
            "medium": len([f for f in failures if f.severity == "MEDIUM"]),
            "low": len([f for f in failures if f.severity == "LOW"]),
            "by_rule": {}
        }
        
        for f in failures:
            if f.rule not in summary["by_rule"]:
                summary["by_rule"][f.rule] = 0
            summary["by_rule"][f.rule] += 1
        
        summary_path = output_dir / "validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Validation summary: {summary['critical']} critical, {summary['high']} high severity")
