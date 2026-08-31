"""
Synthetic loan data generator for development and testing
DEMO DATA - NOT FOR CHALLENGE SUBMISSION
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SyntheticLoanDataGenerator:
    """
    Generate realistic synthetic loan-month panel data for testing
    This data is DEMO ONLY and should not be submitted as challenge results
    """
    
    # Loan states
    STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    SERVICERS = ["Servicer_A", "Servicer_B", "Servicer_C", "Servicer_D"]
    LOAN_PURPOSES = ["PURCHASE", "REFINANCE"]
    OCCUPANCY = ["PRIMARY", "INVESTMENT", "SECOND_HOME"]
    PROPERTY_TYPES = ["SINGLE_FAMILY", "MULTI_FAMILY", "CONDO"]
    CREDIT_BANDS = ["0-600", "600-650", "650-700", "700-750", "750+"]
    LTV_BANDS = ["LTV_0-60", "LTV_60-70", "LTV_70-80", "LTV_80-90", "LTV_90-100"]
    DTI_BANDS = ["DTI_0-36", "DTI_36-50", "DTI_50-63", "DTI_63+"]
    
    @staticmethod
    def generate_training_data(
        n_loans: int = 10000,
        months_per_loan: int = 12,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic training data
        
        Args:
            n_loans: Number of unique loans
            months_per_loan: Average months per loan
            seed: Random seed
            
        Returns:
            Tuple of (training_df, static_attributes_df)
        """
        np.random.seed(seed)
        logger.info(f"Generating synthetic training data: {n_loans} loans × ~{months_per_loan} months")
        
        # Generate loan IDs and their lifecycles
        loan_ids = np.arange(1, n_loans + 1)
        origination_dates = pd.date_range("2018-01-01", "2022-12-01", periods=n_loans)
        
        records = []
        
        for i, loan_id in enumerate(loan_ids):
            # Loan characteristics (fixed at origination)
            state = np.random.choice(SyntheticLoanDataGenerator.STATES)
            servicer = np.random.choice(SyntheticLoanDataGenerator.SERVICERS)
            purpose = np.random.choice(SyntheticLoanDataGenerator.LOAN_PURPOSES)
            occupancy = np.random.choice(SyntheticLoanDataGenerator.OCCUPANCY)
            property_type = np.random.choice(SyntheticLoanDataGenerator.PROPERTY_TYPES)
            credit_band = np.random.choice(SyntheticLoanDataGenerator.CREDIT_BANDS)
            ltv_band = np.random.choice(SyntheticLoanDataGenerator.LTV_BANDS)
            dti_band = np.random.choice(SyntheticLoanDataGenerator.DTI_BANDS)
            
            original_balance = np.random.uniform(100000, 500000)
            interest_rate = np.random.uniform(2.5, 7.5)
            loan_term = np.random.choice([180, 240, 360])  # 15, 20, 30 year
            origination_date = origination_dates[i]
            
            # Loan lifecycle events (random outcomes)
            lifecycle = np.random.choice(["current", "delinquent", "default", "prepaid"], p=[0.7, 0.15, 0.10, 0.05])
            
            # Variable months per loan based on lifecycle
            if lifecycle == "current":
                n_months = np.random.randint(1, months_per_loan + 1)
            elif lifecycle == "delinquent":
                n_months = np.random.randint(1, max(2, months_per_loan))
            elif lifecycle == "default":
                n_months = np.random.randint(6, max(7, months_per_loan))
            else:  # prepaid
                n_months = np.random.randint(8, max(9, int(months_per_loan * 0.8)))
            
            # Generate monthly records
            for month_idx in range(n_months):
                reporting_date = origination_date + pd.DateOffset(months=month_idx)
                month_index = month_idx
                
                # Age
                loan_age = month_idx + 1
                remaining = max(1, loan_term - month_idx)
                
                # Balance trajectory
                balance_pct = 1.0 - (month_idx / loan_term) * (0.8 + np.random.uniform(-0.1, 0.1))
                balance_pct = max(0, min(1, balance_pct))
                current_balance = original_balance * balance_pct
                
                # Status evolution
                if lifecycle == "current" and month_idx == n_months - 1:
                    status = "CURRENT"
                    dpd = 0
                    default_flag = 0
                    prepay_flag = 0
                elif lifecycle == "delinquent":
                    status = "DELINQUENT" if month_idx < n_months - 1 else "DELINQUENT"
                    dpd = np.random.randint(30, 180) if month_idx == n_months - 1 else np.random.randint(1, 120)
                    default_flag = 0
                    prepay_flag = 0
                elif lifecycle == "default":
                    status = "DEFAULT" if month_idx >= n_months - 2 else "DELINQUENT"
                    dpd = 150 if month_idx >= n_months - 2 else np.random.randint(60, 150)
                    default_flag = 1 if month_idx >= n_months - 2 else 0
                    prepay_flag = 0
                else:  # prepaid
                    status = "PAID_OFF" if month_idx == n_months - 1 else "CURRENT"
                    dpd = 0
                    default_flag = 0
                    prepay_flag = 1 if month_idx == n_months - 1 else 0
                
                # Future targets (only for records where we have future data)
                has_future = (month_idx < n_months - 3)
                next_3m_delinq = 1 if (lifecycle in ["delinquent", "default"] and has_future) else np.random.choice([0, 1], p=[0.9, 0.1])
                next_6m_delinq = 1 if (lifecycle in ["delinquent", "default"] and has_future) else np.random.choice([0, 1], p=[0.85, 0.15])
                next_12m_default = 1 if (lifecycle == "default" and has_future) else np.random.choice([0, 1], p=[0.95, 0.05])
                next_12m_prepay = 1 if (lifecycle == "prepaid" and has_future and month_idx > 6) else np.random.choice([0, 1], p=[0.92, 0.08])
                
                record = {
                    "loan_id": loan_id,
                    "month_index": month_index,
                    "reporting_month": reporting_date.strftime("%Y-%m"),
                    "origination_month": origination_date.strftime("%Y-%m"),
                    "loan_age_months": loan_age,
                    "remaining_term_months": remaining,
                    "original_balance": original_balance,
                    "current_balance": current_balance,
                    "interest_rate": interest_rate,
                    "credit_score_band": credit_band,
                    "ltv_band": ltv_band,
                    "dti_band": dti_band,
                    "state": state,
                    "loan_purpose": purpose,
                    "occupancy_type": occupancy,
                    "property_type": property_type,
                    "servicer_name": servicer,
                    "current_status": status,
                    "days_past_due": dpd,
                    "modification_flag": np.random.choice([0, 1], p=[0.95, 0.05]),
                    "prepayment_flag": prepay_flag,
                    "default_flag": default_flag,
                    "loss_severity_band": "LOW" if default_flag == 0 else np.random.choice(["LOW", "MEDIUM", "HIGH"]),
                    "last_updated_at": reporting_date.strftime("%Y-%m-%d"),
                    "source_system": np.random.choice([servicer + "_SYSTEM", "INVESTOR_SYSTEM"], p=[0.7, 0.3]),
                    "document_status": np.random.choice(["COMPLETE", "INCOMPLETE", "PENDING"], p=[0.85, 0.10, 0.05]),
                    # Targets
                    "next_3m_delinquency_flag": next_3m_delinq,
                    "next_6m_delinquency_flag": next_6m_delinq,
                    "next_12m_default_flag": next_12m_default,
                    "next_12m_prepayment_flag": next_12m_prepay,
                    "next_state": np.random.choice(["CURRENT", "DELINQUENT", "DEFAULT", "PREPAID"]),
                    "exception_required": 1 if (dpd > 90 or default_flag == 1) else 0,
                    "exception_type": "DELINQUENCY_ALERT" if dpd > 90 else ("DEFAULT_ALERT" if default_flag == 1 else "NONE"),
                }
                
                records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} total records")
        
        # Generate static attributes (subset)
        static_records = []
        for loan_id in loan_ids:
            first_record = df[df["loan_id"] == loan_id].iloc[0]
            static = {
                "loan_id": loan_id,
                "origination_date": first_record["origination_month"],
                "original_balance": first_record["original_balance"],
                "loan_term_months": 180 if first_record["remaining_term_months"] > 300 else (
                    240 if first_record["remaining_term_months"] > 200 else 360
                ),
                "interest_rate": first_record["interest_rate"],
                "credit_score_band": first_record["credit_score_band"],
                "ltv_band": first_record["ltv_band"],
                "dti_band": first_record["dti_band"],
                "state": first_record["state"],
                "loan_purpose": first_record["loan_purpose"],
                "property_type": first_record["property_type"],
                "occupancy_type": first_record["occupancy_type"],
            }
            static_records.append(static)
        
        static_df = pd.DataFrame(static_records)
        
        return df, static_df
    
    @staticmethod
    def generate_test_data(
        training_df: pd.DataFrame,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic test data (similar structure, no targets)
        
        Args:
            training_df: Training DataFrame to use as template
            seed: Random seed
            
        Returns:
            Test DataFrame
        """
        np.random.seed(seed + 1000)
        
        # Sample random loans and extend with new months
        test_records = []
        sample_loans = training_df["loan_id"].sample(n=min(5000, len(training_df["loan_id"].unique()))).unique()
        
        for loan_id in sample_loans:
            loan_data = training_df[training_df["loan_id"] == loan_id].iloc[-1:].copy()
            
            # Add 1-3 future months
            for month_offset in range(1, np.random.randint(2, 4)):
                future_record = loan_data.iloc[-1].copy()
                future_record["month_index"] += month_offset
                new_date = pd.to_datetime(future_record["reporting_month"]) + pd.DateOffset(months=month_offset)
                future_record["reporting_month"] = new_date.strftime("%Y-%m")
                future_record["loan_age_months"] += month_offset
                future_record["remaining_term_months"] = max(1, future_record["remaining_term_months"] - month_offset)
                
                # Remove targets for test set
                for col in ["next_3m_delinquency_flag", "next_6m_delinquency_flag",
                           "next_12m_default_flag", "next_12m_prepayment_flag",
                           "next_state", "exception_required", "exception_type"]:
                    future_record[col] = np.nan
                
                test_records.append(future_record)
        
        test_df = pd.DataFrame(test_records)
        logger.info(f"Generated {len(test_df)} test records")
        
        return test_df
    
    @classmethod
    def generate_and_save(cls, data_dir: Path, seed: int = 42) -> None:
        """
        Generate all synthetic data and save to CSV files
        
        Args:
            data_dir: Directory to save files
            seed: Random seed
        """
        data_dir = Path(data_dir)
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("GENERATING SYNTHETIC DEMO DATA")
        logger.info("THIS DATA IS FOR DEVELOPMENT/TESTING ONLY")
        logger.info("=" * 60)
        
        # Generate data
        train_df, static_df = cls.generate_training_data(n_loans=10000, months_per_loan=12, seed=seed)
        test_df = cls.generate_test_data(train_df, seed=seed)
        
        # Save files
        train_path = raw_dir / "loan_monthly_performance_train.csv"
        test_path = raw_dir / "loan_monthly_performance_test.csv"
        static_path = raw_dir / "loan_static_attributes.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        static_df.to_csv(static_path, index=False)
        
        logger.info(f"Saved training data: {train_path}")
        logger.info(f"Saved test data: {test_path}")
        logger.info(f"Saved static data: {static_path}")
        logger.info("=" * 60)
        logger.info("DEMO DATA GENERATION COMPLETE")
        logger.info("When real organizer data is placed in data/raw/, it will be used instead")
        logger.info("=" * 60)


if __name__ == "__main__":
    # Configure logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Generate demo data in data/ directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    SyntheticLoanDataGenerator.generate_and_save(data_dir, seed=42)
