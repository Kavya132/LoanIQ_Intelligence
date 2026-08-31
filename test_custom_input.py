"""
Test script: Make predictions on custom loan records
and generate reports with full model outputs.

Usage:
    python test_custom_input.py
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.trainer import ModelTrainer
from src.llm.copilot import GroundedReviewer
from src.config.settings import get_settings


def create_test_loan_records():
    """Create sample test loan records for prediction."""
    
    test_records = pd.DataFrame([
        # Loan 1: Good standing
        {
            'loan_id': 1001,
            'month_index': 0,
            'reporting_month': '2022-01',
            'origination_month': '2020-01',
            'loan_age_months': 24,
            'remaining_term_months': 336,
            'original_balance': 400000,
            'current_balance': 390000,
            'interest_rate': 3.5,
            'credit_score_band': '750+',
            'ltv_band': 'LTV_75-90',
            'dti_band': 'DTI_38-50',
            'state': 'CA',
            'loan_purpose': 'PURCHASE',
            'occupancy_type': 'PRIMARY',
            'property_type': 'SINGLE_FAMILY',
            'servicer_name': 'Servicer_A',
            'current_status': 'CURRENT',
            'days_past_due': 0,
            'modification_flag': 0,
            'prepayment_flag': 0,
            'default_flag': 0,
            'loss_severity_band': 'LOW',
            'last_updated_at': '2022-01-15',
            'source_system': 'INVESTOR_SYSTEM',
            'document_status': 'COMPLETE',
        },
        # Loan 2: Delinquent - high risk
        {
            'loan_id': 1002,
            'month_index': 0,
            'reporting_month': '2022-01',
            'origination_month': '2019-06',
            'loan_age_months': 31,
            'remaining_term_months': 329,
            'original_balance': 350000,
            'current_balance': 280000,
            'interest_rate': 4.2,
            'credit_score_band': '650-699',
            'ltv_band': 'LTV_90-100',
            'dti_band': 'DTI_50-63',
            'state': 'TX',
            'loan_purpose': 'REFINANCE',
            'occupancy_type': 'SECOND_HOME',
            'property_type': 'CONDO',
            'servicer_name': 'Servicer_B',
            'current_status': 'DELINQUENT',
            'days_past_due': 90,
            'modification_flag': 1,
            'prepayment_flag': 0,
            'default_flag': 0,
            'loss_severity_band': 'MEDIUM',
            'last_updated_at': '2022-01-10',
            'source_system': 'Servicer_B_SYSTEM',
            'document_status': 'COMPLETE',
        },
        # Loan 3: About to prepay - lower risk
        {
            'loan_id': 1003,
            'month_index': 0,
            'reporting_month': '2022-01',
            'origination_month': '2018-01',
            'loan_age_months': 49,
            'remaining_term_months': 311,
            'original_balance': 500000,
            'current_balance': 470000,
            'interest_rate': 2.8,
            'credit_score_band': '750+',
            'ltv_band': 'LTV_60-75',
            'dti_band': 'DTI_25-38',
            'state': 'NY',
            'loan_purpose': 'PURCHASE',
            'occupancy_type': 'PRIMARY',
            'property_type': 'SINGLE_FAMILY',
            'servicer_name': 'Servicer_C',
            'current_status': 'CURRENT',
            'days_past_due': 0,
            'modification_flag': 0,
            'prepayment_flag': 1,  # High prepayment flag!
            'default_flag': 0,
            'loss_severity_band': 'LOW',
            'last_updated_at': '2022-01-12',
            'source_system': 'INVESTOR_SYSTEM',
            'document_status': 'COMPLETE',
        },
    ])
    
    return test_records


def make_predictions(test_df):
    """Generate predictions for test records."""
    
    print("\n" + "="*80)
    print("MAKING PREDICTIONS ON CUSTOM TEST DATA")
    print("="*80)
    
    predictions = {
        'loan_id': test_df['loan_id'].values,
        'current_balance': test_df['current_balance'].values,
        'days_past_due': test_df['days_past_due'].values,
        'current_status': test_df['current_status'].values,
    }
    
    # Simple heuristic model for demo
    for idx, row in test_df.iterrows():
        # Delinquency probability: based on days_past_due
        dpd = float(row['days_past_due'])
        delinq_prob = min(1.0, dpd / 365.0)
        predictions.setdefault('delinquency_prob', []).append(delinq_prob)
        
        # Default probability: based on balance deterioration
        balance_ratio = row['current_balance'] / max(row['original_balance'], 1)
        default_prob = max(0.0, min(1.0, (1 - balance_ratio) * 0.8))
        predictions.setdefault('default_prob', []).append(default_prob)
        
        # Prepayment probability: based on prepayment flag + low DPD
        prep_flag = row['prepayment_flag']
        prepped_prob = 0.7 if (prep_flag == 1 and dpd == 0) else 0.1
        predictions.setdefault('prepayment_prob', []).append(prepped_prob)
    
    pred_df = pd.DataFrame(predictions)
    return pred_df


def generate_reports(test_df, predictions_df):
    """Generate detailed reports for each loan."""
    
    print("\n" + "="*80)
    print("GENERATING DETAILED REPORTS")
    print("="*80 + "\n")
    
    reviewer = GroundedReviewer()
    
    reports = []
    
    for idx in range(len(test_df)):
        loan_row = test_df.iloc[idx]
        pred_row = predictions_df.iloc[idx]
        
        # Create review with model predictions and evidence
        review = reviewer.review_case(
            loan_record=loan_row.to_dict(),
            evidence=[
                {
                    "field": "delinquency_probability",
                    "value": f"{pred_row['delinquency_prob']:.2%}"
                },
                {
                    "field": "default_probability",
                    "value": f"{pred_row['default_prob']:.2%}"
                },
                {
                    "field": "prepayment_probability",
                    "value": f"{pred_row['prepayment_prob']:.2%}"
                },
            ]
        )
        
        reports.append(review)
        
        # Print summary
        print(f"LOAN #{loan_row['loan_id']}")
        print(f"  Status: {loan_row['current_status']}")
        print(f"  Days Past Due: {loan_row['days_past_due']}")
        print(f"  Risk Score: {review['risk_score']:.2f}")
        print(f"  Recommended Action: {review['recommended_action']}")
        print(f"  Summary: {review['summary']}")
        print()
    
    return reports


def save_results(test_df, predictions_df, reports):
    """Save all results to output files."""
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save predictions CSV
    pred_csv = output_dir / "custom_predictions.csv"
    predictions_df.to_csv(pred_csv, index=False)
    print(f"✅ Predictions saved to: {pred_csv}")
    
    # Save detailed reports as JSON
    reports_json = output_dir / "custom_loan_reports.json"
    with reports_json.open("w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"✅ Reports saved to: {reports_json}")
    
    # Save combined CSV with predictions + reports
    combined_df = test_df[['loan_id', 'current_status', 'days_past_due', 'current_balance']].copy()
    combined_df['delinquency_prob'] = predictions_df['delinquency_prob']
    combined_df['default_prob'] = predictions_df['default_prob']
    combined_df['prepayment_prob'] = predictions_df['prepayment_prob']
    
    # Add risk scores from reports
    combined_df['risk_score'] = [r['risk_score'] for r in reports]
    combined_df['recommended_action'] = [r['recommended_action'] for r in reports]
    
    combined_csv = output_dir / "custom_loan_analysis.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"✅ Combined analysis saved to: {combined_csv}")
    
    print("\n" + "="*80)
    print("ALL RESULTS SAVED TO outputs/")
    print("="*80)


def main():
    """Main test workflow."""
    
    print("\n" + "="*80)
    print("CUSTOM LOAN TESTING & PREDICTION TOOL")
    print("="*80)
    
    # Step 1: Create test records
    print("\n[1] Creating test loan records...")
    test_records = create_test_loan_records()
    print(f"✅ Created {len(test_records)} test loans")
    print("\nTest Loans:")
    print(test_records[['loan_id', 'current_status', 'days_past_due', 
                        'credit_score_band', 'current_balance']].to_string(index=False))
    
    # Step 2: Make predictions
    print("\n[2] Making predictions using heuristic models...")
    predictions = make_predictions(test_records)
    print("✅ Predictions generated")
    
    # Step 3: Generate detailed reports
    print("\n[3] Generating detailed reviewer reports...")
    reports = generate_reports(test_records, predictions)
    print("✅ Reports generated")
    
    # Step 4: Save all results
    print("\n[4] Saving results to files...")
    save_results(test_records, predictions, reports)
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. View predictions: outputs/custom_predictions.csv")
    print("  2. View reports: outputs/custom_loan_reports.json")
    print("  3. View analysis: outputs/custom_loan_analysis.csv")
    print("  4. Load trained models for advanced predictions (see code comments)")


if __name__ == "__main__":
    main()
