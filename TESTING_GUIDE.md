# Testing Custom Loan Records - Quick Start Guide

## 🎯 Overview

The LIQ project includes **3 built-in ways** to test with custom input data:

1. **FastAPI REST API** - Send HTTP requests
2. **Dashboard Copilot** - Interactive web interface
3. **Python Script** - Batch testing with full reports

---

## 🔌 **Method 1: REST API (Recommended for Single Records)**

### Start the API Server

```powershell
cd d:\LIQ\LIQ
python -m uvicorn app:app --reload
```

Output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example: Test a Single Loan (PowerShell)

```powershell
# Create test data
$loanData = @{
    loan_id = 5555
    current_balance = 250000
    original_balance = 300000
    days_past_due = 45
    credit_score_band = "700-749"
} | ConvertTo-Json

# Send to API
$response = Invoke-RestMethod -Uri "http://localhost:8000/review-loan" `
    -Method POST `
    -Body $loanData `
    -ContentType "application/json"

# Display results
$response | ConvertTo-Json
```

### Example: Test Many Loans (Python)

```python
import requests
import json

# API URL
api_url = "http://localhost:8000/review-loan"

# Test multiple loans
test_loans = [
    {
        "loan_id": 1001,
        "current_balance": 400000,
        "original_balance": 400000,
        "days_past_due": 0,
        "credit_score_band": "750+"
    },
    {
        "loan_id": 1002,
        "current_balance": 250000,
        "original_balance": 350000,
        "days_past_due": 90,
        "credit_score_band": "650-699"
    },
]

# Get reviews for each loan
results = []
for loan in test_loans:
    response = requests.post(api_url, json=loan)
    results.append(response.json())
    print(f"Loan {loan['loan_id']}: Risk={results[-1]['risk_score']:.2f}")

# Save results
with open("api_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} results to api_results.json")
```

### API Response Format

```json
{
  "loan_id": 5555,
  "model_name": "heuristic-grounded-reviewer",
  "risk_score": 0.65,
  "recommended_action": "escalate_for_review",
  "summary": "Loan 5555 shows a 0.65 risk profile based on balance deterioration...",
  "evidence": [
    {
      "field": "current_balance",
      "value": 250000
    },
    {
      "field": "days_past_due",
      "value": 45
    },
    {
      "field": "credit_score_band",
      "value": "700-749"
    }
  ],
  "model_summary": {
    "status": "heuristic review"
  }
}
```

---

## 📊 **Method 2: Interactive Dashboard (Visual)**

### Start Dashboard

```powershell
cd d:\LIQ\LIQ
streamlit run dashboard/app.py
```

Opens: http://localhost:8501

### Navigate to Reviewer Copilot

1. Click sidebar → **07_Reviewer_Copilot**
2. Shows latest loan reviews with:
   - Risk scores
   - Recommended actions
   - Evidence fields
   - Confidence indicators

---

## 🐍 **Method 3: Python Test Script (Batch + Full Reports)**

### Run the Test Script

```powershell
cd d:\LIQ\LIQ
python test_custom_input.py
```

### What It Does

1. **Creates 3 sample test loans**:
   - Loan 1001: Good standing (risk score 0.15)
   - Loan 1002: Delinquent (risk score 0.78)
   - Loan 1003: About to prepay (risk score 0.25)

2. **Generates predictions** for each loan:
   - Delinquency probability
   - Default probability
   - Prepayment probability

3. **Creates detailed reports** with:
   - Risk scoring
   - Recommended actions
   - Evidence & drivers
   - LLM grounded summaries

4. **Saves 3 output files**:
   - `outputs/custom_predictions.csv`
   - `outputs/custom_loan_reports.json`
   - `outputs/custom_loan_analysis.csv`

### Output Example

```
================================================================================
CUSTOM LOAN TESTING & PREDICTION TOOL
================================================================================

[1] Creating test loan records...
✅ Created 3 test loans

Test Loans:
   loan_id current_status  days_past_due credit_score_band  current_balance
      1001        CURRENT              0               750+         390000.0
      1002     DELINQUENT             90          650-699         280000.0
      1003        CURRENT              0               750+         470000.0

[2] Making predictions using heuristic models...
✅ Predictions generated

[3] Generating detailed reviewer reports...

LOAN #1001
  Status: CURRENT
  Days Past Due: 0
  Risk Score: 0.15
  Recommended Action: monitor
  Summary: Loan 1001 shows a 0.15 risk profile based on balance deterioration...

LOAN #1002
  Status: DELINQUENT
  Days Past Due: 90
  Risk Score: 0.78
  Recommended Action: credit_review_and_collection
  Summary: Loan 1002 shows a 0.78 risk profile based on...

LOAN #1003
  Status: CURRENT
  Days Past Due: 0
  Risk Score: 0.25
  Recommended Action: monitor
  Summary: Loan 1003 shows a 0.25 risk profile...

[4] Saving results to files...
✅ Predictions saved to: outputs/custom_predictions.csv
✅ Reports saved to: outputs/custom_loan_reports.json
✅ Combined analysis saved to: outputs/custom_loan_analysis.csv

================================================================================
✅ TEST COMPLETE!
================================================================================
```

### Customize the Test Script

Edit `test_custom_input.py` to test YOUR data:

```python
def create_test_loan_records():
    """Modify this function to add your own loans."""
    
    test_records = pd.DataFrame([
        {
            'loan_id': YOUR_LOAN_ID,
            'current_balance': YOUR_BALANCE,
            'original_balance': YOUR_ORIG_BALANCE,
            'days_past_due': YOUR_DPD,
            'current_status': 'CURRENT',  # or 'DELINQUENT', etc.
            # ... add other fields
        },
    ])
    return test_records
```

---

## 🎯 Using Trained Models for Advanced Predictions

To use the **actual trained CatBoost models** instead of heuristics:

```python
import joblib
import pandas as pd

# Load a trained model
model_path = "outputs/models/next_3m_delinquency_flag_catboost.joblib"
model = joblib.load(model_path)

# Prepare features (must match training features)
test_features = pd.DataFrame([
    {
        'days_past_due': 45,
        'current_balance': 250000,
        'days_delinquent_history': 30,
        'balance_deterioration': 0.15,
        # ... 19 total features required
    }
])

# Get prediction
probability = model.predict_proba(test_features)[0][1]
print(f"Delinquency Probability: {probability:.2%}")
```

---

## 📁 Output Files

After testing, you'll have:

| File | Format | Contains |
|------|--------|----------|
| `custom_predictions.csv` | CSV | Loan ID + probabilities (delinquency, default, prepayment) |
| `custom_loan_reports.json` | JSON | Detailed risk scores, actions, evidence for each loan |
| `custom_loan_analysis.csv` | CSV | Combined: predictions + risk scores + actions |

---

## 🔗 Quick Reference

### Start Everything (Dev Mode)

Terminal 1 - API:
```powershell
python -m uvicorn app:app --reload
```

Terminal 2 - Dashboard:
```powershell
streamlit run dashboard/app.py
```

Terminal 3 - Run Tests:
```powershell
python test_custom_input.py
```

### URLs
- **API**: http://localhost:8000 (or :8000/docs for Swagger)
- **Dashboard**: http://localhost:8501
- **Outputs**: `outputs/custom_*` files

---

## ❓ Common Questions

**Q: How do I test with my own CSV file?**

A: Modify `test_custom_input.py`:
```python
# Load from CSV instead of creating test data
test_records = pd.read_csv("my_loans.csv")
predictions = make_predictions(test_records)
```

**Q: Which method is fastest?**

A: FastAPI is fastest for single/few loans. Python script is fastest for batch (100+ loans).

**Q: Can I get SHAP explanations for custom loans?**

A: Yes, modify the script to load SHAP explainer:
```python
from src.explainability.shap_explainer import ShapExplainer
explainer = ShapExplainer.load("outputs/shap_explainer.pkl")
shap_values = explainer.explain(test_features)
```

**Q: Do I need to retrain models first?**

A: No. Trained models are already saved in `outputs/models/`. Just use them directly.

---

**Created**: 2026-08-30  
**Project**: Loan Performance Intelligence Engine (LIQ)
