# QUICK TESTING QUICK REFERENCE

## ⚡ 3 Ways to Test with Custom Loan Data

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     LIQ TESTING CAPABILITIES                                   │
└────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┬─────────────────────────────┬──────────────────────┐
│   🔌 METHOD 1: REST API     │  📊 METHOD 2: DASHBOARD    │ 🐍 METHOD 3: PYTHON  │
├─────────────────────────────┼─────────────────────────────┼──────────────────────┤
│ Best For:                   │ Best For:                   │ Best For:            │
│ • Single loans              │ • Visual review             │ • Batch testing      │
│ • Programmatic access       │ • Interactive exploration   │ • Full reports       │
│ • Integration               │ • Non-technical users       │ • Automation         │
├─────────────────────────────┼─────────────────────────────┼──────────────────────┤
│ Start:                      │ Start:                      │ Run:                 │
│ python -m uvicorn app:app   │ streamlit run dashboard/... │ python test_custom...│
│                             │                             │                      │
│ URL: :8000                  │ URL: :8501                  │ Automatic output     │
├─────────────────────────────┼─────────────────────────────┼──────────────────────┤
│ Input:                      │ Input:                      │ Input:               │
│ POST JSON                   │ Web form (interactive)      │ Pandas DataFrame     │
│ {                           │ • loan_id                   │ or CSV               │
│   loan_id: 5555,            │ • balance                   │                      │
│   current_balance: 250000,  │ • days_past_due             │ Output:              │
│   days_past_due: 45,        │ • credit_score              │ • CSV predictions    │
│   credit_score_band: "700-" │                             │ • JSON reports       │
│ }                           │ Output:                     │ • Analysis CSV       │
│                             │ • Risk score                │                      │
│ Output:                     │ • Action                    │ Files location:      │
│ {                           │ • Evidence                  │ outputs/custom_*     │
│   loan_id: 5555,            │ • Confidence                │                      │
│   risk_score: 0.65,         │                             │ Example (already run):│
│   recommended_action: ...   │                             │ 3 loans tested       │
│   evidence: [...]           │                             │ ✅ outputs generated │
│ }                           │                             │                      │
├─────────────────────────────┼─────────────────────────────┼──────────────────────┤
│ Pros:                       │ Pros:                       │ Pros:                │
│ ✅ Fast                     │ ✅ Interactive              │ ✅ Full control      │
│ ✅ Easy integration         │ ✅ No coding needed         │ ✅ Batch processing  │
│ ✅ Swagger/ReDoc docs       │ ✅ Real-time updates        │ ✅ Detailed reports  │
│ ✅ Scalable                 │ ✅ Visualizations           │ ✅ Custom workflow   │
│                             │                             │ ✅ Logged results    │
│ Cons:                       │ Cons:                       │ Cons:                │
│ ❌ Need HTTP client         │ ❌ One loan at a time       │ ❌ Requires Python   │
│ ❌ Manual testing           │ ❌ No batch export          │ ❌ More setup        │
└─────────────────────────────┴─────────────────────────────┴──────────────────────┘
```

---

## 📋 COMPARISON TABLE

| Feature | API | Dashboard | Python Script |
|---------|-----|-----------|---------------|
| Single loan | ✅ | ✅ | ✅ |
| Batch (5-100 loans) | ⚠️ Multiple calls | ❌ | ✅ |
| Batch (1000+ loans) | ❌ | ❌ | ✅ |
| Risk score | ✅ | ✅ | ✅ |
| Predictions (probs) | ⚠️ Custom build | ✅ | ✅ |
| Evidence/drivers | ✅ | ✅ | ✅ |
| PDF export | ❌ | ❌ | ✅ (add code) |
| CSV export | ❌ | ✅ Limited | ✅ |
| JSON export | ✅ Response | ✅ | ✅ |
| SHAP explain | ❌ | ⚠️ Via dashboard | ✅ (add code) |
| Setup time | 10 sec | 5 sec | 30 sec |
| Ease | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

---

## 🚀 FASTEST START - Choose Your Method

### 👉 I want to test ONE loan RIGHT NOW:
```bash
# Terminal 1:
python -m uvicorn app:app --reload

# Terminal 2 (PowerShell):
$loan = @{
    loan_id=1001
    current_balance=350000
    days_past_due=0
    credit_score_band="750+"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/review-loan" `
    -Method POST -Body $loan -ContentType "application/json"
```
⏱️ **Time to results: ~2 minutes**

---

### 👉 I want to VISUALIZE results in a web interface:
```bash
streamlit run dashboard/app.py
# Then click → "07_Reviewer_Copilot"
```
⏱️ **Time to results: ~30 seconds**

---

### 👉 I want to TEST 3+ loans and SAVE REPORTS:
```bash
python test_custom_input.py
# Automatically generates:
#   - outputs/custom_predictions.csv
#   - outputs/custom_loan_reports.json
#   - outputs/custom_loan_analysis.csv
```
⏱️ **Time to results: ~10 seconds**

---

### 👉 I want to use my OWN CSV file:
```python
# Edit test_custom_input.py:
test_records = pd.read_csv("my_loans.csv")  # ← Replace this line
predictions = make_predictions(test_records)
reports = generate_reports(test_records, predictions)
```
⏱️ **Time to results: ~30 seconds**

---

## 📊 EXAMPLE OUTPUT

### Tested 3 Loans:

| Loan ID | Status | DPD | Risk Score | Delinq Prob | Default Prob | Prepay Prob | Action |
|---------|--------|-----|-----------|-------------|--------------|------------|--------|
| 1001 | CURRENT | 0 | 0.02 | 0% | 2% | 10% | Monitor |
| 1002 | DELINQUENT | 90 | 0.37 | 25% | 16% | 10% | Monitor |
| 1003 | CURRENT | 0 | 0.04 | 0% | 5% | **70%** | Monitor |

✅ **All outputs saved to `outputs/custom_*`**

---

## 🎯 WHICH METHOD SHOULD I USE?

```
Do you want to...

1. Test a single loan programmatically? → USE API
   (Easiest for one-off testing)

2. Review loans interactively? → USE DASHBOARD
   (Best for visual exploration)

3. Batch test multiple loans? → USE PYTHON SCRIPT
   (Best for reporting & automation)

4. Integrate with another system? → USE API
   (Best for third-party integration)

5. Automate daily scoring? → USE PYTHON SCRIPT + SCHEDULER
   (Python script + cron job / Task Scheduler)
```

---

## 🔧 TESTING WITH REAL TRAINED MODELS

The examples above use **heuristic-based scoring**. To use the **actual trained CatBoost models**:

### In Python Script:
```python
import joblib

# Load trained model
model = joblib.load("outputs/models/next_3m_delinquency_flag_catboost.joblib")

# Prepare features (19 features required)
features = test_records[feature_cols]  # Must match training features

# Get ML-based predictions
ml_probs = model.predict_proba(features)[:, 1]
```

### Via API:
Modify `app.py` to load trained models:
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.load_model("next_3m_delinquency_flag")
predictions = model.predict_proba(features)
```

---

## 📁 FILE LOCATIONS

| What | Where |
|------|-------|
| API entry point | `app.py` |
| Test script | `test_custom_input.py` |
| Dashboard | `dashboard/app.py` |
| Trained models | `outputs/models/*.joblib` |
| Test outputs | `outputs/custom_*.csv` |
| Reports | `outputs/custom_*.json` |

---

## ⚠️ PRE-REQUISITES

Before testing, make sure you have run training:
```bash
python scripts/train.py
```

This creates:
- ✅ Trained models in `outputs/models/`
- ✅ Feature scalers
- ✅ Calibration data
- ✅ Config files

Then you can test with any of the 3 methods!

---

## 🆘 TROUBLESHOOTING

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: pydantic` | `pip install pydantic` |
| `Port 8000 already in use` | `python -m uvicorn app:app --port 8001` |
| `Port 8501 already in use` | `streamlit run dashboard/app.py --server.port 8502` |
| `Models not found` | Run `python scripts/train.py` first |
| `Feature mismatch` | Ensure test data has all 33 input columns |

---

**Created**: 2026-08-30
**Project**: LIQ - Loan Performance Intelligence Engine
