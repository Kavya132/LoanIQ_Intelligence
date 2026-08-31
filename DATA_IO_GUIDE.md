# LIQ Project: Data Input & Output Guide

## 📥 INPUTS (What Goes In)

The project uses 3 main CSV files from `data/raw/`:

### 1. **loan_monthly_performance_train.csv** (66,632 rows)
**Purpose**: Historical loan performance data with targets for model training

**Structure**: Panel data (one row per loan per month)
- 66,632 records
- 33 columns

**Key Columns**:

#### Loan Identity
- `loan_id` - Unique loan identifier
- `month_index` - Month sequence (0, 1, 2, ...)
- `reporting_month` - Calendar month (2018-01, 2018-02, etc.)
- `origination_month` - When loan was created

#### Temporal Features
- `loan_age_months` - Months since origination
- `remaining_term_months` - Months left in loan term

#### Balance & Finance
- `original_balance` - Initial loan amount (e.g., 278,333)
- `current_balance` - Outstanding balance
- `interest_rate` - Loan rate (e.g., 2.99%)

#### Credit & Risk Profile (Static)
- `credit_score_band` - Category: 750+, 700-749, 650-699, <650
- `ltv_band` - Loan-to-value ratio: LTV_90-100, LTV_75-90, etc.
- `dti_band` - Debt-to-income ratio: DTI_50-63, DTI_38-50, etc.

#### Property & Geography (Static)
- `state` - State code (OH, CA, NY, etc.)
- `loan_purpose` - PURCHASE, REFINANCE
- `occupancy_type` - PRIMARY, SECOND_HOME, INVESTOR
- `property_type` - SINGLE_FAMILY, CONDO, etc.

#### Status & Performance (Monthly)
- `current_status` - CURRENT, DELINQUENT, DEFAULTED, PREPAID
- `days_past_due` - Count of delinquent days (0-180+)
- `modification_flag` - Loan modification applied (0/1)
- `prepayment_flag` - Early payment made (0/1)
- `default_flag` - Default occurred (0/1)

#### Loss & Document
- `loss_severity_band` - LOW, MEDIUM, HIGH
- `servicer_name` - Servicer_A, Servicer_B, Servicer_C, Servicer_D
- `document_status` - COMPLETE, INCOMPLETE, MISSING
- `source_system` - INVESTOR_SYSTEM, Servicer_X_SYSTEM
- `last_updated_at` - Last update date

#### **TARGET VARIABLES** (What we predict)
- `next_3m_delinquency_flag` - Will loan be 30+ DPD in next 3 months? (0/1)
- `next_6m_delinquency_flag` - Will loan be 30+ DPD in next 6 months? (0/1)
- `next_12m_default_flag` - Will loan default in next 12 months? (0/1)
- `next_12m_prepayment_flag` - Will loan prepay in next 12 months? (0/1)
- `next_state` - Next status (CURRENT, DELINQUENT, DEFAULTED, PREPAID)
- `exception_required` - Is loan flagged for review? (0/1)
- `exception_type` - DELINQUENCY_ALERT, DEFAULT_WARNING, PREPAYMENT_LIKELY, etc.

**Example Row**:
```
loan_id=1, month_index=0, reporting_month=2018-01
current_status=DELINQUENT, days_past_due=62, current_balance=278,333
credit_score_band=750+, state=OH
next_3m_delinquency_flag=1  ← TARGET: Will stay delinquent in 3 months?
next_12m_default_flag=1     ← TARGET: Will default in 12 months?
```

---

### 2. **loan_monthly_performance_test.csv** (5,857 rows)
**Purpose**: Unlabeled test data for final predictions

**Structure**: Same 33 columns as training, BUT:
- **NO TARGET VARIABLES** (next_3m_delinquency_flag, etc. are missing/null)
- Used to generate predictions for submission

**Example**:
```
loan_id=2000, month_index=0, reporting_month=2022-01
current_status=CURRENT, days_past_due=0, current_balance=350,000
credit_score_band=700-749, state=CA
[next_3m_delinquency_flag = NULL]  ← We must predict this!
```

---

### 3. **loan_static_attributes.csv** (10,000 rows)
**Purpose**: Origination-level (one-time) attributes

**Structure**: One row per loan (not monthly)
- 10,000 unique loans
- 12 columns

**Columns**:
- `loan_id` - Matches train/test
- `original_balance`, `interest_rate`, `loan_term_months`
- `credit_score_band`, `ltv_band`, `dti_band`
- `state`, `loan_purpose`, `property_type`
- `origination_date`, `property_state`

**Purpose**: Enrich monthly data with static origination info

---

## 🔄 PROCESSING PIPELINE

```
DATA FLOW (Training Phase)
╔════════════════════════════════════════════════════════════════════╗
║                    INPUT: 3 CSV Files                             ║
║                                                                    ║
║  train.csv (66k) + test.csv (5.8k) + static.csv (10k)            ║
║                                                                    ║
║  ↓ [DATA LOADING & CLEANING]                                      ║
║    - Load CSVs                                                    ║
║    - Detect encoding (chardet)                                   ║
║    - Clean column types                                          ║
║    - Handle missing values                                       ║
║    - Output: Clean DataFrames                                    ║
║                                                                    ║
║  ↓ [PHASE 1: SCHEMA DISCOVERY]                                   ║
║    - Detect field types (26 expected fields)                     ║
║    - Validate required columns                                   ║
║    - Output: schema_report.json                                  ║
║                                                                    ║
║  ↓ [PHASE 2: DATA PROFILING & QUALITY]                           ║
║    - Column distributions (mean, std, unique, missing)           ║
║    - Outlier detection (IQR, Z-score)                            ║
║    - Record quality scoring                                      ║
║    - Data drift detection (train vs test)                        ║
║    - Output: data_profile.json, data_quality_report.csv          ║
║             drift_report.json, data_intelligence_report.html     ║
║                                                                    ║
║  ↓ [PHASE 3: FEATURE ENGINEERING]                                ║
║    - Create temporal features (lag, rolling averages)            ║
║    - Balance features (ratio, trends)                            ║
║    - Delinquency history                                         ║
║    - Servicer features                                           ║
║    - Output: 19 engineered features                              ║
║                                                                    ║
║  ↓ [PHASE 4: TIME-AWARE SPLIT]                                   ║
║    - Split by reporting_month (not random)                       ║
║    - Train: 59,090 rows (9,493 loans)                            ║
║    - Validation: 7,542 rows (1,467 loans)                        ║
║    - Leakage check: ✅ PASSED                                     ║
║                                                                    ║
║  ↓ [PHASE 5: MODEL TRAINING]                                     ║
║    - Train Logistic Regression (baseline)                        ║
║    - Train CatBoost (improved) for 4 targets:                    ║
║      • next_3m_delinquency_flag (ROC-AUC: 0.7897)               ║
║      • next_6m_delinquency_flag (ROC-AUC: 0.7221)               ║
║      • next_12m_default_flag (ROC-AUC: 0.7825)                 ║
║      • next_12m_prepayment_flag (ROC-AUC: 0.5016)              ║
║    - Calibration (Sigmoid)                                       ║
║    - Output: Trained models in outputs/models/                  ║
║             model_metrics.json                                  ║
║                                                                    ║
║  ↓ [PHASE 6: SURVIVAL ANALYSIS]                                  ║
║    - Lifelines competing-risk model                              ║
║    - States: DEFAULT, PREPAID, DELINQUENT, CURED                ║
║    - Output: survival_analysis_results.json                      ║
║                                                                    ║
║  ↓ [PHASE 7: ANOMALY DETECTION]                                  ║
║    - Isolation Forest (2% contamination)                         ║
║    - Exception classification                                    ║
║    - Output: anomaly_summary.json                                ║
║                                                                    ║
║  ↓ [PHASE 8: SCENARIO SIMULATION]                                ║
║    - Monte Carlo 1000 runs                                       ║
║    - Base / Adverse Credit / High Prepayment scenarios           ║
║    - Output: scenario_report.json, scenario_summary.md           ║
║                                                                    ║
║  ↓ [PHASE 9: EXPLAINABILITY]                                     ║
║    - SHAP feature importance                                     ║
║    - Local explanations                                          ║
║    - Output: feature_importance.json                             ║
║                                                                    ║
║  ↓ [PHASE 10: LLM COPILOT]                                       ║
║    - Grounded reviewer summaries                                 ║
║    - Risk scoring                                                ║
║    - Output: llm_logs.jsonl                                      ║
║                                                                    ║
║  ↓ [EXPORT: SUBMISSION]                                          ║
║    - Generate predictions on test set (5,857 rows)               ║
║    - Output: test_submission.csv                                 ║
║    - Columns: loan_id, next_3m_delinquency_prob,                ║
║              next_6m_delinquency_prob, default_prob, etc.        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## 📤 OUTPUTS (What Comes Out)

### **A. Reports (outputs/)**

#### Data Intelligence Reports
- **`data_profile.json`** (33 columns × metrics)
  ```json
  {
    "loan_id": {
      "dtype": "int64",
      "missing_count": 0,
      "unique_count": 10000,
      "mean": 5000.5,
      "std": 2886.82,
      "min": 1, "max": 10000
    },
    ...
  }
  ```

- **`data_quality_report.csv`** (66,632 rows × quality scores)
  ```
  loan_id, month_index, quality_score, severity, issues
  1,       0,          0.85,          MEDIUM,   ["Missing required field", ...]
  ```

- **`data_quality_summary.json`** (batch statistics)
  ```json
  {
    "total_records": 66632,
    "average_quality_score": 0.700,
    "high_severity_count": 66632,
    "outlier_columns": 2
  }
  ```

- **`data_intelligence_report.html`** (Visual dashboard)
  - Interactive plots
  - Missing value heatmap
  - Distribution charts

#### Drift & Validation
- **`drift_report.json`** (Train vs Test comparison)
  ```json
  {
    "overall_drift_risk": "HIGH",
    "drifted_columns": [
      "current_balance",
      "days_past_due",
      ...
    ]
  }
  ```

- **`column_drifts.csv`** (Per-column drift metrics)

#### Schema
- **`schema_report.json`** (Field detection & validation)
- **`schema_report.html`** (Visual schema report)

---

### **B. Model Reports (outputs/)**

#### Performance Metrics
- **`model_metrics.json`** (All model evaluation metrics)
  ```json
  [
    {
      "target": "next_3m_delinquency_flag",
      "model_type": "baseline",
      "roc_auc": 0.5000,
      "f1": 0.0000,
      "accuracy": 0.6387
    },
    {
      "target": "next_3m_delinquency_flag",
      "model_type": "catboost",
      "roc_auc": 0.7897,
      "f1": 0.6450,
      "accuracy": 0.8126,
      "calibrated_brier": 0.1092
    }
  ]
  ```

- **`model_comparison.csv`** (Baseline vs CatBoost side-by-side)

#### Trained Models
- **`outputs/models/`** (Binary joblib files)
  - `next_3m_delinquency_flag_baseline.joblib`
  - `next_3m_delinquency_flag_catboost.joblib`
  - `next_6m_delinquency_flag_baseline.joblib`
  - `next_6m_delinquency_flag_catboost.joblib`
  - `next_12m_default_flag_baseline.joblib`
  - `next_12m_default_flag_catboost.joblib`
  - `next_12m_prepayment_flag_baseline.joblib`
  - `next_12m_prepayment_flag_catboost.joblib`
  - `exception_required_baseline.joblib`
  - `exception_required_catboost.joblib`

#### Explainability
- **`feature_importance.json`** (SHAP-based top 20 features)
  ```json
  {
    "next_3m_delinquency_flag": {
      "top_features": [
        {"name": "days_past_due", "importance": 0.35},
        {"name": "current_status", "importance": 0.28},
        ...
      ]
    }
  }
  ```

---

### **C. Advanced Analysis (outputs/)**

#### Survival Analysis
- **`survival_analysis_results.json`** (Time-to-event modeling)
  ```json
  {
    "competing_risks": {
      "state": ["DEFAULT", "PREPAID", "DELINQUENT", "CURED"],
      "cumulative_incidence": [...],
      "confidence_intervals": [...]
    }
  }
  ```

#### Anomaly Detection
- **`anomaly_summary.json`** (Suspicious records)
  ```json
  {
    "total_anomalies": 1330,
    "anomaly_rate": 0.02,
    "high_risk_records": [
      {
        "loan_id": 5432,
        "month_index": 12,
        "anomaly_score": 0.95,
        "drivers": ["balance exceeds 110% of original", "DPD mismatch with status"]
      }
    ]
  }
  ```

#### Scenario Simulation
- **`scenario_report.json`** (Stress test results)
  ```json
  {
    "base_case": {
      "default_rate": 0.085,
      "prepayment_rate": 0.031,
      "delinquency_rate": 0.156
    },
    "adverse_credit": {
      "default_rate": 0.142,
      "prepayment_rate": 0.012,
      "delinquency_rate": 0.241
    },
    "high_prepayment": {
      "default_rate": 0.065,
      "prepayment_rate": 0.089,
      "delinquency_rate": 0.098
    }
  }
  ```

- **`scenario_summary.md`** (Markdown summary)

---

### **D. Submission & Logs**

#### Final Submission
- **`outputs/test_submission.csv`** (Predictions on test set)
  ```
  loan_id, predicted_default_probability, predicted_delinquency_probability, final_decision
  2000,    0.23,                          0.45,                             STANDARD_REVIEW
  2001,    0.67,                          0.82,                             APPROVE_WITH_REVIEW
  ```

#### Logs
- **`outputs/llm_logs.jsonl`** (One JSON per line - LLM interactions)
  ```jsonl
  {"loan_id": 1, "risk_score": 0.65, "recommended_action": "escalate_for_review", "timestamp": "2026-08-30 01:36:49"}
  {"loan_id": 2, "risk_score": 0.15, "recommended_action": "monitor", "timestamp": "2026-08-30 01:36:50"}
  ```

- **`outputs/logs/`** (Detailed execution logs from pipeline)

#### Metadata
- **`pipeline_metadata.json`** (Reproducibility info)
  ```json
  {
    "timestamp": "2026-08-30 01:36:49",
    "random_seed": 42,
    "python_version": "3.10",
    "phases_completed": ["schema_discovery", "profiling", "feature_engineering", "training", ...]
  }
  ```

---

## 📊 Quick Data Statistics

### Input Data
| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| train.csv | 66,632 | 33 | Training with targets |
| test.csv | 5,857 | 33 | Scoring (no targets) |
| static.csv | 10,000 | 12 | Origination attributes |

### Output Artifacts
| Category | File Count | Total Size | Purpose |
|----------|-----------|-----------|---------|
| Reports | 8 | ~5 MB | Data quality, profiling, drift |
| Models | 10 | ~200 MB | Trained joblib files |
| Metrics | 4 | ~100 KB | JSON performance reports |
| Analysis | 4 | ~500 KB | Survival, anomaly, scenario, SHAP |
| Submission | 1 | ~50 KB | Predictions for judges |
| Logs | All | ~10 MB | Audit trail, JSONL logs |

---

## 🎯 How to Use

### To Retrain Everything
```bash
python scripts/train.py
```
→ Creates all outputs from scratch

### To Generate Predictions Only (with trained models)
```bash
python scripts/generate_submission.py
```
→ Creates `test_submission.csv`

### To View Dashboard
```bash
streamlit run dashboard/app.py
```
→ Visualize all outputs interactively

### To Validate
```bash
python scripts/validate_project.py
```
→ Verify all outputs exist and are valid

---

## 📋 Summary: Data Flow

```
┌─────────────────────────────────────────────┐
│  INPUTS (CSVs in data/raw/)                │
│  • train.csv (66K rows, 33 cols + targets) │
│  • test.csv (5.8K rows, 33 cols, no targets)│
│  • static.csv (10K rows, 12 cols)          │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  PROCESSING (10 Phases in train_pipeline)  │
│  1. Load & Clean                           │
│  2. Profile & Quality                      │
│  3. Engineer Features                      │
│  4. Time-Aware Split                       │
│  5. Train Models (CatBoost + Baseline)     │
│  6. Survival Analysis                      │
│  7. Anomaly Detection                      │
│  8. Scenario Simulation                    │
│  9. SHAP Explainability                    │
│  10. LLM Copilot & Export                  │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  OUTPUTS (JSON/CSV/HTML in outputs/)       │
│  • Data Intelligence Reports               │
│  • Model Performance Metrics                │
│  • Trained ML Models                       │
│  • Survival & Anomaly Analysis             │
│  • Scenario Projections                    │
│  • Feature Importance (SHAP)               │
│  • LLM Interaction Logs                    │
│  • Final Submission CSV                    │
└─────────────────────────────────────────────┘
```

---

**Created**: 2026-08-30
**Project**: Loan Performance Intelligence Engine (LIQ)
**Version**: 1.0
