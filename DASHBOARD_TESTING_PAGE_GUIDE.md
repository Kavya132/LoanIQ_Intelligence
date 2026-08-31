# 🧪 Custom Loan Testing Dashboard Page - User Guide

**New Dashboard Page**: `08_Custom_Loan_Testing.py`

---

## 🎯 Overview

A brand-new **interactive Streamlit page** in the dashboard that allows you to:

1. ✅ **Test single loans** with custom inputs
2. ✅ **Get instant risk predictions** (risk score, probabilities)
3. ✅ **View recommendations** (monitor, escalate, review)
4. ✅ **Analyze multiple loans** (batch processing)
5. ✅ **Visualize trends** (risk distribution, probability analysis)
6. ✅ **Export results** (JSON, CSV, download)
7. ✅ **Track test history** (see all previous tests)

---

## 📍 How to Access

### Step 1: Start the Dashboard
```bash
cd d:\LIQ\LIQ
streamlit run dashboard/app.py
```

### Step 2: Navigate to New Page
- Open http://localhost:8501
- Look in the **sidebar** for page list
- Click **"08_Custom_Loan_Testing"** ← **NEW PAGE!**

---

## 🎛️ Page Features

### **Tab 1: 📝 Test Single Loan** (Main Testing Interface)

#### Input Sections:

**Loan Identity & Status**
- Loan ID (number)
- Current Status (dropdown: CURRENT, DELINQUENT, DEFAULTED, PREPAID)
- Days Past Due (slider: 0-180)

**Financial Profile**
- Original Balance ($)
- Current Balance ($)
- Interest Rate (%)

**Credit & Risk Profile**
- Credit Score Band (750+, 700-749, 650-699, <650)
- LTV Band (Loan-to-Value ratio)
- DTI Band (Debt-to-Income ratio)

**Loan & Property Details**
- Loan Purpose (PURCHASE or REFINANCE)
- Property Type (SINGLE_FAMILY, CONDO, TOWNHOUSE, MULTI_FAMILY)
- Occupancy Type (PRIMARY, SECOND_HOME, INVESTOR)
- State (CA, TX, NY, OH, FL, IL, PA, NC, MI, GA)

**Performance Flags**
- Modification Flag (checkbox)
- Prepayment Flag (checkbox)
- Default Flag (checkbox)

**Document & Service**
- Servicer (Servicer_A, B, C, D)
- Document Status (COMPLETE, INCOMPLETE, MISSING)

#### Output:

When you click **"🔮 Get Risk Assessment"** button:

```
📊 RISK ASSESSMENT RESULTS
├─ Risk Score: 0.65 (0.0 - 1.0)
├─ Delinquency Probability: 25%
├─ Default Probability: 16%
├─ Prepayment Probability: 10%
├─ Risk Level Gauge (Visual)
├─ Recommended Action: ESCALATE_FOR_REVIEW 🟡
├─ Summary: "Loan shows 0.65 risk based on..."
├─ Evidence Table: [supporting data]
└─ Export Options:
   ├─ Save to JSON
   └─ Copy to Clipboard
```

**Visual Elements**:
- 4 metric cards displaying risk score, probabilities
- Interactive risk gauge (needle meter)
- Color-coded risk level (🟢 LOW / 🟡 MEDIUM / 🔴 HIGH)
- Evidence table with drivers

---

### **Tab 2: 📊 Batch Analysis**

Test **multiple loans at once**

**Three input options**:

1. **CSV File Upload**
   - Upload a CSV with multiple loans
   - System reads all records
   - Click "Analyze All Loans"
   - Gets results as table + download

2. **Paste JSON**
   - Paste JSON array of loan objects
   - Batch process all records
   - View results in table

3. **Sample Data**
   - Uses previous test history
   - Analyzes all loans you've tested
   - Shows summary stats

**Batch Output**:
```
Total Loans: 50
Avg Risk: 0.42
High Risk Loans: 15
[Download Results CSV]
```

---

### **Tab 3: 📈 Risk Analysis Dashboard**

**Summary Statistics**:
- Total Tests Run
- Average Risk Score
- Max Risk Score
- High Risk Loan Count

**Visualizations**:
- **Risk Score Distribution** (histogram)
  - Shows spread of risk scores
  - Identifies clusters
  
- **Probability Distributions** (box plots)
  - Delinquency probabilities
  - Default probabilities
  - Prepayment probabilities

**Test History Table**:
- Loan ID
- Current Status
- Days Past Due
- Risk Score
- Recommended Action
- All Probabilities

---

### **Tab 4: 📋 Test History & Logs**

**Complete Test Record**:
- Select from dropdown (all previous tests)
- View full loan record (JSON)
- View full risk assessment (JSON)
- See all probabilities

**Actions**:
- 🗑️ Clear All History (reset)

---

## 💡 Usage Examples

### Example 1: Test a Good Loan
```
Input:
- loan_id: 1001
- current_status: CURRENT
- days_past_due: 0
- original_balance: $300,000
- current_balance: $290,000
- credit_score_band: 750+
- days_past_due: 0

Output:
✅ Risk Score: 0.10 (LOW)
✅ Action: MONITOR
✅ Delinquency Prob: 0%
✅ Default Prob: 3%
✅ Prepayment Prob: 10%
```

### Example 2: Test a Risky Loan
```
Input:
- loan_id: 1002
- current_status: DELINQUENT
- days_past_due: 120
- original_balance: $350,000
- current_balance: $250,000 (27% down)
- credit_score_band: <650
- modification_flag: checked

Output:
🔴 Risk Score: 0.78 (HIGH)
🔴 Action: CREDIT_REVIEW_AND_COLLECTION
🔴 Delinquency Prob: 33%
🔴 Default Prob: 22%
🔴 Prepayment Prob: 10%
```

### Example 3: Test Multiple Loans
1. Click **Tab 2: Batch Analysis**
2. Select **CSV File**
3. Upload your `loans.csv` (50 loans)
4. Click **"Analyze All Loans"**
5. View results table
6. Click **"Download Results CSV"**
7. Get predictions for all 50 loans

---

## 📤 Export Options

### From Single Loan Test:
- **Save to JSON**: Creates `test_loan_XXXX.json` in `outputs/`
- **Copy to Clipboard**: Copy JSON to paste elsewhere

### From Batch Analysis:
- **Download CSV**: Download results as `batch_results_TIMESTAMP.csv`

### From Risk Dashboard:
- Can screenshot visualizations
- Or extract data from test history table

---

## 🔄 Workflow

### Typical User Journey:

```
1. Open Dashboard
   ↓
2. Click "08_Custom_Loan_Testing"
   ↓
3. Fill in loan details (Tab 1)
   ↓
4. Click "🔮 Get Risk Assessment"
   ↓
5. View Results
   - Risk score
   - Probabilities
   - Recommendation
   - Evidence
   ↓
6. Export (optional)
   - Save JSON or Copy
   ↓
7. View Analytics (Tab 3)
   - See risk distribution
   - View probability trends
   ↓
8. Test another loan or batch (repeat from Step 3)
```

---

## 🎨 Visual Design

**Layout**:
- 📱 Responsive (mobile-friendly)
- 🎯 Two-column forms for input
- 🎨 Color-coded risk levels
- 📊 Interactive Plotly charts
- 📈 Organized tabbed interface

**Colors**:
- 🟢 GREEN (Low risk: 0.0-0.35)
- 🟡 YELLOW (Medium risk: 0.35-0.65)
- 🔴 RED (High risk: 0.65-1.0)

---

## ⚙️ Technical Details

**Architecture**:
```python
# File: dashboard/pages/08_Custom_Loan_Testing.py
# 
# Uses:
# ├─ Streamlit widgets (input, buttons, tabs)
# ├─ GroundedReviewer (risk assessment)
# ├─ Plotly (visualizations)
# ├─ Pandas (data handling)
# ├─ Session State (history tracking)
# └─ JSON/CSV (export)
```

**Risk Calculation**:
- Delinquency Prob: `min(days_past_due / 365, 1.0)`
- Default Prob: `max(0, min(1, (1 - balance_ratio) * 0.8))`
- Prepayment Prob: `0.7 if prepay_flag & dpd==0 else 0.1`

**Session State**:
- Tracks all tests in `st.session_state.test_history`
- Persists during session
- Can be cleared with button

---

## 🚀 Quick Start

### 30-Second Test:

1. Start dashboard: `streamlit run dashboard/app.py`
2. Go to **"08_Custom_Loan_Testing"** page
3. Fill in quick defaults (all fields pre-filled with example values)
4. Click **"🔮 Get Risk Assessment"**
5. See instant results with:
   - ✅ Risk score
   - ✅ Probabilities
   - ✅ Recommendation
   - ✅ Evidence

---

## 📱 What's Better Than API?

| Feature | API | This Dashboard |
|---------|-----|-----------------|
| Visual feedback | ❌ | ✅ |
| Risk gauge | ❌ | ✅ |
| Test history | ❌ | ✅ |
| Batch analysis | ⚠️ Manual | ✅ Auto |
| Export options | ❌ | ✅ JSON + CSV |
| Analytics dashboard | ❌ | ✅ Charts + Stats |
| No coding needed | ❌ | ✅ |
| Mobile friendly | ❌ | ✅ |

---

## 🎯 Perfect For:

- ✅ **Loan officers** reviewing individual files
- ✅ **Risk managers** analyzing portfolios
- ✅ **Data scientists** testing new features
- ✅ **QA teams** validating predictions
- ✅ **Non-technical stakeholders** exploring results
- ✅ **Demo purposes** showing model capabilities

---

## 📞 Support Notes

- All test data stays in session
- Clear history with button if needed
- Refresh page to reset
- Export before closing browser to save results
- CSV/JSON files also auto-saved to `outputs/` folder

---

**Created**: 2026-08-30  
**Project**: LIQ - Loan Performance Intelligence Engine  
**Page**: `dashboard/pages/08_Custom_Loan_Testing.py`
