# Loan Performance Intelligence Engine

An ML-first system for loan-level data profiling, performance prediction, anomaly detection, scenario simulation, and explainable AI-assisted review.

**Challenge**: Intain Campus FinTech Challenge 2026 - AI Track

## Overview

This system builds a complete data science and ML pipeline for:

1. **Data Intelligence** – Profile messy loan data, detect quality issues
2. **Prediction** – Delinquency, default, prepayment, and next-state prediction
3. **Time-to-Event** – Survival/transition modeling with censoring
4. **Anomaly Detection** – Rule-based and ML-based suspicious record identification
5. **Scenario Simulation** – Base, adverse-credit, and high-prepayment stress scenarios
6. **Explainability** – SHAP-based feature importance and local explanations
7. **LLM Copilot** – Grounded, logged, human-controlled reviewer assistance
8. **Governance** – Complete audit trail, leakage prevention, reproducibility

**Important**: The core predictions come from ML/statistical models. The LLM is an assistant for explanations and recommendations, not a prediction engine.

## Project Structure

```
loan-performance-intelligence/
│
├── data/
│   ├── raw/              # Input files from organizer or demo
│   └── processed/        # Cleaned data
│
├── models/               # Trained model artifacts
│
├── outputs/              # Generated reports and predictions
│   └── logs/            # Application logs
│
├── reports/             # Final deliverables
│
├── notebooks/           # Development notebooks (optional)
│
├── src/
│   ├── config/          # Configuration management
│   ├── data/            # Data loading and schema discovery
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and evaluation
│   ├── survival/        # Survival/transition models
│   ├── anomaly/         # Anomaly detection
│   ├── scenarios/       # Scenario simulation
│   ├── explainability/  # SHAP and uncertainty
│   ├── llm/            # LLM provider, RAG, copilot
│   ├── pipeline/        # End-to-end training pipeline
│   └── utils/          # Logging and serialization
│
├── scripts/
│   ├── generate_demo_data.py    # Synthetic data for testing
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Model evaluation
│   ├── run_scenarios.py         # Scenario simulation
│   ├── generate_submission.py   # Submission CSV
│   └── validate_project.py      # Final validation
│
├── dashboard/           # Streamlit application
│
├── tests/              # Pytest tests
│
├── requirements.txt    # Python dependencies
├── config.yaml         # Configuration file
├── .env.example        # Environment template (DO NOT COMMIT .env)
├── .gitignore
├── README.md
├── MODEL_CARD.md
├── AI_DEVELOPMENT_LOG.md
└── Dockerfile
```

## Installation

### Prerequisites
- Python 3.11+
- Git
- pip or conda

### Setup

1. **Clone or create the repository**
```bash
cd loan-performance-intelligence
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings (optional for demo mode)
```

5. **Verify installation**
```bash
python -c "import pandas, numpy, catboost, sklearn; print('All dependencies installed!')"
```

## Quick Start

### Step 1: Generate Demo Data (for testing)
```bash
python scripts/generate_demo_data.py
```

This creates synthetic loan-month panel data in `data/raw/`:
- `loan_monthly_performance_train.csv` (~10k loans × 12 months)
- `loan_monthly_performance_test.csv`
- `loan_static_attributes.csv`

### Step 2: Run Training Pipeline
```bash
python -m src.pipeline.train_pipeline
```

This executes the complete ML pipeline:
- Data discovery and schema validation
- Data profiling and quality scoring
- Validation rule checking
- Feature engineering
- Time-aware train/validation split
- Baseline and improved model training
- Model calibration and evaluation
- Survival/transition modeling
- Anomaly detection
- SHAP explainability
- Scenario simulation
- Report generation
- Submission file creation

### Step 3: Launch Dashboard
```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` with:
- Executive overview and KPIs
- Data quality reports
- Model performance dashboards
- Prediction explorer
- Anomaly viewer
- Scenario simulator
- SHAP explanations
- LLM reviewer copilot
- Challenge compliance checklist

### Step 4: View Outputs

Key outputs are generated in `outputs/`:
- `data_intelligence_report.html` – Profiling and drift analysis
- `data_quality_report.csv` – Record-level quality scores
- `model_metrics.json` – All model evaluation metrics
- `submission.csv` – Final predictions
- `schema_report.json` – Data schema discovery
- `llm_logs.jsonl` – All LLM prompts and responses
- `leakage_report.json` – Leakage detection results

Reports are generated in `reports/`:
- `MODEL_CARD.md` – Model governance and limitations
- `AI_DEVELOPMENT_LOG.md` – AI tool usage documentation

## Data Format

### Input Data (Organizer-Provided)

Expected files in `data/raw/`:

| File | Purpose |
|------|---------|
| `loan_monthly_performance_train.csv` | Panel dataset with one row per loan per month, includes targets |
| `loan_monthly_performance_test.csv` | Unlabeled test data for final scoring |
| `loan_static_attributes.csv` | Origination-level attributes |
| `servicer_updates.csv` | Source conflict detection and reconciliation |
| `data_dictionary.md` | Field definitions for LLM grounding |
| `validation_rules.json` | Deterministic validation checks |
| `macro_scenarios.csv` | Scenario assumptions (base, adverse, high prepayment) |
| `submission_template.csv` | Required output format |

### Schema Flexibility

The system automatically detects available columns and maps them using:
- Exact matching
- Case-insensitive matching
- Field aliases (e.g., `dpd` → `days_past_due`)

If a field is unavailable, the feature is gracefully disabled with clear logging.

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  mode: demo  # 'demo' or 'real'

training:
  random_seed: 42
  validation_months: 6
  time_aware_split: true

models:
  primary: catboost
  catboost:
    iterations: 500
    learning_rate: 0.05
    depth: 7

anomaly:
  contamination: 0.02

llm:
  enabled: true
  provider: openai
  model: gpt-4-turbo
  grounding_enabled: true
```

## Environment Variables

Create `.env` (see `.env.example`):

```bash
# LLM Configuration
LLM_API_KEY=your_key_here
LLM_MODEL=gpt-4-turbo
LLM_BASE_URL=https://api.openai.com/v1
LLM_PROVIDER=openai

# Data Mode
DATA_MODE=demo  # or 'real' when organizer data available

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

**Never commit `.env` with actual secrets.**

## Data Modes

### Demo Mode (Default)
- Uses synthetic data from `scripts/generate_demo_data.py`
- Fast end-to-end testing
- No organizer data required
- Clearly labeled as "DEMO" throughout

### Real Mode
- Uses actual organizer data from `data/raw/`
- Real challenge results
- Automatically activated when organizer files are detected
- Clear "ORGANIZER DATA" labels in outputs

## Key Features

### 1. Data Intelligence
- Column-by-column distribution analysis
- Missing value patterns and heatmaps
- Outlier detection (statistical + ML)
- Cross-column relationship validation
- Train/test data drift detection (PSI, KS test)
- Record-level and batch-level quality scores

### 2. Predictive Modeling
- **Baseline models** – Majority class, logistic regression
- **Improved models** – CatBoost (primary), XGBoost (fallback)
- **Targets predicted:**
  - 3-month and 6-month delinquency
  - 12-month default
  - 12-month prepayment
  - Next loan state
  - Exception need and type

### 3. Time-Aware Validation
- Time-based train/validation split (not random row splitting)
- Prevents same-loan and future leakage
- Explicit leakage detection and reporting
- Time split validation report

### 4. Class Imbalance Handling
- Class weights in model training
- Careful sampling where safe
- Threshold optimization for precision/recall trade-offs
- Imbalance-aware metrics

### 5. Model Calibration
- Calibrated vs uncalibrated probability comparison
- Brier score and calibration curves
- Optional calibration by vintage or credit band

### 6. Survival/Transition Modeling
- Cox Proportional Hazards (lifelines)
- Monthly state transition matrix
- Survival curves and cumulative event probability
- Censoring treatment documentation

### 7. Anomaly Detection
- **Rule-based layer** – Validation rules and logical checks
- **ML layer** – Isolation Forest with configurable contamination
- **Output** – Anomaly score (0-1), explanations, severity flags
- **20+ reviewer examples** with evidence trails

### 8. Explainability
- **Global** – Feature importance ranking, SHAP summary plots
- **Local** – Per-record drivers, SHAP contributions
- **Uncertainty** – Calibrated probabilities, confidence intervals
- **Error analysis** – False positive/negative distributions and drivers

### 9. Scenario Simulation
- Base, adverse-credit, and high-prepayment scenarios
- Portfolio and segment-level projections
- Delinquency, default, prepayment rate changes
- Scenario driver explanation

### 10. LLM-Assisted Reviewer
- **Grounded** – Evidence from data, models, and rules
- **Logged** – Complete prompt/response audit trail
- **Governed** – Human review required, never autonomous
- **Fallback** – Deterministic if API unavailable
- **Examples** – Includes controlled hallucination tests

### 11. Model Governance
- Model card with objective, data, features, metrics
- Leakage prevention checks
- Dataset versioning
- Training reproducibility
- Known limitations documented

## Testing

Run pytest suite:

```bash
pytest -q
pytest -v                    # Verbose
pytest --cov=src           # With coverage
pytest tests/test_data.py   # Specific test file
```

Key test areas:
- Data loader and schema discovery
- Feature engineering
- Leakage detection
- Model training and evaluation
- SHAP output formatting
- LLM grounding and validation
- Submission file validation

## LLM Configuration

### OpenAI (Default)
```bash
export LLM_API_KEY=sk-...
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4-turbo
```

### Local Mode (No API)
```bash
export LLM_ENABLED=false
```
The system uses a deterministic fallback reviewer.

### Custom Provider
Edit `src/llm/provider.py` to add support for other APIs (Claude, Anthropic, LLaMA, etc.)

## Model Development

### Training Pipeline Phases

1. **Phase 1** – Project skeleton, config, logging ✓
2. **Phase 2** – Data profiling, validation, quality scoring
3. **Phase 3** – Feature engineering, target construction, leakage check
4. **Phase 4** – Baseline and improved models, calibration
5. **Phase 5** – Survival and transition models
6. **Phase 6** – Anomaly detection, exception prediction
7. **Phase 7** – SHAP explainability, uncertainty, error analysis
8. **Phase 8** – Scenario simulation, Monte Carlo
9. **Phase 9** – RAG, LLM copilot, grounding, logging
10. **Phase 10** – Reports, submission, model card
11. **Phase 11** – Streamlit dashboard
12. **Phase 12** – FastAPI service
13. **Phase 13** – Tests, validation, demo

### Adding Custom Features

Edit `src/features/feature_pipeline.py`:

```python
class FeaturePipeline:
    def engineer_features(self, df):
        # Raw features
        # Temporal features
        # Balance features
        # Delinquency trends
        # Custom: add here
        return df
```

### Adding Custom Validation Rules

Edit `src/data/validation.py` or load from `validation_rules.json`.

## Reproducibility

All experiments are reproducible with:

```bash
python -m src.pipeline.train_pipeline
```

Reproducibility metadata saved to `outputs/run_metadata.json`:
- Timestamp
- Dataset hash (MD5)
- Configuration
- Model parameters
- Features list
- Training/validation periods
- Random seed
- Metrics

## Performance

Optimized for datasets with 250K-1M rows:

- Vectorized pandas operations
- Categorical dtypes for memory efficiency
- Streamlit caching for fast dashboard
- Model artifact caching
- Configurable SHAP sample size
- Visualization sampling (doesn't affect training)

## Docker

Build and run in container:

```bash
docker build -t loan-intelligence .
docker run -p 8501:8501 -v $(pwd)/data:/app/data loan-intelligence
```

## Troubleshooting

### Missing Data Files
```
ERROR: Training dataset not found. Put loan_monthly_performance_train.csv inside data/raw/
```
→ Generate demo data: `python scripts/generate_demo_data.py`

### LLM Unavailable
```
WARNING: LLM API key not configured. Using deterministic fallback reviewer.
```
→ Set `LLM_API_KEY` in `.env` or use demo mode

### Memory Issues
→ Reduce sample sizes in `config.yaml`:
```yaml
explainability:
  shap_sample_size: 1000  # Instead of 2000
```

### Encoding Errors
The loader auto-detects encoding. If issues persist, specify in config:
```python
df = DataLoader.load_csv(filepath, encoding="utf-16")
```

## Documentation

- **README.md** – This file
- **MODEL_CARD.md** – Model governance and metrics
- **AI_DEVELOPMENT_LOG.md** – AI tools and prompts used
- **outputs/schema_report.html** – Data schema discovery
- **outputs/data_intelligence_report.html** – Profiling and drift
- **outputs/model_metrics.json** – Detailed metrics
- **outputs/leakage_report.json** – Leakage detection results

## Submission

Generate final submission with:

```bash
py train.py
py predict.py --input path/to/new_loan_data.csv --output outputs/submission.csv
# Or use the default challenge test file after training:
py scripts/generate_submission.py
```

Creates `outputs/submission.csv` with required fields:
- `loan_id`
- `delinquency_probability`
- `default_probability`
- `prepayment_probability`
- `next_state`
- `exception_required`
- `exception_type`
- `anomaly_score`
- `top_drivers`
- `action`
- `confidence`

Validated against `submission_validation_report.json`.

## Challenge Compliance

✓ **Task 1** – Data Intelligence and Profiling  
✓ **Task 2** – Loan Performance Prediction  
✓ **Task 3** – Time-to-Event / Survival Modeling  
✓ **Task 4** – Anomaly and Exception Detection  
✓ **Task 5** – Scenario and Stress Simulation  
✓ **Task 6** – Explainability Layer  
✓ **Task 7** – LLM-Assisted Reviewer Copilot  
✓ **Task 8** – Agentic ML Development Evidence  

See `AI_DEVELOPMENT_LOG.md` for agentic coding evidence and prompt examples.

## License

Intain Campus FinTech Challenge 2026

## Support

For issues or questions, review:
1. `outputs/logs/` for detailed error messages
2. This README
3. Individual module docstrings
4. Test files in `tests/`
