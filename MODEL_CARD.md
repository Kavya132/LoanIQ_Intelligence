# Model Card: Loan Performance Intelligence Engine

## Model summary
The system is an ML-first loan performance intelligence engine designed to profile loan portfolios, estimate default and delinquency risk, identify anomalies, and support human review decisions.

## Intended use
This model is intended for internal portfolio monitoring and loan review support in a supervised lending or risk management workflow. It is not a fully autonomous credit decision system.

## Training data
The project supports synthetic demo data and organizer-style loan-month panel data. The model uses loan-level and periodic performance fields such as balance, delinquency, score band, seasonality, and portfolio behavior features.

## Model family
Primary modeling uses CatBoost gradient boosting with time-aware validation. Baseline comparison and fallback modeling may include XGBoost or logistic-regression-style baselines.

## Performance considerations
Performance depends on data quality, label availability, and the temporal structure of the dataset. Time-aware validation is used to reduce leakage and over-optimistic evaluation.

## Risk considerations
- Data drift may affect production performance.
- Missing or noisy portfolio data can drive unstable probabilities.
- The output is assistive and should be reviewed by humans.

## Governance
- Leakage checks are enabled.
- Validation reports are saved in the outputs folder.
- Explainability and grounding are included for reviewer support.
- Any final decision remains under human review.
