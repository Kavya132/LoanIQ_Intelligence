"""# AI Development Log — LoanIQ

## 1. Project Overview

**Project:** LoanIQ – Loan Performance Intelligence Engine  
**Challenge:** Intain Campus FinTech Challenge 2026 – AI Track

LoanIQ is an ML-first loan intelligence platform for loan-data profiling, performance prediction, anomaly detection, scenario simulation, explainability, and grounded LLM-assisted reviewer support.

**Workflow:**  
Loan Data → Data Intelligence → ML Prediction → Anomaly Detection → Scenario Analysis → Explainability → AI Reviewer Copilot → Human Review

The core predictive capabilities are implemented using machine-learning/data-science methods. The LLM is used as an assistance layer for reviewer explanations and development support rather than as the primary loan-risk prediction engine.

---

## 2. AI Tools Used

| AI Tool | Purpose | Usage |
|---|---|---|
| ChatGPT | Architecture, coding assistance, debugging, documentation and prompt refinement | Development |
| Gemini | Grounded reviewer explanations and reviewer assistance | Reviewer Copilot / Application |
| Other AI coding assistance | Where applicable during implementation | Coding support |

AI-generated suggestions were treated as starting points. Important outputs were reviewed, adapted and tested before being incorporated into the final project.

---

## 3. AI-Assisted Development Process

### Step 1 — Understand the problem

AI assistance was used to break the challenge requirements into implementation modules:

- Data profiling
- Feature engineering
- ML prediction
- Time-aware validation
- Anomaly detection
- Scenario analysis
- Explainability
- Reviewer Copilot
- Governance

The final architecture was manually reviewed and adapted to the actual project.

### Step 2 — Generate implementation ideas

AI tools were used to suggest:

- Python implementation approaches
- ML pipeline structures
- Streamlit dashboard components
- Data-processing functions
- Model evaluation approaches
- Explainability structures
- Reviewer Copilot prompts
- Documentation

### Step 3 — Human validation

Important AI-generated suggestions were reviewed against:

1. The Intain problem statement
2. The actual dataset structure
3. The project architecture
4. Model outputs and evaluation metrics
5. Runtime behavior
6. Submission requirements

Only validated and useful outputs were incorporated.

---

## 4. Representative AI Prompt — Architecture

### Prompt

> Design an ML-first architecture for a Loan Performance Intelligence Engine that includes data profiling, feature engineering, time-aware validation, delinquency/default/prepayment prediction, anomaly detection, scenario simulation, explainability and an LLM reviewer copilot. The LLM must not replace the ML prediction layer.

### AI Output / Contribution

The AI suggested separating the solution into:

**Data Layer → ML Layer → Decision/Analysis Layer → AI Assistance Layer**

### Human Review

The architecture was reviewed against the challenge requirements and adapted to the actual LoanIQ implementation.

### Decision

**Accepted after human review.**

---

## 5. Representative AI Prompt — Data Intelligence

### Prompt

> For a loan-level dataset, identify important data-quality checks including missing values, outliers, invalid date relationships, duplicate records, inconsistent fields and distribution issues. Suggest how these checks can be represented in a Streamlit dashboard.

### AI Output / Contribution

Suggested:

- Missing-value analysis
- Data distributions
- Unique-value analysis
- Statistical summaries
- Outlier checks
- Data-quality scoring
- Record-level monitoring

### Human Review

The suggested checks were compared with the available dataset and adapted to the LoanIQ Data Quality page.

### Decision

**Accepted and adapted.**

---

## 6. Representative AI Prompt — Model Evaluation

### Prompt

> Create a model evaluation framework for loan delinquency and default prediction using ROC-AUC, F1, accuracy and Brier score. Include baseline versus improved model comparison.

### AI Output / Contribution

AI suggested a structured model-evaluation table and comparison approach.

### Human Review

The actual metrics were calculated from the implemented ML models and verified separately. The LLM was not used to generate the model performance numbers.

### Decision

**Accepted after verification.**

For example, the LoanIQ dashboard displays a CatBoost three-month delinquency model with an ROC-AUC of approximately **0.7897** and F1 of approximately **0.645**.

---

## 7. Representative AI Prompt — Reviewer Copilot

### Prompt

> Using only the supplied ML evidence for a selected loan, generate a concise reviewer-oriented explanation. Include default, delinquency and prepayment probabilities, important drivers and anomalies if present. Do not invent information. Clearly state that the output is a recommendation and not a lending decision.

### AI Output / Contribution

The LLM generated a structured reviewer-oriented explanation based on the supplied evidence.

### Human Review

The explanation was checked against the available ML evidence before being treated as reviewer guidance.

### Decision

**Accepted when grounded in the supplied evidence; corrected/rejected when unsupported.**

The Reviewer Copilot is intended to assist the reviewer rather than make the final lending decision.

---

## 8. Rejected / Corrected AI Output

The development process included human review for unsupported or overconfident LLM statements.

### Example

**AI-generated statement:**

> The loan is high risk primarily because the borrower has a low credit score.

### Human review

The reviewer checked the available evidence and found that the supplied evidence did not establish that conclusion.

### Action

**Rejected / corrected.**

### Reason

The statement attributed the risk to a specific factor without sufficient supporting evidence.

### Corrected approach

> The model indicates elevated risk based on the available ML evidence and listed feature drivers. The reviewer should validate the underlying loan information before making a decision.

### Lesson

LLM explanations must remain grounded in the evidence provided to the model. A convincing-sounding explanation is not automatically a valid explanation.

---

## 9. Human Review Process

LoanIQ follows a human-in-the-loop workflow.

### 1. AI Inputs

The Reviewer Copilot receives grounded evidence such as:

- ML predictions
- Feature-importance information
- Data-quality evidence
- Validation information
- Loan-level evidence

### 2. AI Recommendation

The LLM generates:

- Reviewer summaries
- Risk explanations
- Natural-language analysis
- Supporting observations

The output is treated as a **recommendation**, not a final decision.

### 3. Human Decision

The reviewer can:

- Accept
- Correct
- Reject

the generated explanation when necessary.

The final decision remains with the human reviewer.

---

## 10. LLM Governance and Logging

LoanIQ includes an **LLM Quality & Governance** area for tracking reviewer-related AI activity.

Relevant information includes:

- Timestamp
- Request ID
- Model
- Purpose
- Success status
- Reviewer status
- Reason

The goal is to make AI assistance traceable and reviewable rather than treating the LLM as an untracked component.

---

## 11. Approximate AI-Generated Code Share

### Estimated AI-assisted code contribution: approximately 40–50%

AI assistance was used for:

- Boilerplate generation
- Streamlit UI components
- Data-processing functions
- Dashboard structures
- Debugging
- Code refactoring
- Documentation
- Prompt development

This percentage is an **approximate development estimate**, not an automatically measured line-of-code percentage.

The final implementation was reviewed, modified, integrated and tested by the developer.

The core ML methodology, data interpretation, model selection, validation decisions, integration and final testing remained human-controlled.

### Statement

> AI assisted a significant portion of the development, but it did not autonomously build or validate the system. AI-generated code was reviewed, modified where required, tested and integrated by the developer.

---

## 12. Where AI Was Not Used

The LLM was not used as the primary prediction engine for:

- Delinquency
- Default
- Prepayment
- Anomaly scores

Those outputs come from the ML/data-science pipeline.

The LLM is used primarily for reviewer assistance, explanation, summarization and natural-language interaction.

This separation maintains the ML-first design of LoanIQ.

---

## 13. Accepted vs Rejected AI Assistance

| Development Area | AI Assistance | Human Action | Result |
|---|---|---|---|
| Architecture | Suggested architecture | Reviewed against challenge | Accepted |
| Data profiling | Suggested checks | Adapted to dataset | Accepted |
| Dashboard UI | Generated component ideas/code | Modified and tested | Accepted |
| ML evaluation | Suggested metrics structure | Verified actual model results | Accepted |
| Reviewer prompts | Generated prompt drafts | Refined for grounding | Accepted |
| Reviewer explanations | Generated summaries | Checked against evidence | Accepted / Corrected |
| Unsupported claims | Could be generated | Manually detected | Rejected |
| Core ML prediction | Not delegated to LLM | Built/validated through ML pipeline | Human-controlled |

---

## 14. Lessons Learned

### 1. AI is most useful when given constraints

Open-ended prompts can produce generic or unsupported answers. Grounding prompts with actual loan evidence produces more useful reviewer assistance.

### 2. LLM output should never be treated as ground truth

An LLM can generate convincing explanations that are not necessarily supported by the model.

### 3. ML and LLM should have separate responsibilities

The ML layer performs prediction.

The LLM provides:

- Explanation
- Summarization
- Reviewer assistance
- Natural-language interaction

### 4. Human review is essential in financial applications

A reviewer should be able to understand the evidence behind a prediction before taking action.

### 5. AI-assisted coding improves development speed

AI was useful for rapid prototyping, debugging, UI generation, refactoring and documentation. Generated code still required testing and manual validation.

### 6. Governance should be built into the application

AI governance should not exist only in documentation. LoanIQ includes an LLM Quality & Governance section to make reviewer-related AI activity more visible and traceable.

---

## 15. Intain Task 8 Evidence Mapping

| Requirement | LoanIQ Evidence |
|---|---|
| AI Development Log | This document |
| AI tools used | ChatGPT / Gemini / AI coding assistance |
| Representative prompts | Sections 4–7 |
| Accepted outputs | Documented throughout the log |
| Rejected/corrected output | Section 8 |
| Human review process | Section 9 |
| Approximate AI-generated code share | Section 11 |
| Lessons learned | Section 14 |
| LLM governance | Section 10 |
| Grounded LLM usage | Reviewer Copilot |
| Human decision control | Section 9 |

---

## 16. Final Statement

> **LoanIQ uses AI as an accelerator and reviewer-assistance layer, not as a replacement for machine learning or human judgment.**
>
> The predictive engine remains ML-first, while the LLM is grounded in model and data evidence to help reviewers understand results. AI-generated code and explanations are subject to human review, validation and correction.
>
> This approach combines the productivity of AI-assisted development with the transparency, reproducibility and governance required for a financial risk-analysis system.
"""
