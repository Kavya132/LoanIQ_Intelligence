"""Gemini explanation layer for evidence produced by the ML engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.llm.gemini_client import DISCLAIMER, GeminiReviewerClient, GeminiUnavailable

st.title("Reviewer Copilot")
st.caption("ML engine → structured evidence → human reviewer. The review layer does not make lending decisions.")

@st.cache_data
def load_scored_loans() -> pd.DataFrame:
    for path in (ROOT / "outputs/submission.csv", ROOT / "outputs/test_submission.csv"):
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_scenarios() -> dict:
    path = ROOT / "outputs/scenario_report.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def value_or_none(value):
    return None if pd.isna(value) else value

scored = load_scored_loans()
if scored.empty or "loan_id" not in scored.columns:
    st.info("No current ML prediction output found. Run training and inference before requesting reviewer assistance.")
    st.stop()

selected_loan = st.selectbox("Loan", scored["loan_id"].astype(str).tolist())
row = scored.loc[scored["loan_id"].astype(str) == selected_loan].iloc[0]
evidence = {key: value_or_none(value) for key, value in row.to_dict().items()}
evidence["scenario_results"] = load_scenarios()
evidence["data_quality_issues"] = []

st.subheader("ML evidence")
metrics = st.columns(3)
for column, label, slot in [("default_probability", "Default probability", metrics[0]),
                            ("delinquency_probability", "Delinquency probability", metrics[1]),
                            ("prepayment_probability", "Prepayment probability", metrics[2])]:
    value = evidence.get(column)
    if isinstance(value, (int, float)) and "probability" in column:
        display_value = f"{float(value):.1%}"
    elif value is not None:
        display_value = str(value)
    else:
        display_value = "Not available"
    slot.metric(label, display_value)

evidence_df = pd.DataFrame([
    {"Field": key, "Value": value}
    for key, value in evidence.items()
    if value is not None
])
st.subheader("Evidence table")
st.dataframe(evidence_df, width="stretch", hide_index=True)

client = GeminiReviewerClient()
if not client.configured:
    st.warning("Reviewer Copilot unavailable. ML analysis remains available. Configure GEMINI_API_KEY to enable reviewer assistance.")

purpose = st.selectbox("Analysis type", ["loan_risk_summary", "risk_explanation", "anomaly_explanation", "scenario_explanation", "reviewer_note"])
question = st.text_input("Ask a grounded question (optional)", placeholder="Why is this loan high risk?")

if st.button("Generate reviewer analysis", type="primary", disabled=not client.configured):
    try:
        response = client.review(evidence, purpose=purpose, question=question or None)
        st.session_state["gemini_response"] = response
    except GeminiUnavailable as exc:
        st.error(str(exc))

response = st.session_state.get("gemini_response")
if response:
    st.subheader("Reviewer analysis")
    st.write(response["summary"])
    st.write("Recommended reviewer action:", response["recommended_reviewer_action"])

    response_items = []
    for key in ("key_evidence", "risk_drivers", "data_quality_concerns", "confidence_statement", "limitations"):
        value = response.get(key, []) if isinstance(response.get(key), list) else response.get(key)
        if isinstance(value, list):
            for item in value:
                response_items.append({"Section": key, "Detail": item})
        elif value is not None:
            response_items.append({"Section": key, "Detail": value})

    if response_items:
        st.dataframe(pd.DataFrame(response_items), width="stretch", hide_index=True)

    st.caption(DISCLAIMER)
    st.subheader("Human reviewer action")
    decision = st.radio("Decision", ["accepted", "rejected", "corrected"], horizontal=True)
    reason = st.text_area("Reason or correction", placeholder="Required for a rejected or corrected recommendation.")
    if st.button("Record reviewer decision"):
        if decision != "accepted" and not reason.strip():
            st.error("Provide a reason for rejection or correction.")
        else:
            client.record_feedback(response["request_id"], decision, reason.strip())
            st.success("Reviewer decision recorded in outputs/llm_logs.jsonl.")

st.divider()
st.subheader("LLM Quality & Governance")
log_path = ROOT / "outputs/llm_logs.jsonl"
if log_path.exists():
    entries = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    feedback = [entry for entry in entries if entry.get("reviewer_status") in {"accepted", "rejected", "corrected"}]
    if feedback:
        st.dataframe(pd.DataFrame(feedback).tail(10), width="stretch")
    else:
        st.info("No real reviewer feedback exists yet. Use the controls above to create an auditable accepted, rejected, or corrected review; no historical interaction is fabricated.")
else:
    st.info("No reviewer log exists yet.")
