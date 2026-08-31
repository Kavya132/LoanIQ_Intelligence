"""Custom-loan testing using persisted training artifacts only."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.inference import InferenceError, LoanInferenceService
from src.llm.copilot import GroundedReviewer

st.set_page_config(
    page_title="Custom Loan Testing",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)
st.title("Custom Loan Testing & Risk Assessment")
st.caption("Predictions use saved ML artifacts from training. No formula-based proxy scores are used.")
if "test_history" not in st.session_state:
    st.session_state.test_history = []

@st.cache_resource
def get_service() -> LoanInferenceService:
    return LoanInferenceService(PROJECT_ROOT / "outputs" / "models")

def service_or_message() -> LoanInferenceService | None:
    try:
        return get_service()
    except InferenceError as exc:
        st.error(str(exc))
        st.info("Train first with `py scripts/train.py`; this page will not invent results without artifacts.")
        return None

def score(frame: pd.DataFrame, service: LoanInferenceService) -> pd.DataFrame | None:
    try:
        return service.predict(frame)
    except InferenceError as exc:
        st.error(str(exc))
        return None

def review(record: dict, prediction: dict) -> dict:
    evidence = [{"field": k, "value": v} for k, v in prediction.items()
                if k.endswith("probability") or k in {"next_state", "exception_type", "top_drivers"}]
    reviewer = GroundedReviewer()
    result = reviewer.review_case(record, model_summary=prediction, evidence=evidence)
    reviewer.log_review(result)
    return result

service = service_or_message()
tab1, tab2, tab3, tab4 = st.tabs(["Test Single Loan", "Batch Analysis", "Risk Dashboard", "Test History"])

with tab1:
    st.header("Test Single Loan")
    if service:
        with st.expander("Required raw fields", expanded=False):
            st.code(", ".join(service.required_columns))
    with st.form("single_loan_form"):
        a, b, c = st.columns(3)
        with a:
            loan_id = st.text_input("Loan ID", "5000")
            current_status = st.selectbox("Current status", ["CURRENT", "DELINQUENT", "DEFAULTED", "PREPAID"])
            days_past_due = st.number_input("Days past due", min_value=0, value=0)
            loan_age_months = st.number_input("Loan age (months)", min_value=0, value=12)
            remaining_term_months = st.number_input("Remaining term (months)", min_value=0, value=348)
        with b:
            original_balance = st.number_input("Original balance", min_value=0.0, value=300000.0)
            current_balance = st.number_input("Current balance", min_value=0.0, value=290000.0)
            interest_rate = st.number_input("Interest rate (%)", min_value=0.0, value=3.5)
            credit_score_band = st.text_input("Credit score band", "700-749")
            ltv_band = st.text_input("LTV band", "LTV_60-75")
            dti_band = st.text_input("DTI band", "DTI_25-38")
        with c:
            state = st.text_input("State", "CA")
            loan_purpose = st.text_input("Loan purpose", "PURCHASE")
            property_type = st.text_input("Property type", "SINGLE_FAMILY")
            occupancy_type = st.text_input("Occupancy type", "PRIMARY")
            servicer_name = st.text_input("Servicer", "Servicer_A")
            document_status = st.text_input("Document status", "COMPLETE")
        submitted = st.form_submit_button("Get ML Risk Assessment", type="primary")
    if submitted and service:
        record = {"loan_id": loan_id, "current_status": current_status, "days_past_due": days_past_due,
                  "loan_age_months": loan_age_months, "remaining_term_months": remaining_term_months,
                  "original_balance": original_balance, "current_balance": current_balance, "interest_rate": interest_rate,
                  "credit_score_band": credit_score_band, "ltv_band": ltv_band, "dti_band": dti_band, "state": state,
                  "loan_purpose": loan_purpose, "property_type": property_type, "occupancy_type": occupancy_type,
                  "servicer_name": servicer_name, "document_status": document_status}
        results = score(pd.DataFrame([record]), service)
        if results is not None:
            prediction = results.iloc[0].to_dict()
            reviewer_output = review(record, prediction)
            st.session_state.test_history.append({"loan_record": record, "prediction": prediction, "review": reviewer_output})
            metrics = st.columns(4)
            for col, label, slot in [("delinquency_probability", "3m Delinquency", metrics[0]),
                                     ("default_probability", "12m Default", metrics[1]),
                                     ("prepayment_probability", "12m Prepayment", metrics[2]),
                                     ("exception_probability", "Exception", metrics[3])]:
                slot.metric(label, f"{prediction[col]:.1%}" if col in prediction else "Not trained")
            st.subheader("ML output")
            st.dataframe(results, width="stretch")
            st.subheader("Reviewer recommendation")
            st.success(reviewer_output["recommended_action"].replace("_", " ").title())
            st.write(reviewer_output["summary"])
            st.caption("Recommendation only — final decision remains with human reviewer.")
            st.download_button("Download prediction JSON", json.dumps({"loan_record": record, "prediction": prediction, "review": reviewer_output}, default=str, indent=2), "loan_prediction.json", "application/json")

with tab2:
    st.header("Batch Loan Analysis")
    st.caption("Test files are scored only; accuracy and F1 are not calculated for unlabeled data.")
    source = st.radio("Input type", ["CSV file", "Paste JSON"], horizontal=True)
    frame = None
    if source == "CSV file":
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            frame = pd.read_csv(uploaded)
    else:
        payload = st.text_area("JSON object or array of loan records", height=200)
        if payload:
            try:
                parsed = json.loads(payload)
                frame = pd.DataFrame(parsed if isinstance(parsed, list) else [parsed])
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc.msg}")
    if frame is not None:
        st.write(f"Loaded {len(frame):,} record(s).")
        st.dataframe(frame.head(), width="stretch")
        if st.button("Analyze uploaded loans", type="primary", disabled=service is None):
            results = score(frame, service)
            if results is not None:
                st.dataframe(results, width="stretch")
                st.download_button("Download scored CSV", results.to_csv(index=False), "custom_loan_predictions.csv", "text/csv")

with tab3:
    st.header("Risk Dashboard")
    if st.session_state.test_history:
        history = pd.DataFrame([item["prediction"] for item in st.session_state.test_history])
        st.dataframe(history, width="stretch")
        probability_columns = [c for c in history if c.endswith("probability")]
        if probability_columns:
            fig = go.Figure([go.Box(y=history[column], name=column) for column in probability_columns])
            fig.update_layout(title="Saved ML prediction distribution", yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run a single-loan assessment to populate this session dashboard.")

with tab4:
    st.header("Test History & Audit Log")
    if st.session_state.test_history:
        selected = st.selectbox("Assessment", range(len(st.session_state.test_history)), format_func=lambda i: f"Assessment {i + 1}")
        st.json(st.session_state.test_history[selected])
        if st.button("Clear session history"):
            st.session_state.test_history = []
            st.rerun()
    else:
        st.info("No assessments in this browser session.")
