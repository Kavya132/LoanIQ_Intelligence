"""Reusable, evidence-only Gemini controls for dashboard analysis pages."""
from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from src.llm.gemini_client import DISCLAIMER, GeminiReviewerClient, GeminiUnavailable


def render_grounded_assistant(evidence: Mapping[str, Any], purpose: str, key: str, title: str = "Reviewer Assistance") -> None:
    """Render an optional Gemini explanation panel for already-calculated evidence."""
    st.subheader(title)
    st.caption("Explains the displayed ML/rule evidence. It does not calculate predictions or make decisions.")
    client = GeminiReviewerClient()
    question = st.text_input("Ask about this analysis (optional)", key=f"{key}_question")
    if not client.configured:
        st.info("Reviewer Copilot unavailable. ML analysis remains available. Configure GEMINI_API_KEY to enable it.")
        return
    if st.button("Generate grounded explanation", key=f"{key}_generate"):
        try:
            st.session_state[f"{key}_response"] = client.review(evidence, purpose=purpose, question=question or None)
        except GeminiUnavailable as exc:
            st.error(str(exc))
    response = st.session_state.get(f"{key}_response")
    if not response:
        return
    st.write(response["summary"])
    st.write("Recommended reviewer action:", response["recommended_reviewer_action"])
    with st.expander("Grounded details"):
        st.json({name: response[name] for name in ("key_evidence", "risk_drivers", "data_quality_concerns", "confidence_statement", "limitations")})
    st.caption(DISCLAIMER)
    decision = st.radio("Human reviewer decision", ["accepted", "rejected", "corrected"], horizontal=True, key=f"{key}_decision")
    reason = st.text_area("Reason or correction", key=f"{key}_reason")
    if st.button("Record reviewer decision", key=f"{key}_feedback"):
        if decision != "accepted" and not reason.strip():
            st.error("Provide a reason for rejection or correction.")
        else:
            client.record_feedback(response["request_id"], decision, reason.strip())
            st.success("Reviewer decision logged.")
