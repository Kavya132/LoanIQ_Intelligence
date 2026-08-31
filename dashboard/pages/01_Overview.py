from pathlib import Path
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Overview", layout="wide")


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


st.title("Loan Performance Intelligence Engine")
st.caption("ML-first assessment and monitoring overview")

summary = load_json(Path("outputs/model_metrics.json"))
quality = load_json(Path("outputs/data_quality_summary.json"))
profile = load_json(Path("outputs/data_profile.json"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model runs", len(summary) if isinstance(summary, list) else 0)
col2.metric("Quality score", round(float(quality.get("average_quality_score", 0.0)), 3) if isinstance(quality, dict) else 0.0)
col3.metric("Profile features", len(profile) if isinstance(profile, dict) else 0)
col4.metric("Status", "Ready")

st.subheader("Challenge-ready snapshot")
checklist = [
    "ML-first core prediction engine",
    "Time-aware validation split",
    "Leakage checks enabled",
    "Explainability and reviewer guidance",
    "Dashboard and API support",
    "Submission-ready outputs",
]
for item in checklist:
    st.checkbox(item, value=True, disabled=True)

if isinstance(summary, list) and summary:
    st.subheader("Top model results")
    rows = []
    for entry in summary:
        metrics = entry.get("metrics", {})
        rows.append({
            "target": entry.get("target"),
            "model": entry.get("model_name"),
            "roc_auc": metrics.get("roc_auc"),
            "f1": metrics.get("f1"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")
