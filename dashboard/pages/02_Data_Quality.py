import sys
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.streamlit_ui import render_grounded_assistant

st.title("Data Quality")

quality_path = Path("outputs/data_quality_summary.json")
profile_path = Path("outputs/data_profile.json")

if quality_path.exists():
    with quality_path.open("r", encoding="utf-8") as f:
        dq = json.load(f)
    st.dataframe(pd.DataFrame([dq]), width="stretch")
else:
    st.info("No quality summary found. Run the pipeline first.")

if profile_path.exists():
    with profile_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)
    st.subheader("Profile snapshot")
    st.dataframe(pd.DataFrame.from_dict(profile, orient="index").reset_index().rename(columns={"index": "feature"}), width="stretch")

if quality_path.exists():
    render_grounded_assistant({"loan_id": "current_dataset", "data_quality_summary": dq, "data_profile": profile if profile_path.exists() else {}}, "data_quality_explanation", "data_quality")
