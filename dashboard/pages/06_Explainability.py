import sys
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.streamlit_ui import render_grounded_assistant

st.title("Explainability")

path = Path("outputs/feature_importance.json")
if path.exists():
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload:
        df = pd.DataFrame(payload)
        if {"feature", "importance"}.issubset(df.columns):
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)
            st.subheader("Feature importance ranking")
            st.bar_chart(df.set_index("feature")["importance"])
            st.subheader("Feature importance table")
            st.dataframe(df, width="stretch")
        else:
            st.dataframe(df, width="stretch")
    else:
        st.info("No feature importance data available yet.")
else:
    st.info("No feature importance output available yet.")
