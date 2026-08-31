import sys
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.streamlit_ui import render_grounded_assistant

st.title("Anomaly Monitoring")

path = Path("outputs/anomaly_summary.json")
if path.exists():
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and payload:
        summary_df = pd.DataFrame([
            {
                "Metric": "Total rows",
                "Value": str(payload.get("total_rows", 0)),
            },
            {
                "Metric": "Anomaly count",
                "Value": str(payload.get("anomaly_count", 0)),
            },
            {
                "Metric": "Anomaly rate",
                "Value": f"{float(payload.get('anomaly_rate', 0)):.2%}",
            },
            {
                "Metric": "Avg anomaly score",
                "Value": f"{float(payload.get('avg_anomaly_score', 0)):.2f}",
            },
        ])

        st.subheader("Summary metrics")
        st.dataframe(summary_df, width="stretch", hide_index=True)

        st.subheader("Key anomaly indicators")
        chart_df = pd.DataFrame({
            "Indicator": ["Anomaly count", "Anomaly rate", "Avg anomaly score"],
            "Value": [
                float(payload.get("anomaly_count", 0)),
                float(payload.get("anomaly_rate", 0)),
                float(payload.get("avg_anomaly_score", 0)),
            ],
        })
        st.bar_chart(chart_df.set_index("Indicator")["Value"])
    else:
        st.info("No anomaly summary values are available yet.")
else:
    st.info("No anomaly summary is available yet.")
