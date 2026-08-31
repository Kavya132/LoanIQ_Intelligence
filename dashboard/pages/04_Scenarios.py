import sys
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.streamlit_ui import render_grounded_assistant

st.title("Scenario Analysis")

scenario_path = Path("outputs/scenario_report.json")
if scenario_path.exists():
    with scenario_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    st.dataframe(pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={"index": "scenario"}), width="stretch")
    render_grounded_assistant({"loan_id": "current_portfolio", "scenario_results": data}, "scenario_explanation", "scenarios")
else:
    st.info("No scenario report found. Run the scenario generator or pipeline first.")
