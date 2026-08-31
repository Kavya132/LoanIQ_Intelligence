import sys
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.streamlit_ui import render_grounded_assistant

st.title("Model Performance")

metrics_path = Path("outputs/model_metrics.json")
if metrics_path.exists():
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows = []
    for entry in metrics:
        m = entry.get("metrics", {})
        rows.append({
            "target": entry.get("target"),
            "model": entry.get("model_name"),
            "roc_auc": m.get("roc_auc"),
            "f1": m.get("f1"),
            "accuracy": m.get("accuracy"),
            "brier": m.get("brier_score"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")
    render_grounded_assistant({"loan_id": "model_validation", "validation_metrics": rows,
                               "scope": "Historical time-aware validation metrics; not current test-set accuracy."}, "model_performance_explanation", "model_performance")
else:
    st.info("No model metrics found. Run training first.")
