"""Scenario and output report generation for challenge submissions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import json

import numpy as np
import pandas as pd


def build_scenario_report(scenario_results: Dict[str, Any], output_path: str | Path) -> Path:
    """Persist scenario results in JSON and markdown summary."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "scenario_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(scenario_results, f, indent=2, default=str)

    lines = ["# Scenario Summary", ""]
    for name, payload in scenario_results.items():
        default_rate = payload.get("default_rate", 0)
        loss_rate = payload.get("loss_rate", 0)
        lines.append(f"## {name}")
        lines.append(f"- Default rate: {default_rate:.4f}")
        lines.append(f"- Loss rate: {loss_rate:.4f}")
        lines.append("")

    md_path = output_path / "scenario_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


def create_submission_file(predictions: pd.DataFrame, output_path: str | Path, required_cols: Iterable[str] = ("loan_id", "predicted_default_probability", "predicted_delinquency_probability", "final_decision")) -> Path:
    """Create a challenge-compliant submission CSV from model predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = predictions.copy()
    for col in required_cols:
        if col not in df.columns:
            if col == "loan_id":
                df[col] = df.index
            elif col == "predicted_default_probability":
                df[col] = df.get("default_probability", 0.0).astype(float)
            elif col == "predicted_delinquency_probability":
                df[col] = df.get("delinquency_probability", 0.0).astype(float)
            elif col == "final_decision":
                df[col] = np.where(df["predicted_default_probability"] >= 0.5, "APPROVE_WITH_REVIEW", "STANDARD_REVIEW")

    submission = df[list(required_cols)].copy()
    submission.to_csv(output_path, index=False)
    return output_path


def build_model_summary(model_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a list of model metrics into a compact summary."""
    return {
        "model_count": len(model_metrics),
        "results": model_metrics,
        "best_auc": max((m.get("roc_auc", 0.0) for m in model_metrics), default=0.0),
    }
