"""Anomaly detection utilities for loan performance monitoring."""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestAnomalyDetector:
    """Classify rare behavioral patterns in loan-level data."""

    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train the detector and return a dataframe with anomaly flags."""
        X = df.copy()
        if X.empty:
            return pd.DataFrame({"anomaly_score": [], "is_anomaly": []})

        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("IsolationForest requires at least one numeric feature column.")

        X_numeric = X[numeric_cols].fillna(X[numeric_cols].median(numeric_only=True))
        scores = self.model.fit_predict(X_numeric)
        anomaly_scores = -self.model.score_samples(X_numeric)

        result = X.copy()
        result["anomaly_score"] = anomaly_scores
        result["is_anomaly"] = (scores == -1).astype(int)
        return result

    def summarize(self, result: pd.DataFrame) -> Dict[str, Any]:
        """Return basic anomaly summary metrics."""
        if result.empty:
            return {"total_rows": 0, "anomaly_count": 0, "anomaly_rate": 0.0}

        anomaly_count = int(result["is_anomaly"].sum())
        return {
            "total_rows": int(len(result)),
            "anomaly_count": anomaly_count,
            "anomaly_rate": float(anomaly_count / len(result)) if len(result) else 0.0,
            "avg_anomaly_score": float(result["anomaly_score"].mean()),
        }
