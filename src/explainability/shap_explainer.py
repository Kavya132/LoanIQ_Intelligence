"""Model explainability utilities built around SHAP."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


class SHAPExplainer:
    """Return the most influential features for a tabular model."""

    def __init__(self, sample_size: int = 2000):
        self.sample_size = sample_size

    def explain_model(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Train a compact model and compute SHAP-style importance values."""
        X = X.copy()
        y = y.copy()

        if X.empty or y.empty:
            return pd.DataFrame(columns=["feature", "importance"])

        X = X.fillna(X.median(numeric_only=True))
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame(columns=["feature", "importance"])

        X_numeric = X[numeric_cols]
        if len(X_numeric) > self.sample_size:
            X_numeric, _, y, _ = train_test_split(
                X_numeric,
                y,
                train_size=self.sample_size,
                random_state=42,
                stratify=y if y.nunique() > 1 else None,
            )

        model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
        model.fit(X_numeric, y)

        if shap is not None:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_numeric)
            if isinstance(shap_values, list):
                # For binary classification, shap returns one array per class; keep the positive class.
                values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                values = shap_values
            if isinstance(values, np.ndarray):
                importances = np.abs(values).mean(axis=0)
            else:
                importances = np.abs(np.asarray(values)).mean(axis=0)
        else:
            importances = np.abs(model.feature_importances_)

        importances = np.asarray(importances).ravel()[: len(X_numeric.columns)]

        result = pd.DataFrame({
            "feature": X_numeric.columns,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        return result.reset_index(drop=True)
