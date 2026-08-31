"""Shared inference path used by the CLI and Streamlit custom-loan page.

This module intentionally never calculates a proxy probability.  Predictions are
only returned when a persisted model artifact can score the supplied records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import numpy as np
import pandas as pd

from src.features.feature_pipeline import FeaturePipeline, PersistedFeaturePipeline


class InferenceError(ValueError):
    """Raised when input data cannot safely be scored by the saved artifacts."""


TARGET_OUTPUTS = {
    "next_3m_delinquency_flag": "delinquency_probability",
    "next_6m_delinquency_flag": "delinquency_6m_probability",
    "next_12m_default_flag": "default_probability",
    "next_12m_prepayment_flag": "prepayment_probability",
    "exception_required": "exception_probability",
}


@dataclass
class LoadedModel:
    target: str
    model: Any


class LoanInferenceService:
    """Load saved models and apply the training feature construction at inference."""

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower().replace("_", " ").replace("-", " ")
            if normalized in {"not available", "n/a", "na", "null", "none"}:
                return None
        return value

    def __init__(self, models_dir: str | Path = "outputs/models"):
        self.models_dir = Path(models_dir)
        pipeline_path = self.models_dir / "preprocessing_pipeline.joblib"
        self.preprocessing = joblib.load(pipeline_path) if pipeline_path.exists() else None
        self.models = self._load_models()
        if not self.models:
            raise InferenceError(
                f"No trained model artifacts found in {self.models_dir}. Run `py scripts/train.py` first."
            )

    def _load_models(self) -> Dict[str, LoadedModel]:
        models: Dict[str, LoadedModel] = {}
        for target in [*TARGET_OUTPUTS, "next_state", "exception_type"]:
            # Prefer the improved model. Baseline is used only when it is the only artifact.
            candidates = [
                self.models_dir / f"{target}_catboost.joblib",
                self.models_dir / f"{target}_baseline.joblib",
            ]
            for path in candidates:
                if path.exists():
                    models[target] = LoadedModel(target, joblib.load(path))
                    break
        return models

    @property
    def required_columns(self) -> list[str]:
        cols: set[str] = set()
        for loaded in self.models.values():
            cols.update(getattr(loaded.model, "feature_names", []))
        # Engineered columns are generated, so they are not raw-input requirements.
        return sorted(c for c in cols if c not in {"age_to_term_ratio", "balance_ratio", "dpd_bucket"})

    def validate_schema(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise InferenceError("The input dataset is empty.")
        missing = sorted(set(self.required_columns) - set(frame.columns))
        if missing:
            raise InferenceError(
                "Input is missing columns required by the saved model: " + ", ".join(missing)
            )

    @staticmethod
    def _coerce_features(features: pd.DataFrame, model: Any) -> pd.DataFrame:
        expected = list(getattr(model, "feature_names", []))
        if not expected:
            raise InferenceError("A saved model has no feature metadata; retrain before inference.")
        aligned = features.reindex(columns=expected).copy()
        cat_columns = set(getattr(model, "cat_feature_names", []))
        for col in expected:
            if col in cat_columns:
                # pandas Categoricals cannot receive a new sentinel category
                # until converted to an object/string representation.
                aligned[col] = aligned[col].astype(object).where(aligned[col].notna(), "__MISSING__").astype(str)
            else:
                aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
        if aligned.isna().any().any():
            bad = aligned.columns[aligned.isna().any()].tolist()
            raise InferenceError(
                "Input contains invalid or missing numeric values for: " + ", ".join(bad)
            )
        return aligned

    @staticmethod
    def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(model.predict_proba(X))
        if probability.ndim == 1:
            return probability.astype(float)
        if probability.shape[1] < 2:
            raise InferenceError("A binary model returned no positive-class probability.")
        return probability[:, 1].astype(float)

    @staticmethod
    def _top_drivers(model: Any, row: pd.Series, limit: int = 3) -> str:
        try:
            importance = model.get_feature_importance(top_n=limit)
            usable = importance[importance["importance"] > 0]["feature"].tolist()
            return "; ".join(f"{name}={row.get(name, 'n/a')}" for name in usable) or "not available"
        except Exception:
            return "not available"

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score a raw input frame; targets are never used as features or evaluated."""
        self.validate_schema(frame)
        if self.preprocessing is not None:
            features = self.preprocessing.transform(frame)
        else:
            features, _ = FeaturePipeline.build_feature_matrix(frame, include_engineered=True)
        # FeaturePipeline preserves the original index, including non-default upload indices.
        output = pd.DataFrame(index=frame.index)
        output["loan_id"] = frame["loan_id"] if "loan_id" in frame else frame.index.astype(str)
        primary_model = next(iter(self.models.values())).model

        for target, output_col in TARGET_OUTPUTS.items():
            if target in self.models:
                X = self._coerce_features(features, self.models[target].model)
                output[output_col] = self._positive_probability(self.models[target].model, X)

        if "next_state" in self.models:
            model = self.models["next_state"].model
            X = self._coerce_features(features, model)
            output["next_state"] = model.predict(X)
        else:
            output["next_state"] = pd.Series([None] * len(output), index=output.index)

        output["anomaly_score"] = np.nan
        output["anomaly_flag"] = False
        output["exception_type"] = pd.Series([None] * len(output), index=output.index)
        if "exception_type" in self.models:
            model = self.models["exception_type"].model
            output["exception_type"] = model.predict(self._coerce_features(features, model))

        output["top_drivers"] = [self._top_drivers(primary_model, features.loc[index]) for index in output.index]
        risk_columns = [c for c in ("delinquency_probability", "default_probability", "prepayment_probability", "exception_probability") if c in output]
        output["confidence"] = output[risk_columns].apply(lambda row: float(np.max(np.abs(row - 0.5) * 2)), axis=1) if risk_columns else np.nan
        output["recommended_reviewer_action"] = np.where(
            output.get("exception_probability", pd.Series(0.0, index=output.index)).fillna(0) >= 0.5,
            "escalate_for_review", "monitor"
        )
        for column in output.columns:
            output[column] = output[column].map(self._normalize_value)
        return output.reset_index(drop=True)
