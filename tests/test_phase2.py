import pandas as pd
import numpy as np

from src.anomaly.detector import IsolationForestAnomalyDetector
from src.explainability.shap_explainer import SHAPExplainer
from src.scenarios.simulator import ScenarioSimulator


def test_isolation_forest_anomaly_detector_flags_anomaly():
    df = pd.DataFrame({
        "feature_a": [10, 11, 12, 13, 100],
        "feature_b": [9, 10, 11, 12, 110],
    })

    detector = IsolationForestAnomalyDetector(contamination=0.2, random_state=42)
    results = detector.fit_predict(df)

    assert "anomaly_score" in results.columns
    assert "is_anomaly" in results.columns
    assert results["is_anomaly"].isin([0, 1]).all()
    assert results["is_anomaly"].sum() >= 1


def test_shap_explainer_returns_feature_importance():
    X = pd.DataFrame({
        "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "x2": [0.5, 0.4, 0.3, 0.2, 0.1],
        "x3": [1, 2, 3, 4, 5],
    })
    y = pd.Series([0, 0, 1, 1, 1])

    explainer = SHAPExplainer()
    feature_importance = explainer.explain_model(X, y)

    assert isinstance(feature_importance, pd.DataFrame)
    assert not feature_importance.empty
    assert {"feature", "importance"}.issubset(set(feature_importance.columns))


def test_scenario_simulator_generates_stress_outputs():
    baseline = {"base_case": 0.02, "adverse": 0.05, "high_prepayment": 0.08}
    simulator = ScenarioSimulator()
    results = simulator.run_scenarios(baseline)

    assert isinstance(results, dict)
    assert "base_case" in results
    assert "adverse" in results
    assert "high_prepayment" in results
