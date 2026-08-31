from src.inference.service import LoanInferenceService


def test_normalize_value_replaces_missing_markers_with_none():
    assert LoanInferenceService._normalize_value("not_available") is None
    assert LoanInferenceService._normalize_value("Not available") is None
    assert LoanInferenceService._normalize_value("monitor") == "monitor"
    assert LoanInferenceService._normalize_value(0.42) == 0.42
