import json

import pytest

from src.llm.gemini_client import GeminiReviewerClient, GeminiUnavailable


def test_missing_api_key_keeps_ml_flow_available(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.gemini_client._api_key", lambda: None)
    client = GeminiReviewerClient(api_key="")
    assert not client.configured
    with pytest.raises(GeminiUnavailable, match="ML analysis remains available"):
        client.review({"loan_id": "L1"})


def test_timeout_and_retries_are_more_robust_for_gemini():
    client = GeminiReviewerClient(api_key="test")
    assert client.timeout_seconds >= 10
    assert client.timeout_ms >= 10000
    assert client.retries >= 2


def test_generic_dashboard_evidence_is_allowed_without_loan_id():
    client = GeminiReviewerClient(api_key="test")
    client._validate_evidence({"data_quality_summary": {"missing_values": 0}, "scope": "dataset"})


def test_missing_evidence_is_rejected_before_api_call():
    with pytest.raises(GeminiUnavailable, match="Missing ML evidence"):
        GeminiReviewerClient(api_key="test").review({})


def test_malformed_response_falls_back_to_safe_structured_output():
    result = GeminiReviewerClient._parse_response("plain text, not JSON")
    assert result["summary"] == "plain text, not JSON"
    assert result["human_decision_required"] is True
    assert result["limitations"]


def test_malformed_json_response_is_sanitized_for_ui():
    messy = """The dataset contains a total of 66,\n\nGrounded details\n{\n\"key_evidence\":[]\n\"risk_drivers\":[]\n}\nAI recommendation only."""
    result = GeminiReviewerClient._parse_response(messy)
    assert "Grounded details" not in result["summary"]
    assert "Insufficient evidence to determine this." in result["recommended_reviewer_action"]
    assert any("not valid JSON" in item.lower() for item in result["limitations"])


def test_authoritative_lending_action_is_sanitized():
    result = GeminiReviewerClient._parse_response(json.dumps({"recommended_reviewer_action": "Approve this loan"}))
    assert "cannot make a lending decision" in result["recommended_reviewer_action"]


def test_reviewer_rejection_is_logged(tmp_path):
    client = GeminiReviewerClient(api_key="test")
    # Redirect the static log writer for this isolated audit test.
    original_log = client._log
    client._log = lambda payload: original_log(payload, tmp_path / "llm_logs.jsonl")
    client.record_feedback("request-1", "rejected", "Balance evidence requires correction")
    entry = json.loads((tmp_path / "llm_logs.jsonl").read_text().strip())
    assert entry["reviewer_status"] == "rejected"
    assert entry["reason"] == "Balance evidence requires correction"
