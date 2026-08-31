from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_review_endpoint():
    payload = {
        "loan_id": 101,
        "current_balance": 80000,
        "original_balance": 100000,
        "days_past_due": 60,
        "credit_score_band": "Low",
    }
    response = client.post("/review-loan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["loan_id"] == 101
    assert "recommended_action" in data
    assert "risk_score" in data
