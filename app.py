from fastapi import FastAPI
from pydantic import BaseModel

from src.llm.copilot import GroundedReviewer

app = FastAPI(title="Loan Performance Intelligence Engine API")
reviewer = GroundedReviewer()


class LoanReviewRequest(BaseModel):
    loan_id: int
    current_balance: float | None = None
    original_balance: float | None = None
    days_past_due: float | None = None
    credit_score_band: str | None = None


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "loan-performance-intelligence"}


@app.post("/review-loan")
def review_loan(payload: LoanReviewRequest):
    loan_record = payload.model_dump()
    review = reviewer.review_case(loan_record)
    return review
