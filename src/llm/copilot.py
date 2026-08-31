"""Grounded reviewer copilot for loan case summaries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime, timezone


class GroundedReviewer:
    """Lightweight, deterministic reviewer that provides evidence-based summaries."""

    def __init__(self, model_name: str = "heuristic-grounded-reviewer"):
        self.model_name = model_name

    def review_case(
        self,
        loan_record: Dict[str, Any],
        model_summary: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a short, grounded review for a single loan record."""
        model_summary = model_summary or {}
        probabilities = [
            value for key, value in model_summary.items()
            if key.endswith("probability") and isinstance(value, (int, float))
        ]
        # The copilot is not a prediction engine: its risk level is derived only
        # from supplied ML/rule evidence.  A direct API call with no evidence
        # remains reviewable but is explicitly marked as unavailable.
        risk_score = max(probabilities) if probabilities else 0.0
        action = "monitor" if risk_score < 0.5 else "escalate_for_review"
        if risk_score > 0.75:
            action = "credit_review_and_collection"

        summary = (
            f"Loan {loan_record.get('loan_id', 'unknown')} has a {risk_score:.2f} review risk level "
            f"derived from the supplied model and rule evidence."
        )

        return {
            "loan_id": loan_record.get("loan_id", "unknown"),
            "model_name": self.model_name,
            "risk_score": round(risk_score, 4),
            "recommended_action": action,
            "summary": summary,
            "evidence": evidence or [
                {"field": "current_balance", "value": loan_record.get("current_balance")},
                {"field": "days_past_due", "value": loan_record.get("days_past_due")},
                {"field": "credit_score_band", "value": loan_record.get("credit_score_band")},
            ],
            "model_summary": model_summary or {"status": "no model evidence supplied"},
            "disclaimer": "Recommendation only — final decision remains with human reviewer.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def log_review(self, review: Dict[str, Any], log_path: str | Path | None = None) -> str:
        """Persist review output to JSONL for auditability."""
        if log_path is None:
            log_path = Path("outputs") / "llm_logs.jsonl"
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "grounded reviewer evidence", "model": self.model_name,
                                "grounded_context": review.get("model_summary"), "output": review,
                                "timestamp": datetime.now(timezone.utc).isoformat()}, default=str) + "\n")
        return str(log_path)
