"""Optional, grounded Gemini reviewer integration.

This module is deliberately downstream of the ML engine: it accepts only an
already-computed evidence payload and never reads raw portfolio data itself.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DISCLAIMER = "AI recommendation only. Final decision remains with the human reviewer."
PROMPT_VERSION = "gemini-grounded-reviewer-v1"
SYSTEM_INSTRUCTION = """You are a loan-review assistant. You do not make lending decisions.
You must only use evidence supplied by the application. Do not invent loan information,
model results, probabilities, business rules, or policy requirements. If evidence is
insufficient, say exactly: Insufficient evidence to determine this. Separate observed
facts from interpretation. Your output is a recommendation for a human reviewer; the
human reviewer makes the final decision."""
REQUIRED_RESPONSE = {
    "summary": "",
    "key_evidence": [],
    "risk_drivers": [],
    "data_quality_concerns": [],
    "recommended_reviewer_action": "Insufficient evidence to determine this.",
    "confidence_statement": "Insufficient evidence to determine this.",
    "limitations": [],
    "human_decision_required": True,
}


class GeminiUnavailable(RuntimeError):
    """Gemini is not configured or cannot serve a request."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _api_key() -> str | None:
    if key := os.getenv("GEMINI_API_KEY"):
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


class GeminiReviewerClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout_seconds: int = 30, retries: int = 2):
        self.api_key = api_key or _api_key()
        # Gemini 2.5 Flash remains supported for existing projects but Google
        # no longer enables it for some new keys.  Use the current stable Flash
        # endpoint unless the deployment explicitly configures another model.
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _dictionary_context() -> str:
        for path in (Path("data_dictionary.md"), Path("data/raw/data_dictionary.md")):
            if path.exists():
                return path.read_text(encoding="utf-8")[:12000]
        return "No approved data dictionary was supplied."

    @staticmethod
    def _validate_evidence(evidence: Mapping[str, Any]) -> None:
        if not evidence or not evidence.get("loan_id"):
            raise GeminiUnavailable("Missing ML evidence or loan_id; Gemini will not analyze an ungrounded loan.")

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = {**REQUIRED_RESPONSE, "summary": text, "limitations": ["Gemini response was not valid JSON."]}
        if not isinstance(parsed, dict):
            parsed = {**REQUIRED_RESPONSE, "summary": str(parsed), "limitations": ["Gemini response was not a JSON object."]}
        result = {**REQUIRED_RESPONSE, **parsed, "human_decision_required": True}
        for key in ("key_evidence", "risk_drivers", "data_quality_concerns", "limitations"):
            if not isinstance(result[key], list):
                result[key] = [str(result[key])]
        # Never surface an authoritative lending instruction as the action.
        action = str(result["recommended_reviewer_action"])
        if re.search(r"\b(approve|reject)(?:\s+this)?\s+loan\b", action, re.IGNORECASE):
            result["recommended_reviewer_action"] = "Escalate to a human reviewer; Gemini cannot make a lending decision."
        result["disclaimer"] = DISCLAIMER
        return result

    @staticmethod
    def _log(payload: dict[str, Any], log_path: Path = Path("outputs/llm_logs.jsonl")) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), default=str) + "\n")

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        """Return a useful diagnosis without echoing provider response details or secrets."""
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        if "429" in text or "resourceexhausted" in name:
            return "Gemini rate limit reached. Wait and try again. ML analysis remains available."
        if "404" in text or "notfound" in name:
            return "Configured Gemini model is unavailable to this API key. Check GEMINI_MODEL and project access."
        if "401" in text or "403" in text or "clienterror" in name or "permission" in text or "api key" in text:
            return "Gemini rejected the request. Verify the API key, its Google project/API access, and access to the configured model."
        if "timeout" in name or "timeout" in text:
            return "Gemini request timed out. Try again; ML analysis remains available."
        return "Reviewer Copilot unavailable due to a Gemini service error. ML analysis remains available."

    def review(self, evidence: Mapping[str, Any], purpose: str = "loan_risk_summary", question: str | None = None) -> dict[str, Any]:
        self._validate_evidence(evidence)
        request_id = str(uuid.uuid4())
        base_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
            "model": self.model, "purpose": purpose, "loan_id": str(evidence["loan_id"]),
            "grounded_context_identifiers": list(evidence.keys()), "prompt_version": PROMPT_VERSION,
            "reviewer_status": "pending",
        }
        if not self.configured:
            message = "Reviewer Copilot unavailable. ML analysis remains available. GEMINI_API_KEY is not configured."
            self._log({**base_log, "success": False, "failure": "missing_api_key", "output": message})
            raise GeminiUnavailable(message)
        prompt = json.dumps({"task": purpose, "reviewer_question": question, "ml_evidence": _json_safe(evidence),
                             "approved_data_dictionary": self._dictionary_context(), "response_schema": REQUIRED_RESPONSE,
                             "required_disclaimer": DISCLAIMER}, default=str)
        try:
            from google import genai
        except ImportError as exc:
            message = "Reviewer Copilot unavailable. ML analysis remains available. Install google-genai to enable Gemini."
            self._log({**base_log, "success": False, "failure": "sdk_unavailable", "output": message})
            raise GeminiUnavailable(message) from exc
        for attempt in range(self.retries + 1):
            try:
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(model=self.model, contents=prompt,
                    config={"system_instruction": SYSTEM_INSTRUCTION, "response_mime_type": "application/json"})
                result = self._parse_response(response.text or "")
                self._log({**base_log, "success": True, "output": result})
                return {"request_id": request_id, **result}
            except Exception as exc:
                if attempt == self.retries:
                    message = self._safe_error_message(exc)
                    self._log({**base_log, "success": False, "failure": type(exc).__name__, "output": message})
                    raise GeminiUnavailable(message) from exc
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def record_feedback(self, request_id: str, status: str, reason: str = "") -> None:
        if status not in {"accepted", "rejected", "corrected"}:
            raise ValueError("Reviewer status must be accepted, rejected, or corrected.")
        self._log({"timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
                   "model": self.model, "purpose": "human_reviewer_feedback", "success": True,
                   "reviewer_status": status, "reason": reason})
