"""Artifact-backed inference for trained loan-performance models."""

from .service import InferenceError, LoanInferenceService

__all__ = ["InferenceError", "LoanInferenceService"]
