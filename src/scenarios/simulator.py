"""Scenario simulation utilities for stress testing loan portfolios."""

from __future__ import annotations

from typing import Dict, Any


class ScenarioSimulator:
    """Generate base, adverse, and prepayment stress scenarios."""

    def __init__(self, baseline_multiplier: float = 1.0):
        self.baseline_multiplier = baseline_multiplier

    def run_scenarios(self, baseline: Dict[str, float]) -> Dict[str, Any]:
        """Return scenario impacts based on a baseline assumption set."""
        if not baseline:
            baseline = {"base_case": 0.02, "adverse": 0.05, "high_prepayment": 0.08}

        base_case = float(baseline.get("base_case", 0.02)) * self.baseline_multiplier
        adverse = float(baseline.get("adverse", 0.05)) * self.baseline_multiplier
        high_prepayment = float(baseline.get("high_prepayment", 0.08)) * self.baseline_multiplier

        return {
            "base_case": {"default_rate": base_case, "loss_rate": base_case * 1.5},
            "adverse": {"default_rate": adverse, "loss_rate": adverse * 1.8},
            "high_prepayment": {"default_rate": high_prepayment, "loss_rate": high_prepayment * 1.2},
        }
