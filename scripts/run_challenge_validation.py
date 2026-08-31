#!/usr/bin/env python3
"""Challenge-style validation summary for the project."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_file(path: Path) -> str:
    if path.exists():
        return "OK"
    return "MISSING"


required_paths = {
    "Core app": ROOT / "app.py",
    "Dashboard": ROOT / "dashboard" / "app.py",
    "Config": ROOT / "config.yaml",
    "Requirements": ROOT / "requirements.txt",
    "Demo data script": ROOT / "scripts" / "generate_demo_data.py",
    "Training pipeline": ROOT / "src" / "pipeline" / "train_pipeline.py",
    "Model trainer": ROOT / "src" / "models" / "trainer.py",
    "LLM reviewer": ROOT / "src" / "llm" / "copilot.py",
    "Dockerfile": ROOT / "Dockerfile",
    "Model card": ROOT / "MODEL_CARD.md",
    "AI log": ROOT / "AI_DEVELOPMENT_LOG.md",
}

optional_paths = {
    "Outputs folder": ROOT / "outputs",
    "Reports folder": ROOT / "reports",
    "Demo data raw": ROOT / "data" / "raw",
}

print("CHALLENGE VALIDATION SUMMARY")
print("=" * 32)
for name, path in required_paths.items():
    status = check_file(path)
    print(f"{name:<22} {status}")

print("\nOPTIONAL/GENERATED")
print("=" * 32)
for name, path in optional_paths.items():
    status = check_file(path)
    print(f"{name:<22} {status}")

metrics_file = ROOT / "outputs" / "model_metrics.json"
if metrics_file.exists():
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    print(f"\nModel metric entries: {len(data)}")
else:
    print("\nModel metric entries: 0")

print("\nAssessment: ML-first project with optional grounded reviewer assistant.")
