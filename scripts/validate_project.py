#!/usr/bin/env python3
"""Validate the project package and key execution requirements."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = [
    ROOT / "README.md",
    ROOT / "config.yaml",
    ROOT / ".env.example",
    ROOT / "requirements.txt",
    ROOT / "Dockerfile",
    ROOT / "MODEL_CARD.md",
    ROOT / "AI_DEVELOPMENT_LOG.md",
    ROOT / "app.py",
    ROOT / "dashboard" / "app.py",
    ROOT / "scripts" / "generate_demo_data.py",
    ROOT / "scripts" / "train.py",
    ROOT / "scripts" / "evaluate.py",
    ROOT / "src" / "pipeline" / "train_pipeline.py",
    ROOT / "src" / "llm" / "copilot.py",
    ROOT / "src" / "models" / "trainer.py",
]

missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    print("MISSING REQUIRED FILES:")
    for item in missing:
        print(f" - {item}")
    raise SystemExit(1)

# Smoke-check that key runtime modules import correctly.
try:
    import app  # noqa: F401
    import src.pipeline.train_pipeline  # noqa: F401
    import src.llm.copilot  # noqa: F401
    from src.config.settings import get_settings
    get_settings()
except Exception as exc:  # pragma: no cover - smoke validation
    print(f"IMPORT VALIDATION FAILED: {exc}")
    raise SystemExit(1)

print("Project validation passed: required files and key modules are present and importable.")
