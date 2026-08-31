from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent


def test_project_artifacts_exist():
    required = [
        ROOT / "Dockerfile",
        ROOT / "MODEL_CARD.md",
        ROOT / "AI_DEVELOPMENT_LOG.md",
        ROOT / "scripts" / "train.py",
        ROOT / "scripts" / "evaluate.py",
        ROOT / "scripts" / "validate_project.py",
    ]
    for path in required:
        assert path.exists(), f"Missing required artifact: {path}"


def test_validate_project_script_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_project.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
