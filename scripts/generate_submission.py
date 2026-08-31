"""Generate a submission file for the challenge dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import InferenceError, LoanInferenceService


def main():
    raw_dir = Path("data") / "raw"
    test_path = raw_dir / "loan_monthly_performance_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Expected input file not found: {test_path}")

    df = pd.read_csv(test_path)
    try:
        predictions = LoanInferenceService().predict(df)
    except InferenceError as exc:
        raise RuntimeError(f"Cannot generate a submission: {exc}") from exc

    template_path = raw_dir / "submission_template.csv"
    if template_path.exists():
        template_columns = pd.read_csv(template_path, nrows=0).columns.tolist()
        missing = [column for column in template_columns if column not in predictions.columns]
        if missing:
            raise ValueError("Saved models cannot produce required template fields: " + ", ".join(missing))
        predictions = predictions[template_columns]

    output_path = Path("outputs") / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
