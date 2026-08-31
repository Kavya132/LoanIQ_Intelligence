"""Score a new loan CSV with the same persisted artifacts used by Streamlit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.inference import InferenceError, LoanInferenceService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV to score")
    parser.add_argument("--output", default="outputs/submission.csv", help="Destination CSV")
    parser.add_argument("--models-dir", default="outputs/models")
    args = parser.parse_args()
    try:
        predictions = LoanInferenceService(args.models_dir).predict(pd.read_csv(args.input))
    except (OSError, pd.errors.ParserError, InferenceError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions)} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
