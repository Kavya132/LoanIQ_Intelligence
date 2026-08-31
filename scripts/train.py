from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.train_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
