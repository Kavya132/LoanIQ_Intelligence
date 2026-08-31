from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
metrics_path = ROOT / "outputs" / "model_metrics.json"

if not metrics_path.exists():
    raise FileNotFoundError(f"Model metrics not found: {metrics_path}")

with metrics_path.open("r", encoding="utf-8") as f:
    metrics = json.load(f)

print("MODEL EVALUATION SUMMARY")
print("=" * 30)
for item in metrics:
    if isinstance(item, dict):
        name = item.get("model_name") or item.get("target") or "model"
        auc = item.get("roc_auc")
        f1 = item.get("f1")
        print(f"{name}: AUC={auc} F1={f1}")
