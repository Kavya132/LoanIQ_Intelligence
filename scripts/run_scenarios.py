"""Generate scenario reports from stress assumptions."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.simulator import ScenarioSimulator
from src.reports.generator import build_scenario_report


def main():
    baseline = {
        "base_case": 0.02,
        "adverse": 0.05,
        "high_prepayment": 0.08,
    }
    results = ScenarioSimulator().run_scenarios(baseline)
    report_path = build_scenario_report(results, Path("reports"))
    print(f"Scenario report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
