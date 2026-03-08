"""
Phase 4: Evaluation + MLflow Tracking
Loads judged cases from SQLite, computes accuracy metrics, logs to MLflow.

Usage:
    python src/eval.py
"""

import json
from collections import defaultdict
from pathlib import Path

import mlflow
from sklearn.metrics import confusion_matrix

from db import get_judged_cases

MLFLOW_DIR = Path(__file__).parent.parent / "mlflow"
LABELS = ["affirmed", "reversed", "vacated"]


def compute_metrics(cases: list[dict]) -> dict:
    total = len(cases)
    correct = sum(1 for c in cases if c["match"])
    accuracy = correct / total if total else 0.0

    by_decade: dict[str, list] = defaultdict(list)
    by_type: dict[str, list] = defaultdict(list)

    for c in cases:
        decade = f"{(c['year'] // 10) * 10}s" if c.get("year") else "unknown"
        by_decade[decade].append(c["match"])
        by_type[c.get("case_type") or "unknown"].append(c["match"])

    accuracy_by_decade = {
        dec: sum(matches) / len(matches)
        for dec, matches in sorted(by_decade.items())
    }
    accuracy_by_type = {
        t: sum(matches) / len(matches)
        for t, matches in sorted(by_type.items())
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_by_decade": accuracy_by_decade,
        "accuracy_by_type": accuracy_by_type,
    }


def log_to_mlflow(metrics: dict, cases: list[dict]):
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("ai-judge")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("total_cases", metrics["total"])
        mlflow.log_metric("correct_cases", metrics["correct"])

        for decade, acc in metrics["accuracy_by_decade"].items():
            mlflow.log_metric(f"accuracy_{decade}", acc)
        for case_type, acc in metrics["accuracy_by_type"].items():
            mlflow.log_metric(f"accuracy_{case_type}", acc)

        # Confusion matrix artifact
        actuals = [c["actual_verdict"] for c in cases if c.get("actual_verdict")]
        preds = [c["ai_verdict"] for c in cases if c.get("actual_verdict")]
        if actuals:
            present_labels = sorted(set(actuals + preds) & set(LABELS))
            cm = confusion_matrix(actuals, preds, labels=present_labels)
            cm_dict = {
                "labels": present_labels,
                "matrix": cm.tolist(),
            }
            cm_path = MLFLOW_DIR / "confusion_matrix.json"
            cm_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cm_path, "w") as f:
                json.dump(cm_dict, f, indent=2)
            mlflow.log_artifact(str(cm_path))

        mlflow.log_params({"model": "claude-sonnet", "top_k": 3})
        print(f"Logged run to MLflow ({MLFLOW_DIR})")


def main():
    cases = get_judged_cases()
    if not cases:
        print("No judged cases found. Run judge.py first.")
        return

    metrics = compute_metrics(cases)

    print(f"\n=== AI Judge Evaluation ===")
    print(f"Total cases judged : {metrics['total']}")
    print(f"Correct            : {metrics['correct']}")
    print(f"Overall accuracy   : {metrics['accuracy']:.1%}")
    print(f"\nAccuracy by decade:")
    for dec, acc in metrics["accuracy_by_decade"].items():
        print(f"  {dec}: {acc:.1%}")
    print(f"\nAccuracy by case type:")
    for t, acc in metrics["accuracy_by_type"].items():
        print(f"  {t}: {acc:.1%}")

    # Flag mismatches
    mismatches = [c for c in cases if c["match"] is False]
    print(f"\nMismatches ({len(mismatches)}):")
    for c in mismatches[:10]:
        print(f"  {c['name']} ({c['year']}) — actual: {c['actual_verdict']}, AI: {c['ai_verdict']}")

    log_to_mlflow(metrics, cases)


if __name__ == "__main__":
    main()
