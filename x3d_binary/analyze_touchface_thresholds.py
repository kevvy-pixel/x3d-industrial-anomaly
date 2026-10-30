from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def metrics(rows: list[dict[str, str]], threshold: float) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        positive = row["truth"] == "touch_face"
        predicted = float(row["touch_face_probability"]) >= threshold
        if positive and predicted:
            tp += 1
        elif positive:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    touch_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    normal_precision = tn / (tn + fn) if tn + fn else 0.0
    normal_recall = tn / (tn + fp) if tn + fp else 0.0
    normal_f1 = (
        2 * normal_precision * normal_recall / (normal_precision + normal_recall)
        if normal_precision + normal_recall
        else 0.0
    )
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "touch_face_f1": touch_f1,
        "macro_f1": (touch_f1 + normal_f1) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    predictions = args.experiment / "best_predictions.csv"
    with predictions.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    results = [metrics(rows, step / 1000) for step in range(1, 1000)]
    best_touch_f1 = max(results, key=lambda item: (item["touch_face_f1"], item["threshold"]))
    best_macro_f1 = max(results, key=lambda item: (item["macro_f1"], item["threshold"]))
    precision_80 = [item for item in results if item["precision"] >= 0.80]
    recall_90 = [item for item in results if item["recall"] >= 0.90]
    payload = {
        "scope": "fixed validation clips; probabilities are before stream smoothing/confirmation",
        "count": len(rows),
        "default_0_5": metrics(rows, 0.5),
        "best_touch_face_f1": best_touch_f1,
        "best_macro_f1": best_macro_f1,
        "best_recall_with_precision_at_least_0_80": max(
            precision_80, key=lambda item: (item["recall"], item["touch_face_f1"])
        ) if precision_80 else None,
        "best_precision_with_recall_at_least_0_90": max(
            recall_90, key=lambda item: (item["precision"], item["touch_face_f1"])
        ) if recall_90 else None,
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
    }
    output = args.experiment / "threshold_analysis.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
