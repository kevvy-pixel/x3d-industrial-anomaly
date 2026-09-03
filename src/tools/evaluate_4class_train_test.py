from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "industrial"
sys.path.insert(0, str(SCRIPT_ROOT))
from train_industrial import WindowDataset, build_model, evaluate  # noqa: E402


def report_rows(frame: pd.DataFrame, classes: list[str], split: str) -> list[dict]:
    report = classification_report(
        frame.target,
        frame.prediction,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    return [
        {
            "split": split,
            "class_name": name,
            "class_accuracy": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": int(report[name]["support"]),
        }
        for name in classes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate train and OOF-test class metrics for CV experiments.")
    parser.add_argument("--experiment", action="append", nargs=2, metavar=("NAME", "CONFIG"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    experiments = {name: Path(config) for name, config in args.experiment}
    all_rows, summaries = [], {}
    for experiment, config_path in experiments.items():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        classes = list(cfg["classes"])
        project_root = Path(cfg["project_root"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(len(classes), device)
        criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))
        train_records = []
        fold_summaries = []
        for fold in range(int(cfg["data"]["folds"])):
            dataset = WindowDataset(Path(cfg["data"]["manifest_root"]) / f"fold_{fold}" / "train.csv", cfg, train=False)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
            checkpoint = torch.load(project_root / "checkpoints" / f"fold_{fold}" / "best.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            result, records = evaluate(model, loader, criterion, device)
            for record in records:
                record["fold"] = fold
            train_records.extend(records)
            fold_summaries.append({"fold": fold, "accuracy": result.accuracy, "macro_f1": result.macro_f1, "count": len(records)})

        train_frame = pd.DataFrame(train_records)
        test_frame = pd.read_csv(project_root / "results" / "oof_test_predictions.csv")
        rows = report_rows(train_frame, classes, "train_manifest_pooled") + report_rows(test_frame, classes, "oof_test")
        for row in rows:
            row["experiment"] = experiment
        all_rows.extend(rows)
        summaries[experiment] = {
            "train_manifest_pooled": {
                "sample_occurrences": len(train_frame),
                "accuracy": accuracy_score(train_frame.target, train_frame.prediction),
                "macro_f1": f1_score(train_frame.target, train_frame.prediction, average="macro", zero_division=0),
                "folds": fold_summaries,
            },
            "oof_test": {
                "unique_windows": len(test_frame),
                "accuracy": accuracy_score(test_frame.target, test_frame.prediction),
                "macro_f1": f1_score(test_frame.target, test_frame.prediction, average="macro", zero_division=0),
            },
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_root / "four_class_train_test_class_metrics.csv", index=False)
    (output_root / "four_class_train_test_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(pd.DataFrame(all_rows).to_string(index=False))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
