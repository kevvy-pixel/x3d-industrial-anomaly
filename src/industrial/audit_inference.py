from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))
from train_industrial import WindowDataset, build_model  # noqa: E402


@torch.inference_mode()
def infer(model, loader, device, classes, label: str) -> list[dict]:
    records: list[dict] = []
    for batch_index, (videos, labels, meta) in enumerate(loader, start=1):
        videos = videos.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(videos)
        probabilities = logits.softmax(1).cpu()
        predictions = logits.argmax(1).cpu().tolist()
        targets = labels.tolist()
        for item_index, (target, prediction) in enumerate(zip(targets, predictions)):
            row = {key: meta[key][item_index] for key in meta}
            row.update({"target": target, "prediction": prediction})
            row.update({f"prob_{index}": float(value) for index, value in enumerate(probabilities[item_index])})
            records.append(row)
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            print(f"{label}: batch {batch_index}/{len(loader)}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for fold in range(int(cfg["data"]["folds"])):
        checkpoint_path = Path(cfg["project_root"]) / "checkpoints" / f"fold_{fold}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = build_model(len(cfg["classes"]), device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        summaries[str(fold)] = {}
        manifest_root = Path(cfg["data"]["manifest_root"]) / f"fold_{fold}"
        for split in ("train", "val"):
            # Audit inference is deterministic: no augmentation, temporal jitter,
            # random normal-window selection, or balanced replacement sampling.
            dataset = WindowDataset(manifest_root / f"{split}.csv", cfg, train=False)
            loader = DataLoader(
                dataset,
                batch_size=int(cfg["training"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            records = infer(model, loader, device, cfg["classes"], f"fold {fold} {split}")
            frame = pd.DataFrame(records)
            frame.to_csv(args.output / f"fold_{fold}_{split}_predictions.csv", index=False)
            summaries[str(fold)][split] = {
                "count": len(frame),
                "accuracy": accuracy_score(frame.target, frame.prediction),
                "macro_f1": f1_score(frame.target, frame.prediction, average="macro", zero_division=0),
            }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for split in ("train", "val"):
        pooled = pd.concat(
            [pd.read_csv(args.output / f"fold_{fold}_{split}_predictions.csv") for fold in range(int(cfg["data"]["folds"]))],
            ignore_index=True,
        )
        pooled.to_csv(args.output / f"pooled_{split}_predictions.csv", index=False)
        report = classification_report(
            pooled.target,
            pooled.prediction,
            labels=list(range(len(cfg["classes"]))),
            target_names=cfg["classes"],
            output_dict=True,
            zero_division=0,
        )
        (args.output / f"pooled_{split}_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    (args.output / "fold_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(args.output), "device": str(device)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
