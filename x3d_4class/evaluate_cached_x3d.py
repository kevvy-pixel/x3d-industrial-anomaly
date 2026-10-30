from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

if os.name == "nt":
    cuda_bin = str(Path(sys.executable).resolve().parent / "bin")
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(cuda_bin)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PRIVATE_PACKAGES = Path(__file__).resolve().parent / "packages"
if PRIVATE_PACKAGES.exists():
    sys.path.append(str(PRIVATE_PACKAGES))

REPORT_CLASSES = ("normal", "single_hand", "board_drop", "touch_face")
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


class CachedDataset(Dataset):
    def __init__(self, rows: list[dict], model_classes: list[str]):
        self.rows = rows
        self.class_to_idx = {c: i for i, c in enumerate(model_classes)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        array = np.load(row["cache_path"], allow_pickle=False)
        clip = torch.from_numpy(array).permute(3, 0, 1, 2).float().div_(255.0)
        clip = (clip - MEAN) / STD
        return clip, self.class_to_idx[row["label"]], row["sample_id"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_classes = list(checkpoint["classes"])
    if set(model_classes) != set(REPORT_CLASSES):
        raise RuntimeError(f"Class mismatch: {model_classes}")
    from pytorchvideo.models.hub import x3d_m
    model = x3d_m(pretrained=False)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, len(model_classes))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    with args.cache_index.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    cached_window = rows[0].get("cache_window_sec") if rows else None
    loader = DataLoader(
        CachedDataset(rows, model_classes), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    truths, predictions, records = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for clips, targets, sample_ids in tqdm(loader, desc="evaluate"):
            logits = model(clips.to(device, non_blocking=True))
            probs = torch.softmax(logits, 1).cpu().numpy()
            pred_idx = probs.argmax(1)
            for sid, target, pred, prob in zip(sample_ids, targets.tolist(), pred_idx.tolist(), probs):
                truth_name, pred_name = model_classes[target], model_classes[pred]
                truths.append(truth_name)
                predictions.append(pred_name)
                records.append({
                    "sample_id": sid, "truth": truth_name, "prediction": pred_name,
                    "confidence": float(prob[pred]),
                    **{f"prob_{c}": float(prob[i]) for i, c in enumerate(model_classes)},
                })
    elapsed = time.perf_counter() - started
    precision, recall, per_f1, support = precision_recall_fscore_support(
        truths, predictions, labels=list(REPORT_CLASSES), zero_division=0,
    )
    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_classes": model_classes,
        "test_samples": len(rows),
        "elapsed_sec": elapsed,
        "samples_per_sec": len(rows) / elapsed,
        "batch_size": args.batch_size,
        "accuracy": accuracy_score(truths, predictions),
        "macro_f1": f1_score(truths, predictions, labels=list(REPORT_CLASSES), average="macro", zero_division=0),
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(per_f1[i]), "support": int(support[i])}
            for i, c in enumerate(REPORT_CLASSES)
        },
        "confusion_matrix_labels": REPORT_CLASSES,
        "confusion_matrix": confusion_matrix(truths, predictions, labels=list(REPORT_CLASSES)).tolist(),
        "cache_window_sec": float(cached_window) if cached_window else "manifest_1.0_0.8_0.7_by_fps",
        "note": ("Centered 2.666667-second, 16-frame validation windows with boundary shifting."
                 if cached_window else
                 "Fixed 1.0/0.8/0.7-second, 16-frame validation windows selected by FPS."),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
