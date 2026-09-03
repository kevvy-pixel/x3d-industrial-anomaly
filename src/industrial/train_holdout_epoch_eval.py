from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_industrial import (
    Tee,
    WindowDataset,
    build_model,
    configure_phase,
    evaluate,
    save_checkpoint,
    seed_everything,
    seed_worker,
    train_epoch,
)


def make_loaders(cfg: dict, device: torch.device, smoke: bool = False) -> dict:
    manifest_root = Path(cfg["data"]["manifest_root"])
    train_dataset = WindowDataset(manifest_root / "train.csv", cfg, train=True)
    audit_train_dataset = WindowDataset(manifest_root / "train.csv", cfg, train=False)
    test_dataset = WindowDataset(manifest_root / "test.csv", cfg, train=False)
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    batch_size = 1 if smoke else int(cfg["training"]["batch_size"])
    common = {
        "batch_size": batch_size,
        "num_workers": 0 if smoke else int(cfg["training"]["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    counts = np.bincount(train_dataset.labels, minlength=len(cfg["classes"]))
    if np.any(counts == 0):
        raise RuntimeError(f"Training split has an empty class: {counts.tolist()}")
    weights = [1.0 / counts[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=1 if smoke else int(cfg["training"]["balanced_samples_per_epoch"]),
        replacement=True,
        generator=generator,
    )
    return {
        "train": DataLoader(train_dataset, sampler=sampler, **common),
        "audit_train": DataLoader(audit_train_dataset, shuffle=False, **common),
        "test": DataLoader(test_dataset, shuffle=False, **common),
    }


def plot_train_history(history: list[dict], path: Path) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(frame.epoch, frame.train_loss, marker="o", label="train (augmented sampler)")
    axes[0].plot(frame.epoch, frame.test_loss, marker="o", label="epoch-eval")
    axes[0].set_title("Loss")
    axes[1].plot(frame.epoch, frame.train_accuracy, marker="o", label="train (augmented sampler)")
    axes[1].plot(frame.epoch, frame.test_accuracy, marker="o", label="epoch-eval")
    axes[1].set_title("Accuracy")
    axes[2].plot(frame.epoch, frame.train_macro_f1, marker="o", label="train (augmented sampler)")
    axes[2].plot(frame.epoch, frame.test_macro_f1, marker="o", label="epoch-eval")
    axes[2].set_title("Macro-F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def append_progress(path: Path, row: dict) -> None:
    records = []
    if path.exists():
        try:
            records = pd.read_csv(path).to_dict("records")
        except pd.errors.EmptyDataError:
            pass
    records = [item for item in records if int(item["epoch"]) != int(row["epoch"])] + [row]
    pd.DataFrame(records).sort_values("epoch").to_csv(path, index=False)


def save_evaluation(records: list[dict], cfg: dict, output_root: Path, prefix: str) -> dict:
    frame = pd.DataFrame(records)
    frame.to_csv(output_root / f"{prefix}_predictions.csv", index=False)
    targets, predictions = frame.target.tolist(), frame.prediction.tolist()
    labels = list(range(len(cfg["classes"])))
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=cfg["classes"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_root / f"{prefix}_classification_report.csv")
    matrix = confusion_matrix(targets, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=cfg["classes"], yticklabels=cfg["classes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_root / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return {
        "count": len(frame),
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def run(cfg: dict, device: torch.device) -> dict:
    seed_everything(int(cfg["seed"]))
    project_root = Path(cfg["project_root"])
    results_root = project_root / "results"
    checkpoint_root = project_root / "checkpoints"
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("Completed result already exists; no retraining performed.", flush=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return summary

    loaders = make_loaders(cfg, device)
    model = build_model(len(cfg["classes"]), device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"]["mixed_precision"]) and device.type == "cuda")
    phases = [
        ("head", int(cfg["training"]["head_epochs"]), float(cfg["training"]["head_lr"])),
        ("finetune", int(cfg["training"]["finetune_epochs"]), float(cfg["training"]["finetune_lr"])),
    ]
    history_path = results_root / "training_history.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    global_epoch = max((int(row["epoch"]) for row in history), default=0)
    best_f1, best_accuracy, best_loss, best_epoch = -1.0, -1.0, float("inf"), 0
    for old_row in history:
        old_f1 = float(old_row["test_macro_f1"])
        old_accuracy = float(old_row["test_accuracy"])
        old_loss = float(old_row["test_loss"])
        improved = old_f1 > best_f1 or (
            math.isclose(old_f1, best_f1)
            and (old_accuracy > best_accuracy or (math.isclose(old_accuracy, best_accuracy) and old_loss < best_loss))
        )
        if improved:
            best_f1, best_accuracy, best_loss, best_epoch = old_f1, old_accuracy, old_loss, int(old_row["epoch"])
    last_path = checkpoint_root / "last.pt"
    resume_checkpoint = None
    if history and last_path.exists():
        resume_checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state"])
        print(f"Resuming after epoch {global_epoch} from {last_path}", flush=True)
    elapsed_offset = max((float(row["elapsed_minutes"]) for row in history), default=0.0)
    started = time.time()

    for phase, epochs, base_lr in phases:
        completed = sum(str(row["phase"]) == phase for row in history)
        if completed >= epochs:
            continue
        resume_lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * completed / epochs))
        optimizer = torch.optim.AdamW(
            configure_phase(model, phase),
            lr=resume_lr,
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - completed))
        if resume_checkpoint and resume_checkpoint.get("phase") == phase and "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            if "scheduler_state" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            if "scaler_state" in resume_checkpoint:
                scaler.load_state_dict(resume_checkpoint["scaler_state"])
        for _ in range(epochs - completed):
            global_epoch += 1
            train_result = train_epoch(
                model,
                loaders["train"],
                optimizer,
                scaler,
                criterion,
                device,
                phase,
                int(cfg["training"]["accumulation_steps"]),
            )
            test_result, _ = evaluate(model, loaders["test"], criterion, device)
            row = {
                "epoch": global_epoch,
                "phase": phase,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "train_macro_f1": train_result.macro_f1,
                "test_loss": test_result.loss,
                "test_accuracy": test_result.accuracy,
                "test_macro_f1": test_result.macro_f1,
                "elapsed_minutes": elapsed_offset + (time.time() - started) / 60,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)
            append_progress(results_root / "training_progress.csv", row)
            plot_train_history(history, results_root / "training_curves.png")
            improved = test_result.macro_f1 > best_f1 or (
                math.isclose(test_result.macro_f1, best_f1)
                and (
                    test_result.accuracy > best_accuracy
                    or (math.isclose(test_result.accuracy, best_accuracy) and test_result.loss < best_loss)
                )
            )
            if improved:
                best_f1, best_accuracy, best_loss, best_epoch = (
                    test_result.macro_f1,
                    test_result.accuracy,
                    test_result.loss,
                    global_epoch,
                )
                save_checkpoint(checkpoint_root / "best_epoch_eval.pt", model, global_epoch, phase, best_f1, cfg)
            save_checkpoint(last_path, model, global_epoch, phase, test_result.macro_f1, cfg, optimizer, scheduler, scaler)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            scheduler.step()

    last_checkpoint = torch.load(last_path, map_location=device)
    model.load_state_dict(last_checkpoint["model_state"])
    _, last_test_records = evaluate(model, loaders["test"], criterion, device)
    last_test_metrics = save_evaluation(last_test_records, cfg, results_root, "last_epoch_test")

    checkpoint = torch.load(checkpoint_root / "best_epoch_eval.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    train_result, train_records = evaluate(model, loaders["audit_train"], criterion, device)
    test_result, test_records = evaluate(model, loaders["test"], criterion, device)
    train_metrics = save_evaluation(train_records, cfg, results_root, "best_epoch_train")
    test_metrics = save_evaluation(test_records, cfg, results_root, "best_epoch_test")
    test_frame = pd.DataFrame(test_records)
    hard = test_frame[test_frame.sample_origin == "normal_from_anomaly"]
    summary = {
        "split": "source_stratified_80_20_holdout",
        "model_selection": "best epoch selected by repeated evaluation on the 20% epoch-eval split",
        "statistical_warning": "the 20% split is used for model selection and is not an unbiased final test set",
        "best_epoch": best_epoch,
        "best_epoch_eval_macro_f1": best_f1,
        "best_epoch_eval_accuracy": best_accuracy,
        "best_epoch_eval_loss": best_loss,
        "final_epoch": global_epoch,
        "best_epoch_train": train_metrics,
        "best_epoch_eval": test_metrics,
        "last_epoch_eval": last_test_metrics,
        "test_hard_negative_count": len(hard),
        "test_hard_negative_normal_recall": float((hard.prediction == hard.target).mean()) if len(hard) else None,
        "elapsed_minutes": elapsed_offset + (time.time() - started) / 60,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        loaders = make_loaders(cfg, device, smoke=True)
        model = build_model(len(cfg["classes"]), device)
        videos, labels, meta = next(iter(loaders["train"]))
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(videos.to(device))
        print(json.dumps({"shape": list(output.shape), "label": labels.tolist(), "sample_id": meta["sample_id"][0], "device": str(device)}, ensure_ascii=False))
        return
    results_root = Path(cfg["project_root"]) / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    log_stream = (results_root / "console.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)
    print(f"=== holdout run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    run(cfg, device)


if __name__ == "__main__":
    main()
