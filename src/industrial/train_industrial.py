from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorchvideo.models.hub import x3d_m
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import RandomCrop


MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def resize_short_side(video: torch.Tensor, target: int) -> torch.Tensor:
    _, _, height, width = video.shape
    if height < width:
        new_h, new_w = target, round(width * target / height)
    else:
        new_h, new_w = round(height * target / width), target
    frames = F.interpolate(video.permute(1, 0, 2, 3), size=(new_h, new_w), mode="bilinear", align_corners=False)
    return frames.permute(1, 0, 2, 3)


def spatial_transform(video: torch.Tensor, train: bool, cfg: dict) -> torch.Tensor:
    crop_size = int(cfg["model"]["crop_size"])
    if train:
        low, high = cfg["model"]["resize_short_train"]
        video = resize_short_side(video, random.randint(int(low), int(high)))
        i, j, h, w = RandomCrop.get_params(video, (crop_size, crop_size))
        video = video[:, :, i:i + h, j:j + w]
        if random.random() < 0.5:
            video = torch.flip(video, dims=[3])
    else:
        video = resize_short_side(video, int(cfg["model"]["resize_short_eval"]))
        height, width = video.shape[-2:]
        top, left = max(0, (height - crop_size) // 2), max(0, (width - crop_size) // 2)
        video = video[:, :, top:top + crop_size, left:left + crop_size]
    return (video - MEAN) / STD


def read_normalized_roi(path: str) -> tuple[float, float, float, float]:
    values = Path(path).read_text(encoding="utf-8-sig").strip().split()
    if len(values) != 5:
        raise RuntimeError(f"ROI must contain class_id cx cy width height: {path}")
    _, cx, cy, width, height = values
    return float(cx), float(cy), float(width), float(height)


def roi_transform(
    video: torch.Tensor,
    roi: tuple[float, float, float, float],
    train: bool,
    cfg: dict,
) -> torch.Tensor:
    """Crop one temporally fixed ROI, square it with real pixels, then resize."""
    _, _, height, width = video.shape
    cx_n, cy_n, width_n, height_n = roi
    center_x, center_y = cx_n * width, cy_n * height
    side = max(width_n * width, height_n * height)
    expand_ratio = float(cfg["model"].get("roi_expand_ratio", 0.08))
    side *= 1.0 + 2.0 * expand_ratio

    if train:
        shift_ratio = float(cfg["model"].get("roi_train_shift_ratio", 0.03))
        scale_low, scale_high = cfg["model"].get("roi_train_scale", [0.97, 1.03])
        center_x += random.uniform(-shift_ratio, shift_ratio) * side
        center_y += random.uniform(-shift_ratio, shift_ratio) * side
        side *= random.uniform(float(scale_low), float(scale_high))

    side_px = max(2, int(math.ceil(side)))
    left = int(round(center_x - side_px / 2))
    top = int(round(center_y - side_px / 2))
    right, bottom = left + side_px, top + side_px
    crop_left, crop_top = max(0, left), max(0, top)
    crop_right, crop_bottom = min(width, right), min(height, bottom)
    cropped = video[:, :, crop_top:crop_bottom, crop_left:crop_right]
    padding = (max(0, -left), max(0, right - width), max(0, -top), max(0, bottom - height))
    if any(padding):
        cropped = F.pad(cropped, padding, mode="replicate")

    crop_size = int(cfg["model"]["crop_size"])
    frames = F.interpolate(
        cropped.permute(1, 0, 2, 3),
        size=(crop_size, crop_size),
        mode="bilinear",
        align_corners=False,
    )
    cropped = frames.permute(1, 0, 2, 3)
    if train and bool(cfg["model"].get("horizontal_flip", True)) and random.random() < 0.5:
        cropped = torch.flip(cropped, dims=[3])
    return (cropped - MEAN) / STD


def decode_window(path: str, start: float, end: float, num_frames: int) -> torch.Tensor:
    """Seek near a clip and retain only the frames needed by the model."""
    targets = np.linspace(start, end, num_frames, endpoint=False)
    selected = []
    with av.open(path, mode="r") as container:
        stream = container.streams.video[0]
        # Repeated AUTO-threaded decoder creation can deadlock on Windows after
        # many short seeks. One decoder thread is fast enough for 2.67 s clips
        # and stays stable across long cross-validation runs.
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        time_base = float(stream.time_base)
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        seek_time = max(0.0, start - 1.0) + origin
        container.seek(int(seek_time / time_base), stream=stream, any_frame=False, backward=True)
        previous_frame, previous_time = None, None
        target_index = 0
        for frame in container.decode(stream):
            if frame.pts is not None:
                timestamp = float(frame.pts * stream.time_base) - origin
            elif previous_time is not None:
                timestamp = previous_time + 1.0 / fps
            else:
                timestamp = seek_time - origin
            current_frame = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            while target_index < num_frames and targets[target_index] <= timestamp:
                target = targets[target_index]
                if previous_frame is not None and abs(previous_time - target) <= abs(timestamp - target):
                    selected.append(previous_frame)
                else:
                    selected.append(current_frame)
                target_index += 1
            previous_frame, previous_time = current_frame, timestamp
            if target_index == num_frames:
                break
            if timestamp > end + 1.0:
                break
        if previous_frame is not None:
            while len(selected) < num_frames:
                selected.append(previous_frame)
    if len(selected) != num_frames:
        raise RuntimeError(f"Decoded {len(selected)}/{num_frames} frames from {path} at {start:.3f}-{end:.3f}s")
    return torch.stack(selected)


class WindowDataset(Dataset):
    def __init__(self, manifest: Path, cfg: dict, train: bool):
        self.cfg, self.train = cfg, train
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.labels = [int(row["label"]) for row in self.rows]
        self.input_mode = cfg["model"].get("input_mode", "full_frame")
        self.rois = {}
        if self.input_mode == "roi":
            for row in self.rows:
                roi_path = row.get("roi_path", "")
                if not roi_path:
                    raise RuntimeError(f"Missing ROI path for {row['sample_id']}")
                if roi_path not in self.rois:
                    self.rois[roi_path] = read_normalized_roi(roi_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        duration = float(row["video_duration"])
        configured_duration = min(duration, float(self.cfg["model"]["clip_duration_sec"]))
        sample_origin = row.get("sample_origin", "anomaly_event")
        # Only dedicated normal videos may be sampled dynamically across their
        # full duration. Hard negatives from anomaly videos must keep their
        # audited fixed window or they could jump back into an anomaly interval.
        if self.train and sample_origin == "normal_source":
            start = random.uniform(0, max(0.0, duration - configured_duration))
            end = min(duration, start + configured_duration)
        else:
            start = float(row["window_start"])
            end = float(row["window_end"])
            window_duration = end - start
            if self.train and sample_origin == "anomaly_event":
                jitter = float(self.cfg["model"]["train_temporal_jitter_sec"])
                start += random.uniform(-jitter, jitter)
                start = max(0.0, min(start, duration - window_duration))
                end = min(duration, start + window_duration)
        if not 0 <= start < end <= duration + 1e-3:
            raise RuntimeError(f"Invalid decoded interval for {row['sample_id']}: {start}-{end}/{duration}")
        decoded = decode_window(row["video_path"], start, end, int(self.cfg["model"]["input_frames"]))
        video = decoded.permute(3, 0, 1, 2).float().div_(255.0)
        if self.input_mode == "roi":
            video = roi_transform(video, self.rois[row["roi_path"]], self.train, self.cfg)
        else:
            video = spatial_transform(video, self.train, self.cfg)
        meta = {key: row[key] for key in ("sample_id", "source_id", "video_path", "sample_origin", "window_start", "window_end")}
        return video, int(row["label"]), meta


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = x3d_m(pretrained=True)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
    model.blocks[-1].activation = None
    return model.to(device)


def configure_phase(model: nn.Module, phase: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [model.blocks[-1]] if phase == "head" else [model.blocks[-2], model.blocks[-1]]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def set_phase_mode(model: nn.Module, phase: str) -> None:
    model.eval()
    model.blocks[-1].train()
    if phase == "finetune":
        model.blocks[-2].train()


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float


def train_epoch(model, loader, optimizer, scaler, criterion, device, phase, accumulation_steps) -> EpochResult:
    set_phase_mode(model, phase)
    optimizer.zero_grad(set_to_none=True)
    total_loss, count, targets, predictions = 0.0, 0, [], []
    for step, (videos, labels, _) in enumerate(loader):
        if step == 0:
            print(f"{phase}: first batch loaded on CPU {tuple(videos.shape)}", flush=True)
        videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if step == 0:
            print(f"{phase}: first batch transferred to {device}", flush=True)
        amp_context = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
        with amp_context():
            logits = model(videos)
            loss = criterion(logits, labels) / accumulation_steps
        if step == 0:
            print(f"{phase}: first forward pass completed", flush=True)
        scaler.scale(loss).backward()
        if step == 0:
            print(f"{phase}: first backward pass completed", flush=True)
        print(f"{phase}: batch {step + 1}/{len(loader)} backward complete", flush=True)
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch = labels.size(0)
        total_loss += loss.item() * accumulation_steps * batch
        count += batch
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return EpochResult(total_loss / count, accuracy_score(targets, predictions), f1_score(targets, predictions, average="macro", zero_division=0))


@torch.inference_mode()
def evaluate(model, loader, criterion, device) -> tuple[EpochResult, list[dict]]:
    model.eval()
    total_loss, count, records = 0.0, 0, []
    for videos, labels, meta in loader:
        videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(videos)
            loss = criterion(logits, labels)
        probs = logits.softmax(1).cpu().numpy()
        preds = logits.argmax(1).cpu().tolist()
        targets = labels.cpu().tolist()
        total_loss += loss.item() * len(targets)
        count += len(targets)
        for i, (target, prediction) in enumerate(zip(targets, preds)):
            record = {key: meta[key][i] for key in meta}
            record.update({"target": target, "prediction": prediction})
            record.update({f"prob_{j}": float(value) for j, value in enumerate(probs[i])})
            records.append(record)
    targets = [row["target"] for row in records]
    predictions = [row["prediction"] for row in records]
    result = EpochResult(total_loss / count, accuracy_score(targets, predictions), f1_score(targets, predictions, average="macro", zero_division=0))
    return result, records


def save_checkpoint(path: Path, model, epoch: int, phase: str, score: float, cfg: dict, optimizer=None, scheduler=None, scaler=None) -> None:
    payload = {"model_state": model.state_dict(), "epoch": epoch, "phase": phase, "score": score, "classes": cfg["classes"], "config": cfg}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, path)


def plot_history(history: list[dict], path: Path) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label="train")
    axes[0].plot(frame.epoch, frame.val_loss, label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(frame.epoch, frame.train_macro_f1, label="train")
    axes[1].plot(frame.epoch, frame.val_macro_f1, label="val")
    axes[1].set_title("Macro F1"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def loaders_for_fold(cfg: dict, fold: int, device: torch.device, smoke: bool = False):
    root = Path(cfg["data"]["manifest_root"]) / f"fold_{fold}"
    datasets = {split: WindowDataset(root / f"{split}.csv", cfg, split == "train") for split in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + fold)
    common = dict(batch_size=1 if smoke else int(cfg["training"]["batch_size"]), num_workers=0 if smoke else int(cfg["training"]["num_workers"]), pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)
    counts = np.bincount(datasets["train"].labels, minlength=len(cfg["classes"]))
    weights = [1.0 / counts[label] for label in datasets["train"].labels]
    sampler = WeightedRandomSampler(weights, num_samples=int(cfg["training"]["balanced_samples_per_epoch"]), replacement=True, generator=generator)
    return {
        "train": DataLoader(datasets["train"], sampler=sampler, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def run_fold(cfg: dict, fold: int, device: torch.device, progress_path: Path) -> dict:
    seed_everything(int(cfg["seed"]) + fold)
    fold_root = Path(cfg["project_root"]) / "results" / f"fold_{fold}"
    checkpoint_root = Path(cfg["project_root"]) / "checkpoints" / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    loaders = loaders_for_fold(cfg, fold, device)
    model = build_model(len(cfg["classes"]), device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"]["mixed_precision"]) and device.type == "cuda")
    phases = [("head", int(cfg["training"]["head_epochs"]), float(cfg["training"]["head_lr"])), ("finetune", int(cfg["training"]["finetune_epochs"]), float(cfg["training"]["finetune_lr"]))]
    history_path = fold_root / "training_history.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    best_f1, best_accuracy, best_loss, best_epoch, global_epoch = -1.0, -1.0, float("inf"), 0, 0
    for old_row in history:
        old_f1, old_accuracy, old_loss = float(old_row["val_macro_f1"]), float(old_row["val_accuracy"]), float(old_row["val_loss"])
        improved = old_f1 > best_f1 or (math.isclose(old_f1, best_f1) and (old_accuracy > best_accuracy or (math.isclose(old_accuracy, best_accuracy) and old_loss < best_loss)))
        if improved:
            best_f1, best_accuracy, best_loss, best_epoch = old_f1, old_accuracy, old_loss, int(old_row["epoch"])
        global_epoch = max(global_epoch, int(old_row["epoch"]))
    last_checkpoint_path = checkpoint_root / "last.pt"
    resume_checkpoint = None
    if history and last_checkpoint_path.exists():
        resume_checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state"])
        print(f"Fold {fold}: resuming after epoch {global_epoch} from {last_checkpoint_path}", flush=True)
    elapsed_offset = max((float(row["elapsed_minutes"]) for row in history), default=0.0)
    started = time.time()
    for phase, epochs, lr in phases:
        completed_in_phase = sum(str(row["phase"]) == phase for row in history)
        if completed_in_phase >= epochs:
            continue
        stale = 0
        resume_lr = lr * 0.5 * (1.0 + math.cos(math.pi * completed_in_phase / epochs))
        optimizer = torch.optim.AdamW(configure_phase(model, phase), lr=resume_lr, weight_decay=float(cfg["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - completed_in_phase))
        if resume_checkpoint and resume_checkpoint.get("phase") == phase and "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            if "scheduler_state" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            if "scaler_state" in resume_checkpoint:
                scaler.load_state_dict(resume_checkpoint["scaler_state"])
        for _ in range(epochs - completed_in_phase):
            global_epoch += 1
            train_result = train_epoch(model, loaders["train"], optimizer, scaler, criterion, device, phase, int(cfg["training"]["accumulation_steps"]))
            val_result, _ = evaluate(model, loaders["val"], criterion, device)
            row = {"fold": fold, "epoch": global_epoch, "phase": phase, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_result.loss, "train_accuracy": train_result.accuracy, "train_macro_f1": train_result.macro_f1, "val_loss": val_result.loss, "val_accuracy": val_result.accuracy, "val_macro_f1": val_result.macro_f1, "elapsed_minutes": elapsed_offset + (time.time() - started) / 60}
            history.append(row)
            pd.DataFrame(history).to_csv(fold_root / "training_history.csv", index=False)
            combined = []
            if progress_path.exists():
                try: combined = pd.read_csv(progress_path).to_dict("records")
                except pd.errors.EmptyDataError: pass
            combined = [item for item in combined if not (int(item["fold"]) == fold and int(item["epoch"]) == global_epoch)] + [row]
            pd.DataFrame(combined).sort_values(["fold", "epoch"]).to_csv(progress_path, index=False)
            plot_history(history, fold_root / "training_curves.png")
            improved = val_result.macro_f1 > best_f1 or (math.isclose(val_result.macro_f1, best_f1) and (val_result.accuracy > best_accuracy or (math.isclose(val_result.accuracy, best_accuracy) and val_result.loss < best_loss)))
            if improved:
                best_f1, best_accuracy, best_loss, best_epoch, stale = val_result.macro_f1, val_result.accuracy, val_result.loss, global_epoch, 0
                save_checkpoint(checkpoint_root / "best.pt", model, global_epoch, phase, best_f1, cfg, optimizer, scheduler, scaler)
            else:
                stale += 1
            save_checkpoint(checkpoint_root / "last.pt", model, global_epoch, phase, val_result.macro_f1, cfg, optimizer, scheduler, scaler)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            scheduler.step()
            if stale >= int(cfg["training"]["early_stopping_patience"]):
                print(f"Fold {fold}: early stopping {phase} after epoch {global_epoch}", flush=True)
                break
    checkpoint = torch.load(checkpoint_root / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_result, records = evaluate(model, loaders["test"], criterion, device)
    pd.DataFrame(records).to_csv(fold_root / "test_predictions.csv", index=False)
    targets, predictions = [r["target"] for r in records], [r["prediction"] for r in records]
    report = classification_report(targets, predictions, labels=list(range(len(cfg["classes"]))), target_names=cfg["classes"], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(fold_root / "classification_report.csv")
    summary = {"fold": fold, "best_epoch": best_epoch, "best_validation_macro_f1": best_f1, "best_validation_accuracy": best_accuracy, "best_validation_loss": best_loss, "test_loss": test_result.loss, "test_accuracy": test_result.accuracy, "test_macro_f1": test_result.macro_f1, "elapsed_minutes": (time.time() - started) / 60}
    (fold_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    del model; torch.cuda.empty_cache()
    return summary


def aggregate(cfg: dict, summaries: list[dict], results_root: Path) -> dict:
    frames = [pd.read_csv(results_root / f"fold_{fold}" / "test_predictions.csv") for fold in range(int(cfg["data"]["folds"]))]
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(results_root / "oof_test_predictions.csv", index=False)
    targets, predictions = pooled.target.tolist(), pooled.prediction.tolist()
    labels = list(range(len(cfg["classes"])))
    report = classification_report(targets, predictions, labels=labels, target_names=cfg["classes"], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(results_root / "oof_classification_report.csv")
    matrix = confusion_matrix(targets, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=cfg["classes"], yticklabels=cfg["classes"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(results_root / "oof_confusion_matrix.png", dpi=180); plt.close(fig)
    prob_cols = [f"prob_{i}" for i in labels]
    # Keep the original 36-source classification metric comparable: hard-negative
    # windows measure false alarms but are not a second label for the same source.
    source_input = pooled[pooled.sample_origin != "normal_from_anomaly"]
    source = source_input.groupby(["source_id", "target"], as_index=False)[prob_cols].mean()
    source["prediction"] = source[prob_cols].to_numpy().argmax(axis=1)
    source.to_csv(results_root / "oof_source_predictions.csv", index=False)
    fold_acc = [item["test_accuracy"] for item in summaries]
    fold_f1 = [item["test_macro_f1"] for item in summaries]
    hard_negatives = pooled[pooled.sample_origin == "normal_from_anomaly"]
    hard_negative_accuracy = float((hard_negatives.prediction == hard_negatives.target).mean()) if len(hard_negatives) else None
    result = {
        "folds": summaries,
        "fold_test_accuracy_mean": float(np.mean(fold_acc)), "fold_test_accuracy_std": float(np.std(fold_acc, ddof=1)),
        "fold_test_macro_f1_mean": float(np.mean(fold_f1)), "fold_test_macro_f1_std": float(np.std(fold_f1, ddof=1)),
        "oof_window_count": len(pooled), "oof_window_accuracy": accuracy_score(targets, predictions), "oof_window_balanced_accuracy": balanced_accuracy_score(targets, predictions), "oof_window_macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "oof_source_count": len(source), "oof_source_accuracy": accuracy_score(source.target, source.prediction), "oof_source_macro_f1": f1_score(source.target, source.prediction, average="macro", zero_division=0),
        "oof_hard_negative_count": len(hard_negatives), "oof_hard_negative_normal_recall": hard_negative_accuracy,
        "device": "cuda" if torch.cuda.is_available() else "cpu", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (results_root / "cv_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--fold", type=int)
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_root = Path(cfg["project_root"]) / "results"; results_root.mkdir(parents=True, exist_ok=True)
    if not args.smoke_test:
        log_stream = (results_root / "console.log").open("a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log_stream)
        sys.stderr = Tee(sys.__stderr__, log_stream)
        print(f"=== run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    if args.smoke_test:
        loaders = loaders_for_fold(cfg, 0, device, smoke=True); model = build_model(len(cfg["classes"]), device)
        videos, labels, meta = next(iter(loaders["train"]))
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(videos.to(device))
        print(json.dumps({"input_mode": cfg["model"].get("input_mode", "full_frame"), "shape": list(output.shape), "label": labels.tolist(), "sample_id": meta["sample_id"][0], "device": str(device)}, ensure_ascii=False)); return
    folds = [args.fold] if args.fold is not None else list(range(int(cfg["data"]["folds"])))
    progress_path = results_root / "cv_training_progress.csv"
    summaries = []
    for fold in folds:
        summary_path = results_root / f"fold_{fold}" / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8")); print(f"Fold {fold}: using completed result", flush=True)
        else:
            summary = run_fold(cfg, fold, device, progress_path)
        summaries.append(summary)
    if args.fold is None:
        print(json.dumps(aggregate(cfg, summaries, results_root), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
