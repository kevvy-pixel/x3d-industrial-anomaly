from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop


MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


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


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resize_short_side(video: torch.Tensor, target: int) -> torch.Tensor:
    _, _, height, width = video.shape
    if height < width:
        new_h = target
        new_w = round(width * target / height)
    else:
        new_w = target
        new_h = round(height * target / width)
    frames = video.permute(1, 0, 2, 3)
    frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return frames.permute(1, 0, 2, 3)


def spatial_transform(video: torch.Tensor, train: bool, cfg: dict) -> torch.Tensor:
    crop_size = int(cfg["model"]["crop_size"])
    if train:
        low, high = cfg["model"]["resize_short_train"]
        target = random.randint(int(low), int(high))
    else:
        target = int(cfg["model"]["resize_short_eval"])
    video = resize_short_side(video, target)
    if train:
        i, j, h, w = RandomCrop.get_params(video, (crop_size, crop_size))
        video = video[:, :, i : i + h, j : j + w]
        if random.random() < 0.5:
            video = torch.flip(video, dims=[3])
    else:
        height, width = video.shape[-2:]
        top = max(0, (height - crop_size) // 2)
        left = max(0, (width - crop_size) // 2)
        video = video[:, :, top : top + crop_size, left : left + crop_size]
    return (video - MEAN) / STD


def decode_video(path: str) -> torch.Tensor:
    frames = []
    with av.open(path, mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return torch.stack(frames, dim=0)


def temporal_indices(total: int, frames: int, rate: int, train: bool, view: int = 0, views: int = 1) -> torch.Tensor:
    span = (frames - 1) * rate + 1
    max_start = max(0, total - span)
    if train:
        start = random.randint(0, max_start) if max_start else 0
    elif views > 1:
        start = round(max_start * view / (views - 1))
    else:
        start = max_start // 2
    indices = start + torch.arange(frames) * rate
    return indices.clamp(max=total - 1).long()


class VideoDataset(Dataset):
    def __init__(self, manifest: Path, cfg: dict, train: bool):
        self.cfg = cfg
        self.train = train
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        decoded = decode_video(row["video_path"])
        indices = temporal_indices(
            len(decoded),
            int(self.cfg["model"]["input_frames"]),
            int(self.cfg["model"]["sampling_rate"]),
            self.train,
        )
        video = decoded[indices].permute(3, 0, 1, 2).float().div_(255.0)
        return spatial_transform(video, self.train, self.cfg), int(row["label"]), row["video_path"]


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = x3d_m(pretrained=True)
    head = model.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    head.activation = None
    return model.to(device)


def configure_phase(model: nn.Module, phase: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if phase == "head":
        modules = [model.blocks[-1]]
    elif phase == "finetune":
        modules = [model.blocks[-2], model.blocks[-1]]
    else:
        raise ValueError(phase)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def set_phase_mode(model: nn.Module, phase: str) -> None:
    model.eval()
    if phase == "head":
        model.blocks[-1].train()
    else:
        model.blocks[-2].train()
        model.blocks[-1].train()


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def train_epoch(model, loader, optimizer, scaler, criterion, device, phase, accumulation_steps) -> EpochResult:
    set_phase_mode(model, phase)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    correct = 0
    count = 0
    for step, (videos, labels, _paths) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        amp_context = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
        with amp_context():
            logits = model(videos)
            loss = criterion(logits, labels) / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch = labels.size(0)
        total_loss += loss.item() * accumulation_steps * batch
        correct += (logits.argmax(1) == labels).sum().item()
        count += batch
    return EpochResult(total_loss / count, correct / count)


@torch.inference_mode()
def evaluate(model, loader, criterion, device) -> tuple[EpochResult, list[int], list[int], list[str]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    targets, predictions, paths = [], [], []
    for videos, labels, batch_paths in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(videos)
            loss = criterion(logits, labels)
        preds = logits.argmax(1)
        batch = labels.size(0)
        total_loss += loss.item() * batch
        correct += (preds == labels).sum().item()
        count += batch
        targets.extend(labels.cpu().tolist())
        predictions.extend(preds.cpu().tolist())
        paths.extend(batch_paths)
    return EpochResult(total_loss / count, correct / count), targets, predictions, paths


def save_checkpoint(path: Path, model, optimizer, epoch: int, phase: str, val_accuracy: float, cfg: dict) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "phase": phase,
            "val_accuracy": val_accuracy,
            "classes": cfg["classes"],
            "config": cfg,
        },
        path,
    )


def plot_history(history: list[dict], path: Path) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    seed_everything(int(cfg["seed"]))
    root = Path(cfg["project_root"])
    results = root / "results"
    checkpoints = root / "checkpoints"
    results.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_root = Path(cfg["data"]["manifest_root"])
    datasets = {
        split: VideoDataset(manifest_root / f"{split}.csv", cfg, train=(split == "train"))
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    common = dict(
        batch_size=1 if args.smoke_test else int(cfg["training"]["batch_size"]),
        num_workers=0 if args.smoke_test else int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    model = build_model(len(cfg["classes"]), device)
    if args.smoke_test:
        videos, labels, paths = next(iter(loaders["train"]))
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(videos.to(device))
        print(json.dumps({"shape": list(output.shape), "label": labels.tolist(), "path": paths[0]}))
        return

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"]["mixed_precision"]) and device.type == "cuda")
    phases = [
        ("head", int(cfg["training"]["head_epochs"]), float(cfg["training"]["head_lr"])),
        ("finetune", int(cfg["training"]["finetune_epochs"]), float(cfg["training"]["finetune_lr"])),
    ]
    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    global_epoch = 0
    stale_epochs = 0
    started = time.time()
    for phase, epochs, lr in phases:
        stale_epochs = 0
        parameters = configure_phase(model, phase)
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=float(cfg["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        for _ in range(epochs):
            global_epoch += 1
            train_result = train_epoch(
                model, loaders["train"], optimizer, scaler, criterion, device, phase,
                int(cfg["training"]["accumulation_steps"]),
            )
            val_result, _, _, _ = evaluate(model, loaders["val"], criterion, device)
            row = {
                "epoch": global_epoch,
                "phase": phase,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
                "elapsed_minutes": (time.time() - started) / 60,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(results / "training_history.csv", index=False)
            plot_history(history, results / "training_curves.png")
            save_checkpoint(checkpoints / "last.pt", model, optimizer, global_epoch, phase, val_result.accuracy, cfg)
            if val_result.accuracy > best_accuracy:
                best_accuracy = val_result.accuracy
                best_epoch = global_epoch
                stale_epochs = 0
                save_checkpoint(checkpoints / "best.pt", model, optimizer, global_epoch, phase, val_result.accuracy, cfg)
            else:
                stale_epochs += 1
            print(json.dumps(row), flush=True)
            scheduler.step()
            if stale_epochs >= int(cfg["training"]["early_stopping_patience"]):
                print(f"Early stopping after epoch {global_epoch}", flush=True)
                break

    checkpoint = torch.load(checkpoints / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_result, targets, predictions, paths = evaluate(model, loaders["test"], criterion, device)
    labels = list(range(len(cfg["classes"])))
    report = classification_report(
        targets, predictions, labels=labels, target_names=cfg["classes"], output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    pd.DataFrame(report).transpose().to_csv(results / "classification_report.csv")
    pd.DataFrame(
        {"video_path": paths, "target": targets, "prediction": predictions}
    ).to_csv(results / "test_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=cfg["classes"], yticklabels=cfg["classes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(results / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "elapsed_minutes": (time.time() - started) / 60,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "classes": cfg["classes"],
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
