from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Project-private packages are appended so the CUDA environment's torch/numpy stay authoritative.
PRIVATE_PACKAGES = Path(__file__).resolve().parent / "packages"
if PRIVATE_PACKAGES.exists():
    sys.path.append(str(PRIVATE_PACKAGES))


CLASSES = ("normal", "single_hand", "board_drop", "touch_face")
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ClipDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, size: int = 224):
        self.rows = rows
        self.train = train
        self.size = size
        # Each persistent validation worker sees ~500 samples. Cache resized uint8
        # clips so epochs 2+ avoid decoding the identical fixed validation set.
        self.cache_limit = 600 if not train else 0
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _decode(row: dict) -> torch.Tensor:
        path = row["video_path"]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(row["fps"])
        frame_count = int(row["frame_count"])
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        indices = np.linspace(start * fps, end * fps, 16, endpoint=False)
        indices = np.clip(np.rint(indices).astype(int), 0, max(0, frame_count - 1))
        wanted = defaultdict(list)
        for out_idx, frame_idx in enumerate(indices.tolist()):
            wanted[frame_idx].append(out_idx)
        frames: list[np.ndarray | None] = [None] * 16
        first, last = int(indices.min()), int(indices.max())
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        last_good = None
        for frame_idx in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if h < w:
                    new_h, new_w = 256, round(w * 256 / h)
                else:
                    new_h, new_w = round(h * 256 / w), 256
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                last_good = rgb
                for out_idx in wanted[frame_idx]:
                    frames[out_idx] = rgb
        cap.release()
        if last_good is None:
            raise RuntimeError(f"No frames decoded: {path} [{start}, {end}]")
        for i in range(16):
            if frames[i] is None:
                frames[i] = frames[i - 1] if i else last_good
        array = np.stack(frames).copy()
        return torch.from_numpy(array).permute(3, 0, 1, 2)

    def _transform(self, clip: torch.Tensor) -> torch.Tensor:
        clip = clip.float().div_(255.0)
        _, _, h, w = clip.shape
        bt = clip.permute(1, 0, 2, 3)
        new_h, new_w = h, w
        if self.train:
            top = random.randint(0, max(0, new_h - self.size))
            left = random.randint(0, max(0, new_w - self.size))
        else:
            top = max(0, (new_h - self.size) // 2)
            left = max(0, (new_w - self.size) // 2)
        bt = bt[:, :, top:top + self.size, left:left + self.size]
        if self.train and random.random() < 0.5:
            bt = torch.flip(bt, dims=[3])
        clip = bt.permute(1, 0, 2, 3)
        return (clip - MEAN) / STD

    def __getitem__(self, index: int):
        row = self.rows[index]
        try:
            sample_id = row["sample_id"]
            raw = self.cache.get(sample_id)
            if raw is None:
                raw = self._decode(row)
                if self.cache_limit:
                    self.cache[sample_id] = raw
                    if len(self.cache) > self.cache_limit:
                        self.cache.popitem(last=False)
            clip = self._transform(raw)
        except Exception as exc:
            raise RuntimeError(f"sample_id={row['sample_id']}: {exc}") from exc
        return clip, int(row["label_idx"]), row["sample_id"]


class BalancedSampler(Sampler[int]):
    """Exactly equal, without-replacement draws per class each epoch."""
    def __init__(self, rows: list[dict], seed: int):
        self.by_class = defaultdict(list)
        for idx, row in enumerate(rows):
            self.by_class[int(row["label_idx"])].append(idx)
        if set(self.by_class) != set(range(len(CLASSES))):
            raise ValueError(f"Training set is missing classes: {set(range(len(CLASSES))) - set(self.by_class)}")
        self.quota = min(map(len, self.by_class.values()))
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.quota * len(CLASSES)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        result = []
        for label in range(len(CLASSES)):
            pool = self.by_class[label]
            result.extend(rng.sample(pool, self.quota))
        rng.shuffle(result)
        return iter(result)


def build_model(pretrained: bool = True) -> nn.Module:
    torch.hub.set_dir(str(Path(__file__).resolve().parent / "torch_cache" / "hub"))
    from pytorchvideo.models.hub import x3d_m
    model = x3d_m(pretrained=pretrained)
    old = model.blocks[-1].proj
    model.blocks[-1].proj = nn.Linear(old.in_features, len(CLASSES), bias=True)
    nn.init.normal_(model.blocks[-1].proj.weight, 0, 0.01)
    nn.init.zeros_(model.blocks[-1].proj.bias)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    losses, labels, preds = [], [], []
    for clips, target, _ in tqdm(loader, desc="val", leave=False):
        clips, target = clips.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(clips)
            loss = criterion(logits, target)
        losses.append(loss.item() * target.size(0))
        labels.extend(target.cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
    precision, recall, per_f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(4)), zero_division=0)
    return {
        "val_loss": sum(losses) / len(labels), "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "per_class": {CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                        "f1": float(per_f1[i]), "support": int(support[i])} for i in range(4)},
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(4))).tolist(),
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, config: dict) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "metrics": metrics, "classes": CLASSES, "config": config}, path)


def append_report(path: Path, config: dict, best: dict) -> None:
    cm = best["confusion_matrix"]
    lines = ["", "## 实际训练结果", "", f"- 完成轮数：{config['epochs']}；batch size：{config['batch_size']}；梯度累积：{config['grad_accum']}。",
             f"- 初始学习率：{config['lr']}；weight decay：{config['weight_decay']}。",
             f"- 最佳 epoch：{best['epoch']}；验证 loss：{best['val_loss']:.6f}；accuracy：{best['accuracy']:.4f}；macro-F1：{best['macro_f1']:.4f}。",
             "", "### 最佳模型逐类指标", "", "| 类别 | precision | recall | F1 | support |", "|---|---:|---:|---:|---:|"]
    for c in CLASSES:
        m = best["per_class"][c]
        lines.append(f"| {c} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    lines += ["", "### 混淆矩阵（行是真值，列是预测）", "", "| true\\pred | " + " | ".join(CLASSES) + " |",
              "|---|" + "---:|" * 4]
    for i, c in enumerate(CLASSES):
        lines.append(f"| {c} | " + " | ".join(map(str, cm[i])) + " |")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    seed_everything(args.seed)
    rows = read_rows(args.experiment / "samples.csv")
    train_rows = [x for x in rows if x["split"] == "train"]
    val_rows = [x for x in rows if x["split"] == "val"]
    train_sources = {x["source_id"] for x in train_rows}
    val_sources = {x["source_id"] for x in val_rows}
    overlap = train_sources & val_sources
    assert not overlap, f"DATA LEAKAGE: {len(overlap)} source ids overlap"
    config = vars(args).copy()
    config["experiment"] = str(config["experiment"])
    config["resume"] = str(config["resume"]) if config["resume"] else None
    config["classes"] = CLASSES
    config["torch"] = torch.__version__
    config["cuda"] = torch.version.cuda
    config["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    (args.experiment / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    train_set, val_set = ClipDataset(train_rows, True), ClipDataset(val_rows, False)
    sampler = BalancedSampler(train_rows, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this X3D-M training run")
    model = build_model(pretrained=args.resume is None).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_f1, best_metrics = 0, -1.0, None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_f1 = float(state["metrics"]["macro_f1"])
    history_path = args.experiment / "history.csv"
    writer = SummaryWriter(args.experiment / "tensorboard")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, seen, started = 0.0, 0, time.time()
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, (clips, target, _) in enumerate(bar):
            clips, target = clips.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(clips)
                loss = criterion(logits, target) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * args.grad_accum * target.size(0)
            seen += target.size(0)
            bar.set_postfix(loss=f"{running_loss / seen:.4f}")
        train_loss = running_loss / seen
        metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        metrics.update({"epoch": epoch + 1, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"],
                        "seconds": time.time() - started})
        flat = {k: v for k, v in metrics.items() if k not in {"per_class", "confusion_matrix"}}
        write_header = not history_path.exists()
        with history_path.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat))
            if write_header: w.writeheader()
            w.writerow(flat)
        writer.add_scalars("loss", {"train": train_loss, "val": metrics["val_loss"]}, epoch + 1)
        writer.add_scalar("val/macro_f1", metrics["macro_f1"], epoch + 1)
        save_checkpoint(args.experiment / "last.pt", model, optimizer, scheduler, epoch, metrics, config)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_metrics = metrics["macro_f1"], metrics
            save_checkpoint(args.experiment / "best.pt", model, optimizer, scheduler, epoch, metrics, config)
            (args.experiment / "best_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(flat, ensure_ascii=False))
    writer.close()
    if best_metrics is None:
        best_metrics = torch.load(args.experiment / "best.pt", map_location="cpu", weights_only=False)["metrics"]
    append_report(args.experiment / "EXPERIMENT.md", config, best_metrics)


if __name__ == "__main__":
    # Required for Windows DataLoader workers.
    torch.multiprocessing.freeze_support()
    main()
