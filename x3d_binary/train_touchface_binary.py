from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PRIVATE_PACKAGES = Path(r"D:\CondaData\factory_x3d_m\packages")
if PRIVATE_PACKAGES.exists():
    sys.path.append(str(PRIVATE_PACKAGES))

CLASSES = ("normal", "touch_face")
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ClipDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows, self.train = rows, train

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def decode_video(row: dict) -> np.ndarray:
        capture = cv2.VideoCapture(row["video_path"])
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {row['video_path']}")
        fps, frame_count = float(row["fps"]), int(row["frame_count"])
        indices = np.linspace(float(row["start_sec"]) * fps, float(row["end_sec"]) * fps, 16, endpoint=False)
        indices = np.clip(np.rint(indices).astype(int), 0, max(0, frame_count - 1))
        wanted = defaultdict(list)
        for output_index, frame_index in enumerate(indices.tolist()):
            wanted[int(frame_index)].append(output_index)
        frames: list[np.ndarray | None] = [None] * 16
        first, last = int(indices.min()), int(indices.max())
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        last_good = None
        for frame_index in range(first, last + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                size = (round(width * 256 / height), 256) if height < width else (256, round(height * 256 / width))
                last_good = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                for output_index in wanted[frame_index]:
                    frames[output_index] = last_good
        capture.release()
        if last_good is None:
            raise RuntimeError("no frames decoded")
        for index in range(16):
            if frames[index] is None:
                frames[index] = frames[index - 1] if index else last_good
        return np.stack(frames)

    def __getitem__(self, index: int):
        row = self.rows[index]
        try:
            array = self.decode_video(row) if self.train else np.load(row["cache_path"], allow_pickle=False)
            clip = torch.from_numpy(np.ascontiguousarray(array)).permute(3, 0, 1, 2).float().div_(255)
            height, width = clip.shape[-2:]
            top = random.randint(0, max(0, height - 224)) if self.train else max(0, (height - 224) // 2)
            left = random.randint(0, max(0, width - 224)) if self.train else max(0, (width - 224) // 2)
            clip = clip[:, :, top : top + 224, left : left + 224]
            if self.train and random.random() < 0.5:
                clip = torch.flip(clip, dims=[3])
            clip = (clip - MEAN) / STD
            return clip, int(row["label_idx"]), row["sample_id"]
        except Exception as exc:
            raise RuntimeError(f"sample_id={row['sample_id']}: {exc}") from exc


class BalancedCycleSampler(Sampler[int]):
    def __init__(self, rows: list[dict], seed: int):
        self.by_class = defaultdict(list)
        for index, row in enumerate(rows):
            self.by_class[int(row["label_idx"])].append(index)
        if set(self.by_class) != {0, 1}:
            raise RuntimeError("both binary classes are required")
        self.quota = min(len(items) for items in self.by_class.values())
        self.seed, self.epoch = seed, 0

    def __len__(self) -> int:
        return self.quota * 2

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        selected = []
        for label in (0, 1):
            pool = self.by_class[label]
            cycle = (self.epoch * self.quota) // len(pool)
            offset = (self.epoch * self.quota) % len(pool)
            rng = random.Random(self.seed + label * 100003 + cycle)
            order = pool.copy()
            rng.shuffle(order)
            take = order[offset : offset + self.quota]
            if len(take) < self.quota:
                rng = random.Random(self.seed + label * 100003 + cycle + 1)
                next_order = pool.copy()
                rng.shuffle(next_order)
                take += next_order[: self.quota - len(take)]
            selected.extend(take)
        random.Random(self.seed + self.epoch).shuffle(selected)
        return iter(selected)


def build_binary_model(backbone_checkpoint: Path) -> nn.Module:
    torch.hub.set_dir(r"D:\CondaData\factory_x3d_m\torch_cache\hub")
    from pytorchvideo.models.hub import x3d_m

    model = x3d_m(pretrained=False)
    old = model.blocks[-1].proj
    model.blocks[-1].proj = nn.Linear(old.in_features, 2)
    source = torch.load(backbone_checkpoint, map_location="cpu")
    source_state = source["model"]
    target_state = model.state_dict()
    transferable = {key: value for key, value in source_state.items() if key in target_state and target_state[key].shape == value.shape}
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    allowed = {"blocks.5.proj.weight", "blocks.5.proj.bias"}
    if set(missing) != allowed or unexpected:
        raise RuntimeError(f"backbone transfer mismatch missing={missing}, unexpected={unexpected}")
    nn.init.normal_(model.blocks[-1].proj.weight, 0, 0.01)
    nn.init.zeros_(model.blocks[-1].proj.bias)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[dict, list[dict]]:
    model.eval()
    losses, labels, predictions, probabilities, sample_ids = [], [], [], [], []
    for clips, targets, ids in tqdm(loader, desc="val", leave=False):
        clips, targets = clips.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(clips)
            loss = criterion(logits, targets)
        probs = torch.softmax(logits.float(), 1)[:, 1]
        losses.append(loss.item() * targets.size(0))
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        sample_ids.extend(ids)
    precision, recall, per_f1, support = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    metrics = {
        "val_loss": sum(losses) / len(labels), "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities), "pr_auc": average_precision_score(labels, probabilities),
        "per_class": {CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(per_f1[i]), "support": int(support[i])} for i in range(2)},
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    rows = [{"sample_id": sid, "truth": CLASSES[y], "prediction": CLASSES[p], "touch_face_probability": prob} for sid, y, p, prob in zip(sample_ids, labels, predictions, probabilities)]
    return metrics, rows


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, config: dict) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "metrics": metrics, "classes": CLASSES, "config": config}, path)


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def append_report(path: Path, config: dict, best: dict) -> None:
    lines = ["", "## 训练结果", "", f"- 实际完成 {best['completed_epochs']} 个 epoch；最佳 epoch：{best['epoch']}。",
             f"- 最佳验证 loss：{best['val_loss']:.6f}；accuracy：{best['accuracy']:.4f}；macro-F1：{best['macro_f1']:.4f}。",
             f"- ROC-AUC：{best['roc_auc']:.4f}；PR-AUC：{best['pr_auc']:.4f}。", "", "| 类别 | precision | recall | F1 | support |", "|---|---:|---:|---:|---:|"]
    for name in CLASSES:
        item = best["per_class"][name]
        lines.append(f"| {name} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |")
    matrix = best["confusion_matrix"]
    lines += ["", "### 混淆矩阵（行是真值，列是预测）", "", "| true\\pred | normal | touch_face |", "|---|---:|---:|", f"| normal | {matrix[0][0]} | {matrix[0][1]} |", f"| touch_face | {matrix[1][0]} | {matrix[1][1]} |", "", "### 超参数", "", "```json", json.dumps(config, ensure_ascii=False, indent=2), "```"]
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    seed_all(args.seed)
    all_rows = read_rows(args.experiment / "samples.csv")
    train_rows = [row for row in all_rows if row["split"] == "train"]
    val_rows = [row for row in all_rows if row["split"] == "val"]
    overlap = {row["source_id"] for row in train_rows} & {row["source_id"] for row in val_rows}
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE: {len(overlap)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    device = torch.device("cuda")
    config = vars(args).copy()
    config.update(experiment=str(args.experiment), backbone=str(args.backbone), classes=list(CLASSES), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), selection_metric="macro_f1")
    (args.experiment / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    sampler = BalancedCycleSampler(train_rows, args.seed)
    train_loader = DataLoader(ClipDataset(train_rows, True), batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(ClipDataset(val_rows, False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    model = build_binary_model(args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    history_path = args.experiment / "history.csv"
    tensorboard = SummaryWriter(args.experiment / "tensorboard")
    best_f1, best_metrics, best_predictions = -1.0, None, None

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True)
        running, seen, started = 0.0, 0, time.time()
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, (clips, targets, _) in enumerate(bar):
            clips, targets = clips.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss = criterion(model(clips), targets) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running += loss.item() * args.grad_accum * targets.size(0); seen += targets.size(0)
            bar.set_postfix(loss=f"{running / seen:.4f}")
        metrics, predictions = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        metrics.update(epoch=epoch + 1, train_loss=running / seen, lr=optimizer.param_groups[0]["lr"], seconds=time.time() - started)
        flat = {key: value for key, value in metrics.items() if key not in {"per_class", "confusion_matrix"}}
        with history_path.open("a", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat))
            if stream.tell() == 0: writer.writeheader()
            writer.writerow(flat)
        tensorboard.add_scalars("loss", {"train": metrics["train_loss"], "val": metrics["val_loss"]}, epoch + 1)
        tensorboard.add_scalar("val/macro_f1", metrics["macro_f1"], epoch + 1)
        save_checkpoint(args.experiment / "last.pt", model, optimizer, scheduler, epoch, metrics, config)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_metrics, best_predictions = metrics["macro_f1"], metrics, predictions
            save_checkpoint(args.experiment / "best.pt", model, optimizer, scheduler, epoch, metrics, config)
            (args.experiment / "best_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            write_predictions(args.experiment / "best_predictions.csv", predictions)
        print(json.dumps(flat, ensure_ascii=False), flush=True)
    tensorboard.close()
    assert best_metrics is not None
    best_metrics["completed_epochs"] = args.epochs
    (args.experiment / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    append_report(args.experiment / "EXPERIMENT.md", config, best_metrics)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
