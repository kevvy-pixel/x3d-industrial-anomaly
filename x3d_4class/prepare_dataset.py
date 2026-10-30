from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2


CLASSES = ("normal", "single_hand", "board_drop", "touch_face")
ALIASES = {"single_hand_or_drag": "single_hand"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"fps": 0.0, "frame_count": 0, "duration": 0.0, "width": 0, "height": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {"fps": fps, "frame_count": frames, "duration": duration, "width": width, "height": height}


def fps_profile(fps: float) -> tuple[str, float]:
    # The observed frame-rate modes are near 16, 24 and 30 fps.
    options = (("16", 16.0, 1.0), ("24", 24.0, 0.8), ("30", 30.0, 0.7))
    name, _, window = min(options, key=lambda x: abs(fps - x[1]))
    return name, window


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def make_windows(start: float, end: float, video_duration: float, window: float) -> list[tuple[float, float, str]]:
    """Non-overlapping full windows; discard remainder, except a sole 0.9-1.0s 16fps interval is padded."""
    duration = max(0.0, end - start)
    full_count = int(math.floor((duration + 1e-7) / window))
    windows = [(start + i * window, start + (i + 1) * window, "full") for i in range(full_count)]
    if not windows and duration > 0.9:
        # This only matters for ~16fps clips that are just short of one second.
        center = (start + end) / 2.0
        left = center - window / 2.0
        left = min(max(0.0, left), max(0.0, video_duration - window))
        windows.append((left, min(video_duration, left + window), "padded"))
    return windows


def find_video(data_dir: Path, category: str, file_name: str) -> Path | None:
    category_dir = data_dir / category
    matches = list(category_dir.rglob(file_name)) if category_dir.exists() else []
    if not matches:
        matches = [p for p in data_dir.rglob(file_name) if "json" not in {x.lower() for x in p.parts}]
    return matches[0] if len(matches) == 1 else None


def collect_sources(root: Path) -> tuple[list[dict], list[str]]:
    sources: list[dict] = []
    warnings: list[str] = []
    for data_idx in range(1, 5):
        data_dir = root / f"data{data_idx}"
        # Normal videos have no JSON and their entire valid duration is normal.
        normal_dir = data_dir / "normal"
        if normal_dir.exists():
            for video in sorted(p for p in normal_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS):
                meta = probe_video(video)
                if meta["fps"] <= 0 or meta["duration"] <= 0:
                    warnings.append(f"Unreadable normal video: {video}")
                    continue
                sources.append({
                    "source_id": f"data{data_idx}:{video.relative_to(data_dir).as_posix()}",
                    "data_group": f"data{data_idx}", "video_path": str(video), "meta": meta,
                    "annotations": [{"start": 0.0, "end": meta["duration"], "label": "normal", "id": 0}],
                })

        json_root = data_dir / "json"
        if not json_root.exists():
            continue
        for json_path in sorted(json_root.rglob("*.json")):
            category = json_path.parent.name
            if category.lower().replace("_", "") in {"nogloves", "noglove"}:
                continue
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                warnings.append(f"Bad JSON {json_path}: {exc}")
                continue
            file_name = payload.get("video", {}).get("file_name", json_path.stem + ".mp4")
            video = find_video(data_dir, category, file_name)
            if video is None:
                warnings.append(f"Video match missing/ambiguous: {json_path} -> {file_name}")
                continue
            meta = probe_video(video)
            if meta["fps"] <= 0 or meta["duration"] <= 0:
                warnings.append(f"Unreadable annotated video: {video}")
                continue
            annotations = []
            for ann in payload.get("annotations", []):
                label = ALIASES.get(str(ann.get("label", "")), str(ann.get("label", "")))
                if label not in CLASSES or label == "normal":
                    continue
                start = max(0.0, float(ann["start_sec"]))
                end = min(meta["duration"], float(ann["end_sec"]))
                if end > start:
                    annotations.append({"start": start, "end": end, "label": label, "id": int(ann.get("id", 0))})
            if annotations:
                sources.append({
                    "source_id": f"data{data_idx}:{video.relative_to(data_dir).as_posix()}",
                    "data_group": f"data{data_idx}", "video_path": str(video), "meta": meta,
                    "annotations": annotations, "json_path": str(json_path),
                })
    return sources, warnings


def split_sources(sources: list[dict], seed: int) -> None:
    strata: dict[str, list[dict]] = defaultdict(list)
    for source in sources:
        signature = "+".join(sorted({a["label"] for a in source["annotations"]}))
        strata[signature].append(source)
    for signature, items in sorted(strata.items()):
        random.Random(stable_seed(signature, seed)).shuffle(items)
        n_val = 0 if len(items) == 1 else max(1, round(len(items) * 0.2))
        n_val = min(n_val, len(items) - 1)
        val_ids = {x["source_id"] for x in items[:n_val]}
        for item in items:
            item["split"] = "val" if item["source_id"] in val_ids else "train"


def build_samples(sources: list[dict]) -> tuple[list[dict], Counter]:
    samples: list[dict] = []
    excluded = Counter()
    for source in sources:
        fps = source["meta"]["fps"]
        bucket, window = fps_profile(fps)
        for ann in source["annotations"]:
            ann_windows = make_windows(ann["start"], ann["end"], source["meta"]["duration"], window)
            if not ann_windows:
                excluded[f"short_{ann['label']}"] += 1
            for window_idx, (left, right, kind) in enumerate(ann_windows):
                window_duration = max(1e-9, right - left)
                ambiguous = any(
                    other["label"] != ann["label"]
                    and max(0.0, min(right, other["end"]) - max(left, other["start"])) / window_duration >= 0.5
                    for other in source["annotations"]
                )
                if ambiguous:
                    excluded[f"ambiguous_{ann['label']}"] += 1
                    continue
                samples.append({
                    "sample_id": f"{source['source_id']}#a{ann['id']}-w{window_idx}",
                    "source_id": source["source_id"], "data_group": source["data_group"],
                    "split": source["split"], "label": ann["label"], "label_idx": CLASSES.index(ann["label"]),
                    "video_path": source["video_path"], "json_path": source.get("json_path", ""),
                    "annotation_id": ann["id"], "annotation_start": round(ann["start"], 6),
                    "annotation_end": round(ann["end"], 6), "start_sec": round(left, 6),
                    "end_sec": round(right, 6), "window_kind": kind, "fps": round(fps, 6),
                    "fps_bucket": bucket, "frame_count": source["meta"]["frame_count"],
                    "width": source["meta"]["width"], "height": source["meta"]["height"],
                })
    return samples, excluded


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(out: Path, root: Path, sources: list[dict], samples: list[dict], excluded: Counter, warnings: list[str], seed: int) -> None:
    sample_counts = Counter((x["split"], x["label"]) for x in samples)
    source_counts = Counter((x["split"], a["label"]) for x in sources for a in x["annotations"])
    split_source_counts = Counter(x["split"] for x in sources)
    lines = [
        "# X3D-M 四分类数据与训练实验记录", "",
        "## 数据规则", "",
        f"- 原始数据：`{root}` 下的 data1/data2/data3/data4。",
        "- 类别顺序：`normal=0, single_hand=1, board_drop=2, touch_face=3`。`no_gloves` 完全排除；`single_hand_or_drag` 按 `single_hand` 归并。",
        "- 先以原始视频为单位固定切分，再生成样本。同一原视频（含所有 annotation 和窗口）只会出现在 train 或 val 一侧。",
        f"- 固定随机种子：{seed}；按每个视频的标签组合分层，约 80% train / 20% val。验证清单见 `val_sources.csv`，后续不得用于训练、调参或重采样。",
        "- 以实测 FPS 就近归入 16/24/30 档，对应窗口 1.0/0.8/0.7 秒；每个窗口在读取时均匀抽取 16 帧。",
        "- 从 annotation 起点连续生成不重叠完整窗口，尾部余段丢弃。仅当整个片段没有完整窗口且长度大于 0.9 秒时，围绕该片段补足一个窗口；小于等于 0.9 秒不生成样本。normal 视频整段采用同样规则。",
        "- 单标签冲突控制：若一个候选窗口有至少 50% 时长同时落入另一目标类别的 annotation，则该窗口作为歧义样本排除，避免同一画面给四分类 softmax 提供互相矛盾的标签。",
        "", "## 固定切分与样本统计", "",
        f"- 原始视频总数：{len(sources)}；train {split_source_counts['train']}，val {split_source_counts['val']}。",
        f"- 样本总数：{len(samples)}；train {sum(v for (s, _), v in sample_counts.items() if s == 'train')}，val {sum(v for (s, _), v in sample_counts.items() if s == 'val')}。",
        "", "| split | normal | single_hand | board_drop | touch_face | 合计 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        vals = [sample_counts[(split, c)] for c in CLASSES]
        lines.append(f"| {split} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} |")
    total_vals = [sum(sample_counts[(s, c)] for s in ("train", "val")) for c in CLASSES]
    lines.append("| total | " + " | ".join(map(str, total_vals)) + f" | {sum(total_vals)} |")
    lines += ["", "## 排除与审计", ""]
    for key, value in sorted(excluded.items()):
        lines.append(f"- `{key}`：{value} 个 annotation/视频区间未生成样本。")
    lines.append(f"- 数据读取警告：{len(warnings)} 条，详见 `audit_warnings.txt`。")
    lines += [
        "", "## 训练实验设置", "",
        "- 架构：PyTorchVideo X3D-M，Kinetics-400 预训练；最后分类层替换为 4 类。输入为 RGB 16 帧、224×224。",
        "- 训练增强：保持宽高比缩放、随机裁剪、随机水平翻转；验证使用中心裁剪。归一化采用 Kinetics 均值/标准差。",
        "- 类别均衡：训练每个 epoch 对四类使用相同配额，配额取训练集中最少类的样本数；各类均无放回采样，多数类每轮轮换子集。因此不会因某类原始样本多而在单轮多采。验证集保持原始自然分布，不做均衡采样。",
        "- 优化：AdamW、交叉熵（label smoothing 0.1）、AMP、余弦学习率；按验证 macro-F1 保存最佳权重。具体超参数与每轮指标会由训练脚本追加到本文件，并另存 `config.json`、`history.csv`、`best_metrics.json`。",
        "- 防泄漏约束：训练脚本启动时会再次断言 train/val 的 `source_id` 交集为空。",
    ]
    (out / "EXPERIMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources, warnings = collect_sources(args.data_root)
    split_sources(sources, args.seed)
    samples, excluded = build_samples(sources)
    write_csv(args.output / "samples.csv", samples)
    source_rows = [{
        "source_id": x["source_id"], "split": x["split"], "data_group": x["data_group"],
        "video_path": x["video_path"], "json_path": x.get("json_path", ""),
        "labels": "+".join(sorted({a["label"] for a in x["annotations"]})),
        "fps": x["meta"]["fps"], "duration": x["meta"]["duration"], "frame_count": x["meta"]["frame_count"],
    } for x in sources]
    write_csv(args.output / "source_split.csv", source_rows)
    write_csv(args.output / "val_sources.csv", [x for x in source_rows if x["split"] == "val"])
    (args.output / "audit_warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    (args.output / "classes.json").write_text(json.dumps({c: i for i, c in enumerate(CLASSES)}, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(args.output, args.data_root, sources, samples, excluded, warnings, args.seed)
    print(json.dumps({"sources": len(sources), "samples": len(samples), "warnings": len(warnings),
                      "counts": {f"{s}/{c}": sum(x["split"] == s and x["label"] == c for x in samples)
                                 for s in ("train", "val") for c in CLASSES}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
