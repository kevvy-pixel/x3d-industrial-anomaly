from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


CLASSES = ("normal", "touch_face")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_samples_path = args.source_experiment / "samples.csv"
    source_split_path = args.source_experiment / "source_split.csv"
    cache_index_path = args.source_experiment.parent / "val_cache_224" / "cache_index.csv"
    rows = read_csv(source_samples_path)
    cache = {row["sample_id"]: row["cache_path"] for row in read_csv(cache_index_path)}
    output_rows: list[dict] = []
    for row in rows:
        original_label = row["label"]
        binary_label = "touch_face" if original_label == "touch_face" else "normal"
        converted = dict(row)
        converted["original_label"] = original_label
        converted["label"] = binary_label
        converted["label_idx"] = str(CLASSES.index(binary_label))
        converted["cache_path"] = cache.get(row["sample_id"], "") if row["split"] == "val" else ""
        output_rows.append(converted)

    train_sources = {row["source_id"] for row in output_rows if row["split"] == "train"}
    val_sources = {row["source_id"] for row in output_rows if row["split"] == "val"}
    overlap = train_sources & val_sources
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE: {len(overlap)} source IDs overlap")
    missing_cache = [row["sample_id"] for row in output_rows if row["split"] == "val" and not Path(row["cache_path"]).is_file()]
    if missing_cache:
        raise RuntimeError(f"Missing cached validation clips: {len(missing_cache)}")

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "samples.csv", output_rows)
    (args.output / "source_split.csv").write_bytes(source_split_path.read_bytes())
    (args.output / "val_sources.csv").write_bytes((args.source_experiment / "val_sources.csv").read_bytes())
    (args.output / "classes.json").write_text(json.dumps({name: i for i, name in enumerate(CLASSES)}, indent=2), encoding="utf-8")

    counts = Counter((row["split"], row["label"]) for row in output_rows)
    originals = Counter((row["split"], row["original_label"]) for row in output_rows)
    source_counts = Counter(row["split"] for row in {row["source_id"]: row for row in output_rows}.values())
    audit = {
        "source_samples": str(source_samples_path.resolve()),
        "source_split": str(source_split_path.resolve()),
        "source_samples_sha256": sha256(source_samples_path),
        "source_split_sha256": sha256(source_split_path),
        "split_reused_without_resplitting": True,
        "train_val_source_overlap": 0,
        "classes": list(CLASSES),
        "negative_definition": ["normal", "single_hand", "board_drop"],
        "sample_counts": {f"{split}/{label}": counts[(split, label)] for split in ("train", "val") for label in CLASSES},
        "source_counts": dict(source_counts),
        "original_label_counts": {f"{split}/{label}": value for (split, label), value in sorted(originals.items())},
    }
    (args.output / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# X3D-M normal / touch_face 二分类实验", "", "## 冻结划分与防泄漏", "",
        f"- 完全复用四分类实验的源视频级划分：`{source_split_path}`，没有重新随机划分。",
        f"- 划分文件 SHA-256：`{audit['source_split_sha256']}`。",
        f"- 原样本清单 SHA-256：`{audit['source_samples_sha256']}`。",
        f"- 唯一源视频：train {source_counts['train']}，val {source_counts['val']}；交集为 0。",
        "- 验证集只用于每轮评估、最佳 epoch 选择和阈值分析，绝不进入梯度更新。", "",
        "## 二分类定义", "",
        "- `touch_face`：原四分类清单中的 touch_face 异常片段。",
        "- `normal`：原 normal、single_hand、board_drop 样本。后两类作为不摸脸硬负样本，以降低其他手部动作被误报为摸脸的风险。",
        "- `nogloves` 仍不使用。混合异常视频沿用原 JSON 注释窗口及其源视频 split。", "",
        "## 样本统计", "", "| split | normal（含硬负样本） | touch_face | 合计 |", "|---|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        normal, touch = counts[(split, "normal")], counts[(split, "touch_face")]
        lines.append(f"| {split} | {normal} | {touch} | {normal + touch} |")
    lines += [
        "", "## 采样与训练约束", "",
        "- 每个样本固定 16 帧；约 16 FPS 使用 1.0 秒窗口、24 FPS 使用 0.8 秒、30 FPS 使用 0.7 秒；沿用原清单的 `start_sec/end_sec`。",
        "- 每个训练 epoch 两类严格等量、无放回采样；配额为较少类 touch_face 的全部训练样本数。负类使用跨 epoch 循环洗牌，以尽量覆盖全部硬负样本。",
        "- 验证每轮使用全部冻结验证样本，不做均衡抽样。模型按验证 macro-F1 保存最佳权重。",
        "- 初始化：载入原四分类最佳模型的 X3D-M 主干，丢弃四分类头，重新初始化二分类头，并对全模型微调。", "",
        "训练完成后，本文件将追加逐轮最佳指标、混淆矩阵和阈值分析。",
    ]
    (args.output / "EXPERIMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
