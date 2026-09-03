from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)

    classes = cfg["classes"]
    class_to_idx = {name: index for index, name in enumerate(classes)}
    video_root = Path(cfg["data"]["video_root"])
    manifest_root = Path(cfg["data"]["manifest_root"])
    seed = int(cfg["seed"])
    val_fraction = float(cfg["data"]["validation_fraction"])
    test_fraction = float(cfg["data"]["test_fraction"])

    # Split by UCF video group (gXX), not by individual clip. Clips from one
    # recording group therefore cannot leak across train/val/test.
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in video_root.rglob("*.avi"):
        class_name = path.parent.name
        if class_name not in class_to_idx:
            continue
        match = re.search(r"_g(\d+)_c\d+\.avi$", path.name, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot parse UCF group from {path}")
        grouped[class_name][match.group(1)].append(path.resolve())

    rng = random.Random(seed)
    split_paths: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}
    for class_name in classes:
        group_names = sorted(grouped[class_name])
        if len(group_names) < 3:
            raise ValueError(f"{class_name} has only {len(group_names)} groups")
        rng.shuffle(group_names)
        test_groups = max(1, round(len(group_names) * test_fraction))
        val_groups = max(1, round(len(group_names) * val_fraction))
        assignments = {
            "test": group_names[:test_groups],
            "val": group_names[test_groups : test_groups + val_groups],
            "train": group_names[test_groups + val_groups :],
        }
        for split, names in assignments.items():
            for group_name in names:
                split_paths[split].extend((path, class_name) for path in grouped[class_name][group_name])

    def materialize(source: list[tuple[Path, str]]) -> list[dict]:
        result = []
        for absolute, class_name in sorted(source, key=lambda item: str(item[0])):
            result.append(
                {
                    "video_path": str(absolute),
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                }
            )
        return result

    manifests = {split: materialize(paths) for split, paths in split_paths.items()}
    for split, rows in manifests.items():
        write_manifest(manifest_root / f"{split}.csv", rows)

    summary = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "seed": seed,
        "split_strategy": "class-stratified UCF gXX recording-group split",
        "validation_group_fraction": val_fraction,
        "test_group_fraction": test_fraction,
        "counts": {
            split: {
                "total": len(rows),
                "per_class": dict(Counter(row["class_name"] for row in rows)),
            }
            for split, rows in manifests.items()
        },
    }
    (manifest_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
