from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video_path", "label", "class_name", "group"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    classes = cfg["classes"]
    class_to_idx = {name: index for index, name in enumerate(classes)}
    video_root = Path(cfg["data"]["video_root"])
    output_root = Path(cfg["data"]["manifest_root"])
    rng = random.Random(int(cfg["seed"]))

    candidates: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in video_root.rglob("*.avi"):
        class_name = path.parent.name
        if class_name not in class_to_idx:
            continue
        # The compact mirror contains a few misplaced files. Require the UCF
        # filename class prefix to agree with the parent-directory label.
        if not path.name.lower().startswith(f"v_{class_name}_".lower()):
            continue
        match = re.search(r"_g(\d+)_c\d+\.avi$", path.name, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot parse UCF group: {path}")
        candidates[class_name][f"g{match.group(1)}"].append(path.resolve())

    manifests = {"train": [], "val": [], "test": []}
    selection = {}
    for class_name in classes:
        counts = cfg["data"]["split_counts"][class_name]
        total = sum(int(counts[split]) for split in manifests)
        group_names = sorted(candidates[class_name])
        if len(group_names) < total:
            raise ValueError(
                f"{class_name}: need {total} distinct groups, have {len(group_names)}"
            )
        rng.shuffle(group_names)
        selected_groups = group_names[:total]
        # Select exactly one clip from every group, then allocate whole groups.
        selected = []
        for group in selected_groups:
            options = sorted(candidates[class_name][group])
            selected.append((group, rng.choice(options)))

        cursor = 0
        selection[class_name] = {}
        for split in ("train", "val", "test"):
            count = int(counts[split])
            allocated = selected[cursor : cursor + count]
            cursor += count
            selection[class_name][split] = [
                {"group": group, "video": str(path)} for group, path in allocated
            ]
            for group, path in allocated:
                manifests[split].append(
                    {
                        "video_path": str(path),
                        "label": class_to_idx[class_name],
                        "class_name": class_name,
                        "group": group,
                    }
                )

    for split, rows in manifests.items():
        rows.sort(key=lambda row: (row["label"], row["group"], row["video_path"]))
        write_manifest(output_root / f"{split}.csv", rows)

    all_groups = {
        split: {(row["class_name"], row["group"]) for row in rows}
        for split, rows in manifests.items()
    }
    overlaps = {
        "train_val": sorted(all_groups["train"] & all_groups["val"]),
        "train_test": sorted(all_groups["train"] & all_groups["test"]),
        "val_test": sorted(all_groups["val"] & all_groups["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Group leakage detected: {overlaps}")

    summary = {
        "seed": int(cfg["seed"]),
        "strategy": "one video per distinct UCF gXX recording group",
        "counts": {split: len(rows) for split, rows in manifests.items()},
        "total": sum(len(rows) for rows in manifests.values()),
        "group_overlaps": overlaps,
        "selection": selection,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "fewshot_selection.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
