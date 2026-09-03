from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import av
import yaml
from sklearn.model_selection import train_test_split


FIELDS = [
    "sample_id", "video_path", "source_id", "label", "class_name",
    "sample_origin", "roi_path", "video_duration", "window_start", "window_end", "event_start", "event_end",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        if container.duration:
            return float(container.duration / av.time_base)
        stream = container.streams.video[0]
        if stream.frames and stream.average_rate:
            return float(stream.frames / stream.average_rate)
    raise RuntimeError(f"Cannot determine video duration: {path}")


def fit_window(center: float, duration: float, clip_duration: float) -> tuple[float, float]:
    size = min(duration, clip_duration)
    start = max(0.0, min(center - size / 2, duration - size))
    return start, start + size


def normal_windows_outside_events(
    duration: float,
    events: list[tuple[float, float]],
    clip_duration: float,
    safety_margin: float,
    stride: float,
    max_windows: int,
) -> list[tuple[float, float]]:
    """Return fixed normal windows that cannot overlap an expanded anomaly interval."""
    blocked = []
    for start, end in sorted(events):
        start = max(0.0, start - safety_margin)
        end = min(duration, end + safety_margin)
        if blocked and start <= blocked[-1][1]:
            blocked[-1] = (blocked[-1][0], max(blocked[-1][1], end))
        else:
            blocked.append((start, end))

    allowed = []
    cursor = 0.0
    for start, end in blocked:
        if start > cursor:
            allowed.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        allowed.append((cursor, duration))

    windows = []
    for start, end in allowed:
        if end - start + 1e-6 < clip_duration:
            continue
        position = start
        while position + clip_duration <= end + 1e-6:
            windows.append((position, min(end, position + clip_duration)))
            position += stride
        last_start = end - clip_duration
        if not windows or abs(windows[-1][0] - last_start) > 1e-3:
            windows.append((last_start, end))

    if max_windows > 0 and len(windows) > max_windows:
        indices = [round(i * (len(windows) - 1) / (max_windows - 1)) for i in range(max_windows)] if max_windows > 1 else [len(windows) // 2]
        windows = [windows[index] for index in indices]
    return windows


def validate_roi(path: Path) -> None:
    values = path.read_text(encoding="utf-8-sig").strip().split()
    if len(values) != 5:
        raise ValueError(f"ROI must contain class_id cx cy width height: {path}")
    _, cx, cy, width, height = values
    cx, cy, width, height = map(float, (cx, cy, width, height))
    x1, x2 = cx - width / 2, cx + width / 2
    y1, y2 = cy - height / 2, cy + height / 2
    # YOLO labels are commonly rounded to six decimals, which can put an edge
    # a few millionths outside [0, 1]. The cropper clips/pads that harmless
    # rounding error; keep rejecting materially invalid boxes.
    tolerance = 1e-4
    if width <= 0 or height <= 0 or x1 < -tolerance or y1 < -tolerance or x2 > 1 + tolerance or y2 > 1 + tolerance:
        raise ValueError(f"Invalid normalized ROI coordinates in {path}: {values}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    classes = list(cfg["classes"])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    video_root = Path(cfg["data"]["video_root"])
    annotation_root = Path(cfg["data"]["annotation_root"])
    manifest_root = Path(cfg["data"]["manifest_root"])
    test_fraction = float(cfg["data"].get("test_fraction", 0.2))
    clip_duration = float(cfg["model"]["clip_duration_sec"])
    hard_negative_cfg = cfg["data"].get("normal_from_anomaly", {})
    use_hard_negatives = bool(hard_negative_cfg.get("enabled", False))
    safety_margin = float(hard_negative_cfg.get("safety_margin_sec", 1.0))
    normal_stride = float(hard_negative_cfg.get("stride_sec", clip_duration))
    max_normal_windows = int(hard_negative_cfg.get("max_windows_per_source", 6))
    source_groups = cfg["data"].get("source_groups")
    if not source_groups:
        source_groups = [
            {"name": name, "video_dir": name, "annotation_dir": name, "default_label": name}
            for name in classes
        ]
    label_aliases = cfg["data"].get("label_aliases", {})
    require_roi = cfg["model"].get("input_mode", "full_frame") == "roi"
    rng = random.Random(int(cfg["seed"]))

    sources: list[dict] = []
    issues: list[str] = []
    for group in source_groups:
        group_name = group["name"]
        video_dir = group.get("video_dir", group_name)
        annotation_dir = group.get("annotation_dir", group_name)
        is_normal = bool(group.get("normal", group_name == "normal"))
        default_label = group.get("default_label")
        event_window_mode = group.get("event_window_mode", "centered")
        for video in sorted((video_root / video_dir).glob("*.mp4"), key=lambda p: int(p.stem)):
            duration = video_duration(video)
            source_id = f"{group_name}/{video.stem}"
            json_path = annotation_root / annotation_dir / f"{video.stem}.json"
            roi_path = video.parent / f"{video.stem}_first_frame.txt"
            if roi_path.exists():
                try:
                    validate_roi(roi_path)
                except ValueError as exc:
                    issues.append(str(exc))
            elif require_roi:
                issues.append(f"missing ROI: {roi_path}")
            events = []
            if is_normal:
                if json_path.exists():
                    issues.append(f"normal source unexpectedly annotated: {json_path}")
            else:
                if not json_path.exists():
                    issues.append(f"missing annotation: {json_path}")
                else:
                    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
                    if payload.get("video", {}).get("file_name") != video.name:
                        issues.append(f"filename mismatch: {json_path}")
                    for event in payload.get("annotations", []):
                        start, end = float(event["start_sec"]), float(event["end_sec"])
                        raw_label = event.get("label")
                        event_label = label_aliases.get(raw_label, raw_label)
                        if event_label not in class_to_idx:
                            issues.append(f"unknown event label {raw_label!r}: {json_path}")
                        if default_label and event_label != default_label:
                            issues.append(f"label mismatch, expected {default_label}, found {raw_label}: {json_path}")
                        if not 0 <= start < end <= duration + 0.1:
                            issues.append(f"invalid interval: {json_path}: {start}-{end}/{duration}")
                        events.append({"start": start, "end": end, "label": event_label})
                    events.sort(key=lambda item: (item["start"], item["end"]))
                    for previous, current in zip(events, events[1:]):
                        if previous["end"] > current["start"]:
                            issues.append(f"overlapping annotations: {json_path}")
                if not events:
                    issues.append(f"no anomaly intervals: {json_path}")
            sources.append({
                "source_id": source_id,
                "source_group": group_name,
                "is_normal": is_normal,
                "event_window_mode": event_window_mode,
                "video_path": str(video.resolve()),
                "roi_path": str(roi_path.resolve()) if roi_path.exists() else "",
                "duration": duration,
                "events": events,
            })
    if issues:
        raise RuntimeError("\n".join(issues))

    def samples_for(source: dict) -> list[dict]:
        if source["is_normal"]:
            start, end = fit_window(source["duration"] / 2, source["duration"], clip_duration)
            samples = [(None, None, start, end, "normal_source", class_to_idx["normal"], "normal")]
        else:
            samples = []
            for event in source["events"]:
                event_start, event_end, event_label = event["start"], event["end"], event["label"]
                if source["event_window_mode"] == "annotation":
                    start, end = event_start, event_end
                    sample_origin = "segmented_anomaly_event"
                else:
                    start, end = fit_window((event_start + event_end) / 2, source["duration"], clip_duration)
                    sample_origin = "anomaly_event"
                samples.append((event_start, event_end, start, end, sample_origin, class_to_idx[event_label], event_label))
            if use_hard_negatives:
                normal_windows = normal_windows_outside_events(
                    source["duration"], [(event["start"], event["end"]) for event in source["events"]], clip_duration,
                    safety_margin, normal_stride, max_normal_windows,
                )
                samples.extend((None, None, start, end, "normal_from_anomaly", class_to_idx["normal"], "normal") for start, end in normal_windows)
        rows = []
        for event_index, (event_start, event_end, start, end, sample_origin, label, class_name) in enumerate(samples):
            rows.append({
                "sample_id": f"{source['source_id']}#{sample_origin}#{event_index}",
                "video_path": source["video_path"],
                "source_id": source["source_id"],
                "label": label,
                "class_name": class_name,
                "sample_origin": sample_origin,
                "roi_path": source["roi_path"],
                "video_duration": f"{source['duration']:.6f}",
                "window_start": f"{start:.6f}",
                "window_end": f"{end:.6f}",
                "event_start": "" if event_start is None else f"{event_start:.6f}",
                "event_end": "" if event_end is None else f"{event_end:.6f}",
            })
        return rows

    source_indices = list(range(len(sources)))
    stratify_groups = [source["source_group"] for source in sources]
    train_indices, test_indices = train_test_split(
        source_indices,
        test_size=test_fraction,
        random_state=int(cfg["seed"]),
        shuffle=True,
        stratify=stratify_groups,
    )
    split_sources = {
        "train": [sources[index] for index in train_indices],
        "test": [sources[index] for index in test_indices],
    }
    split_rows = {
        split: [row for source in selected for row in samples_for(source)]
        for split, selected in split_sources.items()
    }
    for split, rows in split_rows.items():
        rows.sort(key=lambda row: (int(row["label"]), row["source_id"], row["sample_id"]))
        write_csv(manifest_root / f"{split}.csv", rows)
    source_sets = {split: {source["source_id"] for source in selected} for split, selected in split_sources.items()}
    overlap = sorted(source_sets["train"] & source_sets["test"])
    if overlap:
        raise RuntimeError(f"source leakage between train and test: {overlap}")
    audit = {
        "seed": int(cfg["seed"]),
        "split": "source_stratified_holdout",
        "requested_test_fraction": test_fraction,
        "classes": classes,
        "all_sources": len(sources),
        "all_samples": sum(len(samples_for(source)) for source in sources),
        "source_counts": {split: len(items) for split, items in split_sources.items()},
        "actual_test_source_fraction": len(split_sources["test"]) / len(sources),
        "sample_counts": {split: len(items) for split, items in split_rows.items()},
        "source_group_counts": {
            split: dict(Counter(item["source_group"] for item in items))
            for split, items in split_sources.items()
        },
        "sample_class_counts": {
            split: dict(Counter(item["class_name"] for item in items))
            for split, items in split_rows.items()
        },
        "sample_origin_counts": {
            split: dict(Counter(item["sample_origin"] for item in items))
            for split, items in split_rows.items()
        },
        "source_overlap": overlap,
        "train_sources": sorted(source_sets["train"]),
        "test_sources": sorted(source_sets["test"]),
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
