import hashlib
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import av


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description="Audit industrial videos and JSON annotations.")
parser.add_argument("--video-root", type=Path, required=True)
parser.add_argument("--annotation-root", type=Path, required=True)
args = parser.parse_args()

videos = sorted(args.video_root.glob("*/*.mp4"))
annotations = {}
for path in sorted(args.annotation_root.glob("*/*.json")):
    with path.open("r", encoding="utf-8-sig") as stream:
        annotations[(path.parent.name, path.stem)] = json.load(stream)

issues = []
rows = []
hash_groups = defaultdict(list)
for path in videos:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        frames = int(stream.frames or 0)
        duration = float(container.duration / av.time_base) if container.duration else None
        width, height = stream.width, stream.height
    file_hash = sha256(path)
    hash_groups[file_hash].append(str(path))
    key = (path.parent.name, path.stem)
    item = annotations.get(key)
    intervals = [] if item is None else item.get("annotations", [])
    abnormal_seconds = sum(float(x["end_sec"]) - float(x["start_sec"]) for x in intervals)
    for interval in intervals:
        start, end = float(interval["start_sec"]), float(interval["end_sec"])
        if not (0 <= start < end <= duration + 0.1):
            issues.append(f"invalid interval {path}: {start}-{end}, duration={duration}")
        if interval.get("label") != path.parent.name:
            issues.append(f"label mismatch {path}: {interval.get('label')}")
    if path.parent.name == "normal" and item is not None:
        issues.append(f"normal video unexpectedly has JSON: {path}")
    if path.parent.name != "normal" and item is None:
        issues.append(f"missing JSON: {path}")
    rows.append(
        {
            "class": path.parent.name,
            "file": path.name,
            "duration": duration,
            "fps": fps,
            "frames": frames,
            "resolution": f"{width}x{height}",
            "intervals": len(intervals),
            "abnormal_seconds": abnormal_seconds,
            "abnormal_ratio": abnormal_seconds / duration if duration else None,
        }
    )

class_summary = {}
for class_name in sorted({row["class"] for row in rows}):
    selected = [row for row in rows if row["class"] == class_name]
    ratios = [row["abnormal_ratio"] for row in selected if row["intervals"]]
    class_summary[class_name] = {
        "videos": len(selected),
        "duration_sec_total": round(sum(row["duration"] for row in selected), 3),
        "duration_sec_min": round(min(row["duration"] for row in selected), 3),
        "duration_sec_max": round(max(row["duration"] for row in selected), 3),
        "abnormal_sec_total": round(sum(row["abnormal_seconds"] for row in selected), 3),
        "abnormal_ratio_mean": round(sum(ratios) / len(ratios), 4) if ratios else 0,
    }

report = {
    "video_count": len(videos),
    "json_count": len(annotations),
    "class_summary": class_summary,
    "resolutions": Counter(row["resolution"] for row in rows),
    "fps_rounded": Counter(round(row["fps"], 3) for row in rows),
    "exact_duplicate_groups": [paths for paths in hash_groups.values() if len(paths) > 1],
    "annotation_count_distribution": Counter(row["intervals"] for row in rows),
    "issues": issues,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
