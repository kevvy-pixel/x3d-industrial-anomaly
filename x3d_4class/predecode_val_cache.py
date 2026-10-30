from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def cache_name(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + ".npy"


def decode_one(task: tuple[dict, str]) -> tuple[str, str, int, str]:
    row, output_dir = task
    cv2.setNumThreads(0)
    destination = Path(output_dir) / cache_name(row["sample_id"])
    if destination.exists():
        return row["sample_id"], str(destination), destination.stat().st_size, "cached"
    cap = cv2.VideoCapture(row["video_path"])
    if not cap.isOpened():
        return row["sample_id"], "", 0, "cannot_open"
    fps, frame_count = float(row["fps"]), int(row["frame_count"])
    start = float(row.get("cache_start_sec", row["start_sec"]))
    end = float(row.get("cache_end_sec", row["end_sec"]))
    indices = np.clip(np.rint(np.linspace(start * fps, end * fps, 16, endpoint=False)).astype(int), 0, frame_count - 1)
    wanted = {int(x) for x in indices}
    decoded: dict[int, np.ndarray] = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices.min()))
    last_good = None
    for frame_idx in range(int(indices.min()), int(indices.max()) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if h < w:
                nh, nw = 256, round(w * 256 / h)
            else:
                nh, nw = round(h * 256 / w), 256
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            top, left = (nh - 224) // 2, (nw - 224) // 2
            last_good = np.ascontiguousarray(rgb[top:top + 224, left:left + 224])
            decoded[frame_idx] = last_good
    cap.release()
    if last_good is None:
        return row["sample_id"], "", 0, "no_frames"
    frames = []
    previous = last_good
    for idx in indices:
        frame = decoded.get(int(idx), previous)
        frames.append(frame)
        previous = frame
    array = np.stack(frames).astype(np.uint8, copy=False)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(temporary, destination)
    return row["sample_id"], str(destination), destination.stat().st_size, "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--window-sec", type=float, default=None,
                   help="Override each sample with a centered full-duration window")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    if args.window_sec is not None:
        for row in rows:
            video_duration = int(row["frame_count"]) / float(row["fps"])
            center = (float(row["start_sec"]) + float(row["end_sec"])) / 2.0
            if video_duration <= args.window_sec:
                left, right = 0.0, video_duration
            else:
                left = min(max(0.0, center - args.window_sec / 2.0), video_duration - args.window_sec)
                right = left + args.window_sec
            row["cache_start_sec"] = f"{left:.6f}"
            row["cache_end_sec"] = f"{right:.6f}"
            row["cache_window_sec"] = f"{right - left:.6f}"
    by_id, failures, total_bytes = {}, [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(decode_one, (row, str(args.output))) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="predecode"):
            sample_id, path, size, status = future.result()
            if path:
                by_id[sample_id] = path
                total_bytes += size
            else:
                failures.append({"sample_id": sample_id, "status": status})
    index_rows = []
    for row in rows:
        if row["sample_id"] in by_id:
            index_rows.append({**row, "cache_path": by_id[row["sample_id"]]})
    with (args.output / "cache_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    summary = {"samples": len(index_rows), "failures": failures, "bytes": total_bytes,
               "gib": total_bytes / 1024**3, "shape": [16, 224, 224, 3], "dtype": "uint8"}
    summary["requested_window_sec"] = args.window_sec
    summary["window_alignment"] = "centered_and_boundary_shifted" if args.window_sec is not None else "manifest"
    (args.output / "cache_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} samples failed to decode")


if __name__ == "__main__":
    main()
