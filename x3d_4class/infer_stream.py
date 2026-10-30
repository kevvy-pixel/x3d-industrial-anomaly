from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

if os.name == "nt":
    cuda_bin = str(Path(sys.executable).resolve().parent / "bin")
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(cuda_bin)
import cv2
import numpy as np
import torch
import torch.nn.functional as F


CLASSES = ("normal", "single_hand", "board_drop", "touch_face")
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)


def fps_window(fps: float) -> float:
    return min(((16.0, 1.0), (24.0, 0.8), (30.0, 0.7)), key=lambda x: abs(fps - x[0]))[1]


def resize_rgb(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h < w:
        nh, nw = 256, round(w * 256 / h)
    else:
        nh, nw = round(h * 256 / w), 256
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def make_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    ids = np.rint(np.linspace(0, len(frames) - 1, 16)).astype(int)
    arr = np.stack([frames[i] for i in ids])
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
    h, w = x.shape[-2:]
    top, left = (h - 224) // 2, (w - 224) // 2
    x = x[:, :, top:top + 224, left:left + 224].permute(1, 0, 2, 3).unsqueeze(0)
    return ((x - MEAN) / STD).to(device)


def main() -> None:
    p = argparse.ArgumentParser(description="X3D-M four-class sliding-window stream inference")
    p.add_argument("--model", type=Path, required=True, help="x3d_m_4class.ts")
    p.add_argument("--source", default="0", help="camera index, video path, or RTSP URL")
    p.add_argument("--stride-sec", type=float, default=0.3)
    p.add_argument("--smooth", type=int, default=3, help="moving-average prediction count")
    p.add_argument("--board-threshold", type=float, default=0.60)
    p.add_argument("--single-threshold", type=float, default=0.60)
    p.add_argument("--touch-threshold", type=float, default=0.65)
    p.add_argument("--merge-gap-sec", type=float, default=2.0)
    p.add_argument("--display", action="store_true")
    p.add_argument("--output-video", type=Path)
    args = p.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 5:
        fps = 30.0
    raw_count = max(16, round(fps * fps_window(fps)))
    stride_frames = max(1, round(fps * args.stride_sec))
    buffer: deque[np.ndarray] = deque(maxlen=raw_count)
    probs_history: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
    thresholds = {
        "board_drop": args.board_threshold,
        "single_hand": args.single_threshold,
        "touch_face": args.touch_threshold,
    }
    decisions = {name: deque(maxlen=2) for name in thresholds}
    events = {
        name: {"active": False, "start_sec": None, "last_positive_sec": -1e9}
        for name in thresholds
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(args.model), map_location=device).eval()
    writer = None
    frame_idx = 0
    latest = {"class": "warming_up", "confidence": 0.0, "probabilities": {c: 0.0 for c in CLASSES}}
    print(json.dumps({
        "event": "startup", "device": str(device), "camera_fps": fps,
        "window_sec": fps_window(fps), "raw_buffer_frames": raw_count,
        "sampled_frames": 16, "stride_sec": args.stride_sec,
        "smooth_windows": args.smooth, "thresholds": thresholds,
        "confirmation": {"board_drop": "1_of_2", "single_hand": "consecutive_2", "touch_face": "consecutive_2"},
        "merge_gap_sec": args.merge_gap_sec,
    }, ensure_ascii=False), flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buffer.append(resize_rgb(frame))
        if len(buffer) == raw_count and frame_idx % stride_frames == 0:
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(make_tensor(list(buffer), device))
                probs = torch.softmax(logits.float(), 1)[0].cpu().numpy()
            probs_history.append(probs)
            avg = np.mean(probs_history, axis=0)
            pred = int(avg.argmax())
            now_sec = frame_idx / fps
            event_updates = []
            for name, threshold in thresholds.items():
                score = float(avg[CLASSES.index(name)])
                decisions[name].append(score >= threshold)
                if name == "board_drop":
                    confirmed = any(decisions[name])
                else:
                    confirmed = len(decisions[name]) == 2 and all(decisions[name])
                state = events[name]
                if confirmed:
                    if not state["active"]:
                        state["active"] = True
                        state["start_sec"] = now_sec
                        event_updates.append({"event": "alarm_start", "class": name, "score": round(score, 5),
                                              "time_sec": round(now_sec, 3)})
                    state["last_positive_sec"] = now_sec
                elif state["active"] and now_sec - state["last_positive_sec"] > args.merge_gap_sec:
                    event_updates.append({"event": "alarm_end", "class": name,
                                          "start_sec": round(float(state["start_sec"]), 3),
                                          "end_sec": round(now_sec, 3)})
                    state["active"] = False
                    state["start_sec"] = None
            latest = {"frame": frame_idx, "time_sec": round(frame_idx / fps, 3), "class": CLASSES[pred],
                      "confidence": round(float(avg[pred]), 5),
                      "probabilities": {c: round(float(avg[i]), 5) for i, c in enumerate(CLASSES)},
                      "alarms_active": [name for name, state in events.items() if state["active"]],
                      "event_updates": event_updates,
                      "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
            print(json.dumps(latest, ensure_ascii=False), flush=True)
        text = f"{latest['class']} {latest['confidence']:.2f}"
        alarms = latest.get("alarms_active", [])
        color = (0, 0, 255) if alarms else (0, 255, 0)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if alarms:
            cv2.putText(frame, "ALARM: " + ", ".join(alarms), (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if args.output_video:
            if writer is None:
                args.output_video.parent.mkdir(parents=True, exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if args.display:
            cv2.imshow("X3D-M inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
    cap.release()
    if writer: writer.release()
    if args.display: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
