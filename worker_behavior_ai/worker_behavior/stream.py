"""worker_ai.stream: real-time loop for file / webcam / RTSP.

Emits events (normal/single_hand/board_drop/touch_face) with start/end time,
optionally writes annotated video, snapshots at event start, and an events json.

Examples:
  python -m worker_ai.stream --source 0                     # webcam
  python -m worker_ai.stream --source rtsp://user:pw@ip/..  # RTSP
  python -m worker_ai.stream --source video.mp4 --out out.mp4 --json events.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from .engine import BehaviorEngine
from . import config as C

COLOR = dict(normal=(0, 255, 0), single_hand=(0, 0, 255),
             board_drop=(0, 165, 255), touch_face=(255, 0, 255))


def _open_source(src):
    if isinstance(src, int) or src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)


class _LearnedEngine:
    """前置检测（板/手/骨架）+ 共享特征 + 学习到的时序模型。"""

    def __init__(self):
        from .detectors import DetectorSuite
        from .features import frame_features
        from .temporal import TemporalModel
        self.det = DetectorSuite()
        self._ff = frame_features
        self.model = TemporalModel()
        self._prev_bottom = None
        self.reset()

    def reset(self):
        self._prev_bottom = None
        self.model.reset()
        self.state = "normal"

    def step(self, frame, dt=1.0 / 24.0):
        H, W = frame.shape[:2]
        board, hands, kp = self.det.detect(frame)
        x, bottom = self._ff(board, hands, kp, W, H, self._prev_bottom, dt)
        self._prev_bottom = bottom
        self.state = self.model.update(x)
        return self.state


def run(source, every=C.SAMPLE_EVERY, out_video=None, json_path=None,
        snap_dir=None, show=False, on_event=None, mode=C.MODE):
    eng = _LearnedEngine() if mode == "learn" else BehaviorEngine()
    cap = _open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_video:
        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / every, (W, H))
    if snap_dir:
        os.makedirs(snap_dir, exist_ok=True)
    events, cur = [], None
    cnt = 0
    dt = 1.0 / (fps / every)
    prev = "normal"
    t0 = time.time()
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if cnt % every != 0:
                cnt += 1
                continue
            t = cnt / fps
            out_step = eng.step(f, dt)
            label = out_step if mode == "learn" else eng.classify(out_step, fps)
            # event segmentation
            if label != "normal" and prev == "normal":
                cur = dict(label=label, t0=round(t, 2), snap=None)
                if snap_dir:
                    p = os.path.join(snap_dir, f"{int(t*1000):07d}_{label}.jpg")
                    cv2.imwrite(p, f)
                    cur["snap"] = p
            elif label == "normal" and prev != "normal" and cur is not None:
                cur["t1"] = round(t, 2)
                if cur["t1"] - cur["t0"] >= C.EVENT_MIN_SEC:
                    events.append(cur)
                    if on_event:
                        on_event(cur)
                cur = None
            elif label != prev and cur is not None:
                cur["t1"] = round(t, 2)
                if cur["t1"] - cur["t0"] >= C.EVENT_MIN_SEC:
                    events.append(cur)
                    if on_event:
                        on_event(cur)
                cur = dict(label=label, t0=round(t, 2))
            prev = label
            # annotate
            if writer is not None or show:
                cv2.rectangle(f, (8, 8), (int(W * 0.55), 46), (0, 0, 0), -1)
                cv2.putText(f, f"{label} [{t:.1f}s]", (14, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR.get(label, (255, 255, 255)), 2)
            if writer is not None:
                writer.write(f)
            if show:
                cv2.imshow("worker_ai", f)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cnt += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    if cur is not None:                     # close an event still active at end
        cur["t1"] = round(cnt / fps, 2)
        if cur["t1"] - cur["t0"] >= C.EVENT_MIN_SEC:
            events.append(cur)
    if json_path:
        json.dump(events, open(json_path, "w"), indent=1)
    print(f"[worker_behavior] {time.time()-t0:.0f}s processed {cnt} frames -> events={len(events)}")
    for e in events:
        print("   ", e)
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="video path | webcam index | rtsp://..")
    ap.add_argument("--every", type=int, default=C.SAMPLE_EVERY)
    ap.add_argument("--out", default=None, help="annotated video out (.mp4)")
    ap.add_argument("--json", default=None, help="events out (.json)")
    ap.add_argument("--snap-dir", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--mode", default=C.MODE, choices=["learn", "rule"],
                    help="learn: classifier+Viterbi (default); rule: threshold state machine")
    a = ap.parse_args()
    run(a.source, a.every, a.out, a.json, a.snap_dir, a.show, mode=a.mode)


if __name__ == "__main__":
    main()
