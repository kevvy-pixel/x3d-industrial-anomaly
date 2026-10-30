"""Production delivery loop for worker_behavior.

This is the deployable counterpart of :mod:`worker_behavior.stream`: it adds
RTSP reconnects, alarm confirmation/merging, pre/post-event evidence,
metadata, health, rotating logs and an optional HTTP callback.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import logging.handlers
import os
import signal
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from urllib import error, request

import cv2
import numpy as np
import yaml

from .stream import _LearnedEngine

LOG = logging.getLogger("worker_behavior.delivery")
STOP = Event()
ALARM_TYPES = {"single_hand": "one_hand_board", "board_drop": "board_fall", "touch_face": "touch_face"}


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() else root / p


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def redact_source(source):
    text = str(source)
    if "://" not in text:
        return text
    from urllib.parse import urlsplit, urlunsplit
    u = urlsplit(text)
    host = u.hostname or ""
    if u.port:
        host += f":{u.port}"
    return urlunsplit((u.scheme, ("***@" if u.username else "") + host, u.path, u.query, u.fragment))


def setup_logging(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.handlers.clear()
    LOG.setLevel(getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.handlers.RotatingFileHandler(path, maxBytes=int(cfg.get("max_bytes", 10_485_760)),
                                                     backupCount=int(cfg.get("backups", 5)), encoding="utf-8"),
              logging.StreamHandler()):
        h.setFormatter(fmt); LOG.addHandler(h)


class AlarmTracker:
    def __init__(self, cfg, events_path):
        self.thresholds = {k: float(v) for k, v in cfg["thresholds"].items()}
        self.rules = cfg["confirmation"]
        self.merge_gap = float(cfg.get("merge_gap_sec", 2))
        n = max(int(x.get("window", x.get("count", 1))) for x in self.rules.values())
        self.history = {k: deque(maxlen=n) for k in self.thresholds}
        self.active = None
        self.events_path = events_path
        events_path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self):
        for h in self.history.values(): h.clear()
        self.active = None

    def _confirmed(self, label):
        rule, values = self.rules[label], list(self.history[label])
        if rule.get("mode") == "any":
            return sum(values[-int(rule.get("window", 1)):]) >= int(rule.get("min_count", 1))
        count = int(rule.get("count", 1))
        return len(values) >= count and all(values[-count:])

    def close(self, t, reason):
        if not self.active: return
        item = dict(self.active)
        end = max(item.pop("last_positive_sec"), t - self.merge_gap)
        item.update(end_sec=round(end, 3), duration_sec=round(max(0, end-item["start_sec"]), 3), close_reason=reason)
        with self.events_path.open("a", encoding="utf-8") as f: f.write(json.dumps(item, ensure_ascii=False) + "\n")
        LOG.info("alarm closed label=%s reason=%s", item["label"], reason)
        self.active = None

    def update(self, scores, t, wall):
        confirmed = []
        for label, threshold in self.thresholds.items():
            self.history[label].append(scores.get(label, 0) >= threshold)
            if self._confirmed(label): confirmed.append(label)
        selected = max(confirmed, key=lambda x: scores.get(x, 0)-self.thresholds[x], default=None)
        started = None
        if selected:
            if self.active and self.active["label"] != selected: self.close(t, "class_changed")
            if not self.active:
                self.active = {"event_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"), "event_time": wall,
                               "label": selected, "start_sec": round(t, 3), "last_positive_sec": t,
                               "max_confidence": round(scores.get(selected, 0), 6)}
                started = dict(self.active)
                LOG.warning("alarm started label=%s confidence=%.3f", selected, scores.get(selected, 0))
            else:
                self.active["last_positive_sec"] = t
                self.active["max_confidence"] = round(max(self.active["max_confidence"], scores.get(selected, 0)), 6)
        elif self.active and t - self.active["last_positive_sec"] > self.merge_gap:
            self.close(t, "merge_gap_elapsed")
        return (self.active["label"] if self.active else None), started


def open_capture(source, cfg):
    if isinstance(source, str):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{cfg.get('transport', 'tcp')}")
    params = []
    for name, key in (("CAP_PROP_OPEN_TIMEOUT_MSEC", "open_timeout_ms"), ("CAP_PROP_READ_TIMEOUT_MSEC", "read_timeout_ms")):
        if hasattr(cv2, name): params += [getattr(cv2, name), int(cfg.get(key, 10000))]
    try: cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params) if isinstance(source, str) else cv2.VideoCapture(source)
    except (cv2.error, TypeError): cap = cv2.VideoCapture(source)
    if isinstance(source, str) and params and not cap.isOpened():
        cap.release(); cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cfg.get("buffer_size", 2)))
    return cap


def encode_mp4(path, frames, fps, max_width):
    h, w = frames[0].shape[:2]
    scale = min(1, max_width / max(w, 1))
    size = (max(2, int(w*scale)//2*2), max(2, int(h*scale)//2*2))
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".writing.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), size)
    if not writer.isOpened(): raise RuntimeError("no usable MP4 encoder")
    for frame in frames: writer.write(cv2.resize(frame, size))
    writer.release(); os.replace(tmp, path)
    return "mp4v", *size


def callback(url, token, metadata, video_path, cfg):
    payload = dict(metadata)
    payload[cfg.get("media_field", "imageBase64")] = "data:video/mp4;base64," + base64.b64encode(video_path.read_bytes()).decode()
    headers = {"Content-Type": "application/json"}
    if token: headers["Authorization"] = f"Bearer {token}"
    for attempt in range(1, int(cfg.get("retries", 3))+1):
        try:
            req = request.Request(url, json.dumps(payload, ensure_ascii=False).encode(), headers, method="POST")
            with request.urlopen(req, timeout=float(cfg.get("timeout_sec", 15))) as r: r.read()
            LOG.info("callback OK status=%s event=%s", r.status, metadata["eventId"]); return
        except (error.URLError, TimeoutError, OSError) as exc:
            LOG.warning("callback failed attempt=%s error=%s", attempt, exc)
            if attempt < int(cfg.get("retries", 3)): time.sleep(float(cfg.get("retry_backoff_sec", 1)) * 2**(attempt-1))


def finalize(record, cfg, callback_cfg, callback_url, token, device_id):
    try:
        frames = record["frames"]
        path = Path(cfg["directory"]) / datetime.now().strftime("%Y%m%d") / f"{device_id}_{record['label']}_{record['event_id']}.mp4"
        codec, width, height = encode_mp4(path, frames, record["fps"], int(cfg.get("max_width", 1280)))
        metadata = {"eventId": record["event_id"], "deviceId": device_id, "eventTime": record["event_time"],
                    "label": record["label"], "alarmType": ALARM_TYPES.get(record["label"], record["label"]),
                    "maxConfidence": record["confidence"], "videoPath": str(path.resolve()), "codec": codec,
                    "fps": round(record["fps"], 3), "frames": len(frames), "width": width, "height": height,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        LOG.warning("evidence saved label=%s path=%s frames=%s", record["label"], path, len(frames))
        if callback_url: callback(callback_url, token, metadata, path, callback_cfg)
    except Exception: LOG.exception("evidence finalization failed event=%s", record.get("event_id"))


def run(source, config_path, device_id=None, callback_url=None, display=False, max_windows=None):
    root = Path(config_path).resolve().parent
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    for section, key in (("logging", "file"), ("health", "file"), ("alarm", "events_jsonl"), ("evidence", "directory")):
        cfg[section][key] = str(resolve(root, cfg[section][key]))
    setup_logging(Path(cfg["logging"]["file"]), cfg["logging"])
    source = int(source) if str(source).isdecimal() else str(source)
    device_id = device_id or os.environ.get("WORKER_DEVICE_ID") or cfg["deployment"]["device_id"]
    callback_url = callback_url if callback_url is not None else os.environ.get("WORKER_CALLBACK_URL", cfg["callback"].get("url", ""))
    token = os.environ.get("WORKER_CALLBACK_TOKEN", "")
    cap = open_capture(source, cfg["stream"]); is_file = isinstance(source, str) and Path(source).is_file()
    fallback = float(cfg["stream"].get("fallback_fps", 24)); fps = cap.get(cv2.CAP_PROP_FPS) or fallback
    tracker = AlarmTracker(cfg["alarm"], Path(cfg["alarm"]["events_jsonl"]))
    eng = _LearnedEngine(); pre, post = float(cfg["evidence"]["pre_sec"]), float(cfg["evidence"]["post_sec"])
    buf, recordings = deque(), {}; last_health = 0; windows = reconnects = failures = 0; began = time.monotonic(); last_label = "normal"
    LOG.info("source=%s fps=%.3f device=%s", redact_source(source), fps, device_id)
    try:
        while not STOP.is_set():
            if not cap.isOpened():
                failures += 1; maximum = int(cfg["stream"].get("max_reconnect_attempts", 0))
                if maximum and failures > maximum: break
                delay = min(float(cfg["stream"].get("reconnect_max_sec", 30)), float(cfg["stream"].get("reconnect_initial_sec", 2))*2**min(failures-1, 6))
                LOG.warning("source unavailable; reconnect attempt=%s delay=%.1fs", failures, delay); STOP.wait(delay)
                cap = open_capture(source, cfg["stream"])
                if cap.isOpened(): failures = 0; reconnects += 1; buf.clear(); tracker.reset(); eng.reset()
                continue
            ok, frame = cap.read()
            if not ok:
                if is_file: break
                cap.release(); tracker.close(time.monotonic()-began+tracker.merge_gap, "stream_disconnected"); continue
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if is_file else time.monotonic()-began
            if buf and frame.shape[:2] != buf[-1][1].shape[:2]: buf.clear(); tracker.reset(); eng.reset()
            buf.append((t, frame.copy()))
            while buf and t-buf[0][0] > max(pre+post+2, 5): buf.popleft()
            for record in list(recordings.values()):
                if t > record["last_frame_sec"]: record["frames"].append(frame.copy()); record["last_frame_sec"] = t
                if t >= record["until"]: recordings.pop(record["event_id"], None); finalize(record, cfg["evidence"], cfg["callback"], callback_url, token, device_id)
            label = eng.step(frame, 1/max(fps, 1)); p = eng.model.last_proba
            scores = {name: float(p[i]) for i, name in enumerate(eng.model.classes)} if p is not None else {label: 1.0}
            active, started = tracker.update(scores, t, now_iso()); windows += 1
            if started:
                recordings[started["event_id"]] = {"event_id": started["event_id"], "event_time": started["event_time"], "label": started["label"], "confidence": started["max_confidence"], "frames": [x[1].copy() for x in buf if x[0] >= t-pre], "last_frame_sec": t, "until": t+post, "fps": float(fps)}
            last_label = label
            if display:
                cv2.putText(frame, f"worker: {label} alarm={active or '-'}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255) if active else (0,200,0), 2); cv2.imshow("worker_behavior", frame)
                if cv2.waitKey(1) & 255 in (27, ord("q")): break
            if time.monotonic()-last_health >= float(cfg["health"].get("update_interval_sec", 5)):
                atomic_json(Path(cfg["health"]["file"]), {"status":"running", "updatedAt":now_iso(), "deviceId":device_id, "source":redact_source(source), "sourceFps":round(float(fps),3), "windows":windows, "reconnects":reconnects, "prediction":last_label, "activeAlarm":active})
                last_health = time.monotonic()
            if max_windows and windows >= max_windows: break
    finally:
        final_t = buf[-1][0] if buf else 0; tracker.close(final_t+tracker.merge_gap, "shutdown")
        for record in recordings.values():
            if record["frames"]: finalize(record, cfg["evidence"], cfg["callback"], callback_url, token, device_id)
        cap.release()
        if display: cv2.destroyAllWindows()
        atomic_json(Path(cfg["health"]["file"]), {"status":"stopped", "updatedAt":now_iso(), "deviceId":device_id, "windows":windows, "reconnects":reconnects})
    return 0


def main():
    ap = argparse.ArgumentParser(description="Production worker_behavior delivery")
    ap.add_argument("--source", help="RTSP URL, camera index or video; alternatively WORKER_RTSP_URL")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"config.yaml")
    ap.add_argument("--device-id"); ap.add_argument("--callback-url"); ap.add_argument("--display", action="store_true"); ap.add_argument("--max-windows", type=int)
    a = ap.parse_args(); source = a.source or os.environ.get("WORKER_RTSP_URL")
    if not source: print("no source; pass --source or set WORKER_RTSP_URL"); return 2
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    if hasattr(signal, "SIGTERM"): signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    return run(source, a.config, a.device_id, a.callback_url, a.display, a.max_windows)


if __name__ == "__main__": raise SystemExit(main())
