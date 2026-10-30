from __future__ import annotations

import argparse
import base64
import json
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from urllib import error, request

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml


def resize_short_side(video: torch.Tensor, target: int) -> torch.Tensor:
    _, _, height, width = video.shape
    if height < width:
        size = (target, round(width * target / height))
    else:
        size = (round(height * target / width), target)
    frames = F.interpolate(video.permute(1, 0, 2, 3), size=size, mode="bilinear", align_corners=False)
    return frames.permute(1, 0, 2, 3)


def preprocess(frames_bgr: list[np.ndarray], cfg: dict, device: torch.device) -> torch.Tensor:
    if not frames_bgr:
        raise ValueError("Cannot preprocess an empty frame list")
    # A decoder may briefly return a different resolution after an RTSP
    # reconnect. Normalizing defensively prevents np.stack() from crashing if
    # one such frame was already sampled into the inference window.
    target_height, target_width = frames_bgr[-1].shape[:2]
    normalized_frames = [
        frame
        if frame.shape[:2] == (target_height, target_width)
        else cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        for frame in frames_bgr
    ]
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in normalized_frames])
    video = torch.from_numpy(rgb).permute(3, 0, 1, 2).float().div_(255.0)
    video = resize_short_side(video, int(cfg["model"]["resize_short"]))
    crop = int(cfg["model"]["crop_size"])
    height, width = video.shape[-2:]
    top, left = max(0, (height - crop) // 2), max(0, (width - crop) // 2)
    video = video[:, :, top : top + crop, left : left + crop]
    mean = torch.tensor(cfg["model"]["mean"]).view(3, 1, 1, 1)
    std = torch.tensor(cfg["model"]["std"]).view(3, 1, 1, 1)
    return ((video - mean) / std).unsqueeze(0).to(device, non_blocking=True)


def infer_clip(
    model: torch.jit.ScriptModule,
    frames_bgr: list[np.ndarray],
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    tensor = preprocess(frames_bgr, cfg, device)
    with torch.inference_mode():
        return torch.softmax(model(tensor), dim=1)[0].float().cpu().numpy()


def sample_clip(buffer: deque, end_time: float, duration: float, count: int) -> list[np.ndarray]:
    targets = np.linspace(end_time - duration, end_time, count)
    times = np.asarray([item[0] for item in buffer])
    indices = [int(np.abs(times - target).argmin()) for target in targets]
    return [buffer[index][1] for index in indices]


def build_model(checkpoint_path: Path, classes: list[str], device: torch.device) -> torch.jit.ScriptModule:
    if checkpoint_path.suffix.lower() not in {".ts", ".torchscript"}:
        raise RuntimeError("This delivery build requires the exported TorchScript model (.ts)")
    model = torch.jit.load(str(checkpoint_path), map_location=device).eval()
    if len(classes) != 2 or classes != ["normal", "touch_face"]:
        raise RuntimeError(f"Expected class order [normal, touch_face], got {classes}")
    return model


class AlarmTracker:
    def __init__(self, config: dict, output_path: Path):
        self.thresholds = config["class_thresholds"]
        self.confirm = int(config["confirm_windows"])
        self.merge_gap = float(config["merge_gap_sec"])
        self.pending_label = None
        self.pending_count = 0
        self.active = None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path

    def _write(self, event: dict) -> None:
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def close(self, now: float) -> None:
        if self.active is None:
            return
        event = dict(self.active)
        event["end_sec"] = round(max(event["last_positive_sec"], now - self.merge_gap), 3)
        event["duration_sec"] = round(event["end_sec"] - event["start_sec"], 3)
        event.pop("last_positive_sec", None)
        self._write(event)
        self.active = None

    def update(self, label: str, confidence: float, now: float, stride: float) -> str | None:
        positive = label in self.thresholds and confidence >= float(self.thresholds[label])
        if positive:
            if self.active and self.active["label"] == label:
                self.active["last_positive_sec"] = now
                self.active["max_confidence"] = round(max(self.active["max_confidence"], confidence), 6)
            elif self.active:
                self.close(now)
            if self.active is None:
                if self.pending_label == label:
                    self.pending_count += 1
                else:
                    self.pending_label, self.pending_count = label, 1
                if self.pending_count >= self.confirm:
                    self.active = {
                        "label": label,
                        "start_sec": round(max(0.0, now - (self.confirm - 1) * stride), 3),
                        "last_positive_sec": now,
                        "max_confidence": round(confidence, 6),
                    }
                    self.pending_label, self.pending_count = None, 0
        else:
            self.pending_label, self.pending_count = None, 0
            if self.active and now - self.active["last_positive_sec"] > self.merge_gap:
                self.close(now)
        return self.active["label"] if self.active else None


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def open_capture(source) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return capture


def capture_fps(capture: cv2.VideoCapture, fallback: float = 24.0) -> float:
    value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return value if 1.0 <= value <= 120.0 else fallback


def select_clip_duration(fps: float, cfg: dict) -> tuple[float, int]:
    window = cfg["model"].get("window", {})
    if not window.get("adaptive_by_fps", False):
        return float(cfg["model"]["clip_duration_sec"]), round(fps)
    choices = {
        16: float(window["fps16"]),
        24: float(window["fps24"]),
        30: float(window["fps30"]),
    }
    bucket = min(choices, key=lambda nominal: abs(fps - nominal))
    return choices[bucket], bucket


def output_segment_path(output_path: Path, part: int) -> Path:
    if part <= 1:
        return output_path
    return output_path.with_name(f"{output_path.stem}_part{part:03d}{output_path.suffix}")


ALARM_TYPE_MAP = {
    "single_hand": "one_hand_board",
    "touch_face": "touch_face",
    "board_drop": "board_fall",
}


def encode_alarm_jpeg(frames: list[np.ndarray]) -> tuple[str, bytes]:
    # The last frame often occurs after a short action has ended. The middle
    # frame is normally closer to the action that caused the causal window to
    # become positive.
    frame = frames[len(frames) // 2]
    encoded_ok, encoded_image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not encoded_ok:
        raise RuntimeError("could not encode frame as JPEG")
    return "data:image/jpg;base64,", encoded_image.tobytes()


def encode_alarm_mp4(
    frames: list[np.ndarray],
    fps: float,
    max_width: int,
) -> tuple[str, bytes]:
    first_height, first_width = frames[0].shape[:2]
    scale = min(1.0, max_width / max(first_width, 1))
    width = max(2, int(round(first_width * scale)) // 2 * 2)
    height = max(2, int(round(first_height * scale)) // 2 * 2)
    temporary = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        rate = Fraction(max(1.0, fps)).limit_denominator(1000)
        with av.open(
            str(temporary_path),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        ) as container:
            stream = container.add_stream("libx264", rate=rate)
            stream.width = width
            stream.height = height
            # yuv420p + H.264/AVC (avc1 in MP4) is supported by browsers and
            # common Java/Node middleware players; OpenCV's mp4v often is not.
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "23"}
            stream.codec_context.gop_size = max(1, round(float(rate)))
            stream.codec_context.max_b_frames = 0
            for index, frame in enumerate(frames):
                output = frame
                if frame.shape[:2] != (height, width):
                    output = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                video_frame = av.VideoFrame.from_ndarray(output, format="bgr24")
                video_frame.pts = index
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        payload = temporary_path.read_bytes()
        if not payload:
            raise RuntimeError("temporary MP4 is empty")
        return "data:video/mp4;base64,", payload
    finally:
        temporary_path.unlink(missing_ok=True)


def save_alarm_video(
    output_dir: Path,
    label: str,
    frames: list[np.ndarray],
    event_time: str,
    video_fps: float,
    video_width: int,
) -> None:
    try:
        _, media_bytes = encode_alarm_mp4(frames, video_fps, video_width)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = "".join(character if character.isdigit() else "-" for character in event_time).strip("-")
        path = output_dir / f"{stamp}_{label}.mp4"
        sequence = 2
        while path.exists():
            path = output_dir / f"{stamp}_{label}_{sequence}.mp4"
            sequence += 1
        path.write_bytes(media_bytes)
        print(f"alarm video saved: {path} bytes={len(media_bytes)}", flush=True)
    except (OSError, RuntimeError) as exc:
        print(f"alarm video save FAILED: label={label} error={exc}", flush=True)


def post_alarm(
    callback_url: str,
    device_id: str,
    label: str,
    frames: list[np.ndarray],
    event_time: str,
    media_type: str,
    video_fps: float,
    video_width: int,
) -> None:
    alarm_type = ALARM_TYPE_MAP.get(label)
    if alarm_type is None:
        print(f"alarm callback skipped: unsupported label={label}", flush=True)
        return

    try:
        if media_type == "mp4":
            prefix, media_bytes = encode_alarm_mp4(frames, video_fps, video_width)
        else:
            prefix, media_bytes = encode_alarm_jpeg(frames)
    except (OSError, RuntimeError) as exc:
        if media_type != "mp4":
            print(f"alarm callback failed: {exc}", flush=True)
            return
        print(f"alarm MP4 encoding failed; falling back to JPEG: {exc}", flush=True)
        try:
            prefix, media_bytes = encode_alarm_jpeg(frames)
        except RuntimeError as fallback_exc:
            print(f"alarm callback failed: {fallback_exc}", flush=True)
            return

    payload = {
        "deviceId": device_id,
        "imageBase64": prefix + base64.b64encode(media_bytes).decode("ascii"),
        "eventTime": event_time,
        "alarmType": alarm_type,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        callback_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=15.0) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            print(
                f"alarm callback OK: status={response.status} alarm_type={alarm_type} "
                f"media={prefix.split(';', 1)[0]} bytes={len(media_bytes)} response={response_body}",
                flush=True,
            )
    except (error.URLError, TimeoutError, OSError) as exc:
        print(f"alarm callback FAILED: alarm_type={alarm_type} error={exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time X3D-M inference for camera, RTSP, or a video file.")
    parser.add_argument("--source", required=True, help="Camera index (0), video path, or RTSP URL")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--headless", action="store_true", help="Do not open the preview window")
    parser.add_argument("--output-video", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--alarm-callback-url", help="Alarm callback URL, e.g. http://127.0.0.1:9000/alarm/callback")
    parser.add_argument("--device-id", default="robot01", help="Device ID sent to the alarm callback")
    parser.add_argument(
        "--alarm-media",
        choices=("jpg", "mp4"),
        default="mp4",
        help="Alarm callback media. MP4 contains the causal clip that triggered the alarm (default: mp4)",
    )
    parser.add_argument("--alarm-video-width", type=int, default=640, help="Maximum alarm MP4 width")
    parser.add_argument(
        "--alarm-output-dir",
        type=Path,
        help="Directory for locally saved alarm MP4 files (default: alarm.evidence_dir in config)",
    )
    parser.add_argument("--reconnect-delay", type=float, default=2.0, help="Seconds between RTSP reconnect attempts")
    parser.add_argument(
        "--max-reconnect-attempts",
        type=int,
        default=0,
        help="Stop after N consecutive reconnect failures; 0 retries forever",
    )
    parser.add_argument("--max-windows", type=int, help="Stop after N inference windows (for testing)")
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    classes = list(cfg["model"]["classes"])
    requested_device = cfg["inference"].get("device", "auto")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else ("cpu" if requested_device == "auto" else requested_device))
    model = build_model(root / cfg["model"]["checkpoint"], classes, device)
    stride = float(cfg["inference"]["stride_sec"])
    frame_count = int(cfg["model"]["input_frames"])
    smooth_count = int(cfg["inference"]["smoothing_windows"])
    tracker = AlarmTracker(cfg["alarm"], root / cfg["alarm"]["events_jsonl"])
    configured_alarm_dir = args.alarm_output_dir or cfg["alarm"].get("evidence_dir")
    alarm_output_dir = Path(configured_alarm_dir) if configured_alarm_dir else None
    if alarm_output_dir is not None and not alarm_output_dir.is_absolute():
        alarm_output_dir = root / alarm_output_dir
    if device.type == "cpu":
        print(
            "warning: X3D is running on CPU; watch inference_lag because multi-second model latency is possible",
            flush=True,
        )

    source = parse_source(args.source)
    is_file = isinstance(source, str) and Path(source).is_file()
    capture = open_capture(source)
    if not capture.isOpened() and is_file:
        raise RuntimeError(f"Cannot open source: {args.source}")
    if not capture.isOpened():
        print(f"cannot open live source; reconnecting every {args.reconnect_delay:.1f}s", flush=True)
    fps = capture_fps(capture)
    duration, fps_bucket = select_clip_duration(fps, cfg)
    print(
        f"runtime: device={device} stream_fps={fps:.3f} fps_bucket={fps_bucket} "
        f"clip={duration:.3f}s frames={frame_count} stride={stride:.3f}s "
        f"smoothing_windows={smooth_count} confirm_windows={tracker.confirm} "
        f"touch_face_threshold={tracker.thresholds.get('touch_face')} alarm_media={args.alarm_media}",
        flush=True,
    )
    buffer: deque = deque()
    probability_history: deque = deque(maxlen=smooth_count)
    last_inference = -1e9
    prediction = ("warming_up", 0.0)
    writer = None
    writer_size = None
    writer_path = None
    writer_part = 1
    buffer_frame_shape = None
    windows = 0
    started = time.monotonic()
    alarm_executor = (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="alarm-output")
        if args.alarm_callback_url or alarm_output_dir
        else None
    )
    # RTSP decoding/recording must keep consuming frames while X3D is running.
    # Otherwise synchronous inference can starve FFmpeg's receive buffer and
    # produce a recording whose camera clock jumps forward by many seconds.
    inference_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="x3d-inference") if not is_file else None
    )
    inference_future = None
    inference_context = None
    pending_alarm_media = None
    stream_generation = 0
    reconnect_attempts = 0

    try:
        while True:
            if not capture.isOpened():
                reconnect_attempts += 1
                if args.max_reconnect_attempts and reconnect_attempts > args.max_reconnect_attempts:
                    print(f"RTSP reconnect stopped after {args.max_reconnect_attempts} failed attempts", flush=True)
                    break
                attempt_text = (
                    f"{reconnect_attempts}/{args.max_reconnect_attempts}"
                    if args.max_reconnect_attempts
                    else str(reconnect_attempts)
                )
                print(f"RTSP reconnect attempt {attempt_text} in {args.reconnect_delay:.1f}s...", flush=True)
                time.sleep(max(0.0, args.reconnect_delay))
                capture = open_capture(source)
                if not capture.isOpened():
                    continue
                fps = capture_fps(capture, fps)
                duration, fps_bucket = select_clip_duration(fps, cfg)
                buffer.clear()
                probability_history.clear()
                buffer_frame_shape = None
                last_inference = -1e9
                prediction = ("warming_up", 0.0)
                reconnect_attempts = 0
                print(
                    f"RTSP reconnected; stream_fps={fps:.3f} fps_bucket={fps_bucket} "
                    f"clip={duration:.3f}s; warming up inference buffer",
                    flush=True,
                )

            ok, frame = capture.read()
            if not ok:
                if is_file:
                    break
                disconnect_time = time.monotonic() - started
                tracker.close(disconnect_time + tracker.merge_gap)
                tracker.pending_label, tracker.pending_count = None, 0
                pending_alarm_media = None
                stream_generation += 1
                if writer is not None:
                    writer.release()
                    print(f"output video segment closed: {writer_path}", flush=True)
                    writer = None
                    writer_size = None
                    writer_path = None
                    writer_part += 1
                print("RTSP frame read failed; connection will be reopened", flush=True)
                capture.release()
                continue
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
                print("RTSP returned an invalid frame; dropping it", flush=True)
                continue
            current_shape = frame.shape[:2]
            if buffer_frame_shape is None:
                buffer_frame_shape = current_shape
            elif current_shape != buffer_frame_shape:
                print(
                    f"RTSP frame size changed from {buffer_frame_shape[1]}x{buffer_frame_shape[0]} "
                    f"to {current_shape[1]}x{current_shape[0]}; resetting inference buffer",
                    flush=True,
                )
                buffer.clear()
                probability_history.clear()
                last_inference = -1e9
                prediction = ("warming_up", 0.0)
                tracker.pending_label, tracker.pending_count = None, 0
                pending_alarm_media = None
                stream_generation += 1
                if not is_file and writer is not None:
                    writer.release()
                    print(f"output video segment closed after frame-size change: {writer_path}", flush=True)
                    writer = None
                    writer_size = None
                    writer_path = None
                    writer_part += 1
                buffer_frame_shape = current_shape
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if is_file else time.monotonic() - started

            # Collect a completed live inference without blocking frame capture.
            if inference_future is not None and inference_future.done():
                result_timestamp, result_generation, result_frame, result_clip, result_event_time = inference_context
                try:
                    probabilities = inference_future.result()
                except Exception as exc:
                    print(f"background inference failed: {exc}", flush=True)
                    probabilities = None
                inference_future = None
                inference_context = None
                if probabilities is not None and result_generation == stream_generation:
                    probability_history.append(probabilities)
                    smoothed = np.mean(probability_history, axis=0)
                    index = int(smoothed.argmax())
                    prediction = classes[index], float(smoothed[index])
                    previous_alarm = tracker.active["label"] if tracker.active else None
                    positive = (
                        prediction[0] in tracker.thresholds
                        and prediction[1] >= float(tracker.thresholds[prediction[0]])
                    )
                    if positive and (tracker.active is None or tracker.active["label"] != prediction[0]):
                        if tracker.pending_label != prediction[0]:
                            pending_alarm_media = (
                                prediction[0],
                                [item.copy() for item in result_clip],
                                result_event_time,
                            )
                    elif not positive and tracker.active is None:
                        pending_alarm_media = None
                    active_alarm = tracker.update(prediction[0], prediction[1], result_timestamp, stride)
                    if alarm_executor and active_alarm and active_alarm != previous_alarm:
                        media_frames = result_clip
                        media_event_time = result_event_time
                        if pending_alarm_media and pending_alarm_media[0] == active_alarm:
                            _, media_frames, media_event_time = pending_alarm_media
                        if alarm_output_dir:
                            alarm_executor.submit(
                                save_alarm_video,
                                alarm_output_dir,
                                active_alarm,
                                media_frames,
                                media_event_time,
                                frame_count / duration,
                                args.alarm_video_width,
                            )
                        if args.alarm_callback_url:
                            alarm_executor.submit(
                                post_alarm,
                                args.alarm_callback_url,
                                args.device_id,
                                active_alarm,
                                media_frames,
                                media_event_time,
                                args.alarm_media,
                                frame_count / duration,
                                args.alarm_video_width,
                            )
                        pending_alarm_media = None
                    windows += 1
                    scores = " ".join(f"{name}={smoothed[i]:.3f}" for i, name in enumerate(classes))
                    inference_lag = max(0.0, (time.monotonic() - started) - result_timestamp)
                    print(
                        f"t={result_timestamp:8.3f}s pred={prediction[0]:>12s} "
                        f"conf={prediction[1]:.3f} alarm={active_alarm or '-':>12s} "
                        f"inference_lag={inference_lag:.3f}s {scores}",
                        flush=True,
                    )

            buffer.append((timestamp, frame.copy()))
            while buffer and timestamp - buffer[0][0] > duration + 1.0:
                buffer.popleft()

            full_clip_ready = buffer and timestamp - buffer[0][0] >= duration - (1.0 / max(fps, 1.0))
            if full_clip_ready and timestamp - last_inference >= stride:
                clip = sample_clip(buffer, timestamp, duration, frame_count)
                if is_file:
                    probabilities = infer_clip(model, clip, cfg, device)
                    probability_history.append(probabilities)
                    smoothed = np.mean(probability_history, axis=0)
                    index = int(smoothed.argmax())
                    prediction = classes[index], float(smoothed[index])
                    previous_alarm = tracker.active["label"] if tracker.active else None
                    positive = (
                        prediction[0] in tracker.thresholds
                        and prediction[1] >= float(tracker.thresholds[prediction[0]])
                    )
                    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if positive and (tracker.active is None or tracker.active["label"] != prediction[0]):
                        if tracker.pending_label != prediction[0]:
                            pending_alarm_media = (
                                prediction[0],
                                [item.copy() for item in clip],
                                event_time,
                            )
                    elif not positive and tracker.active is None:
                        pending_alarm_media = None
                    active_alarm = tracker.update(prediction[0], prediction[1], timestamp, stride)
                    if alarm_executor and active_alarm and active_alarm != previous_alarm:
                        media_frames = clip
                        media_event_time = event_time
                        if pending_alarm_media and pending_alarm_media[0] == active_alarm:
                            _, media_frames, media_event_time = pending_alarm_media
                        if alarm_output_dir:
                            alarm_executor.submit(
                                save_alarm_video,
                                alarm_output_dir,
                                active_alarm,
                                media_frames,
                                media_event_time,
                                frame_count / duration,
                                args.alarm_video_width,
                            )
                        if args.alarm_callback_url:
                            alarm_executor.submit(
                                post_alarm,
                                args.alarm_callback_url,
                                args.device_id,
                                active_alarm,
                                media_frames,
                                media_event_time,
                                args.alarm_media,
                                frame_count / duration,
                                args.alarm_video_width,
                            )
                        pending_alarm_media = None
                    windows += 1
                    scores = " ".join(f"{name}={smoothed[i]:.3f}" for i, name in enumerate(classes))
                    print(f"t={timestamp:8.3f}s pred={prediction[0]:>12s} conf={prediction[1]:.3f} alarm={active_alarm or '-':>12s} {scores}", flush=True)
                    last_inference = timestamp
                elif inference_future is None:
                    inference_future = inference_executor.submit(infer_clip, model, clip, cfg, device)
                    inference_context = (
                        timestamp,
                        stream_generation,
                        frame.copy(),
                        clip,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    last_inference = timestamp

            active = tracker.active["label"] if tracker.active else None
            color = (0, 0, 255) if active else (0, 200, 0)
            cv2.putText(frame, f"X3D: {prediction[0]} {prediction[1]:.2f}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"ALARM: {active or '-'}", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if args.output_video:
                if writer is None:
                    args.output_video.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer_size = (width, height)
                    writer_path = output_segment_path(args.output_video, writer_part)
                    writer = cv2.VideoWriter(
                        str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, writer_size
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot open output video writer: {writer_path}")
                    print(
                        f"output video segment opened: {writer_path} "
                        f"size={writer_size[0]}x{writer_size[1]} fps={fps:.3f}",
                        flush=True,
                    )
                output_frame = frame
                if (frame.shape[1], frame.shape[0]) != writer_size:
                    output_frame = cv2.resize(frame, writer_size, interpolation=cv2.INTER_LINEAR)
                writer.write(output_frame)
            if not args.headless:
                cv2.imshow("X3D Industrial Anomaly", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.max_windows and windows >= args.max_windows:
                break
    except KeyboardInterrupt:
        pass
    finally:
        final_time = buffer[-1][0] if buffer else 0.0
        tracker.close(final_time + tracker.merge_gap)
        capture.release()
        if writer:
            writer.release()
        if inference_executor:
            inference_executor.shutdown(wait=True)
        if alarm_executor:
            alarm_executor.shutdown(wait=True)
        cv2.destroyAllWindows()
        print(f"finished: inference_windows={windows}, event_log={tracker.output_path}")


if __name__ == "__main__":
    main()
