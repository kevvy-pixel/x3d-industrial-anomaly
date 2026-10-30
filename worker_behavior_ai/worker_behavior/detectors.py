"""前置检测模型：板 / 手 / 人体骨架 —— 权重与推理封装。

三个检测器是整条流水线的基础，可单独使用（调试、可视化、二次开发）：

    from worker_behavior.detectors import DetectorSuite
    det = DetectorSuite()
    board, hands, kp = det.detect(frame_bgr)
    # board : ((x1,y1,x2,y2), conf) | None
    # hands : [((x1,y1,x2,y2), conf), ...]   按 x 排序，最多 2 只
    # kp    : (17,3) float32 像素坐标+置信度 | None

也可以直接跑可视化（把三类检测画到视频/图片上）：

    python -m worker_behavior.detectors --source demo.mp4 --out det.mp4
    python -m worker_behavior.detectors --source frame.jpg --out det.jpg
"""
import argparse
import os

import cv2
import numpy as np

from . import config as C

# COCO 17 关键点骨架连线（右肩5-右肘7-右腕9、左肩6-左肘8-左腕10 …）
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

COLOR_BOARD = (0, 165, 255)   # BGR 橙
COLOR_HAND = (0, 0, 255)      # 红
COLOR_POSE = (0, 255, 0)      # 绿


class _Yolo:
    def __init__(self, ckpt):
        from ultralytics import YOLO
        self.model = YOLO(ckpt)

    def _one(self, frame, imgsz, conf):
        return self.model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]


class BoardDetector(_Yolo):
    """板检测：取置信度最高的一个实例。"""

    def __init__(self, ckpt=None, imgsz=None, conf=None):
        super().__init__(ckpt or C.BOARD_CKPT)
        self.imgsz = imgsz or C.BOARD_IMGSZ
        self.conf = conf if conf is not None else C.CONF

    def detect(self, frame):
        r = self._one(frame, self.imgsz, self.conf)
        if r.boxes is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.cpu().numpy().argmax())
        return (r.boxes.xyxy.cpu().numpy()[i].copy(),
                float(r.boxes.conf.cpu().numpy()[i]))


class HandDetector(_Yolo):
    """手检测：返回最多 2 只（按 x 排序，便于区分左右）。"""

    def __init__(self, ckpt=None, imgsz=None, conf=None, max_hands=2):
        super().__init__(ckpt or C.HAND_CKPT)
        self.imgsz = imgsz or C.HAND_IMGSZ
        self.conf = conf if conf is not None else 0.25
        self.max_hands = max_hands

    def detect(self, frame):
        r = self._one(frame, self.imgsz, self.conf)
        if r.boxes is None or len(r.boxes) == 0:
            return []
        bb = r.boxes.xyxy.cpu().numpy()
        cc = r.boxes.conf.cpu().numpy()
        order = np.argsort((bb[:, 0] + bb[:, 2]) / 2)[:self.max_hands]
        return [(bb[i].copy(), float(cc[i])) for i in order]


class PoseDetector(_Yolo):
    """人体骨架：yolo26l-pose，17 个 COCO 关键点（像素坐标 + 置信度）。"""

    def __init__(self, ckpt=None, imgsz=None, conf=None):
        super().__init__(ckpt or C.POSE_CKPT)
        self.imgsz = imgsz or C.POSE_IMGSZ
        self.conf = conf if conf is not None else C.CONF

    def detect(self, frame):
        r = self._one(frame, self.imgsz, self.conf)
        if r.keypoints is None or len(r.keypoints.data) == 0:
            return None
        return r.keypoints.data[0].cpu().numpy()      # (17,3)


class DetectorSuite:
    """一次调用拿到板 / 手 / 骨架（流式推理即用这个）。"""

    def __init__(self, board_conf=None, hand_conf=None, pose_conf=None):
        self.board = BoardDetector(conf=board_conf)
        self.hand = HandDetector(conf=hand_conf)
        self.pose = PoseDetector(conf=pose_conf)

    def detect(self, frame):
        return (self.board.detect(frame),
                self.hand.detect(frame),
                self.pose.detect(frame))

    def draw(self, frame, board=None, hands=None, kp=None):
        """把检测结果画到帧上（原地修改并返回）。"""
        if board is not None:
            b, c = board
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOARD, 3)
            cv2.putText(frame, f"board {c:.2f}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BOARD, 2)
        for i, (b, c) in enumerate(hands or []):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HAND, 3)
            cv2.putText(frame, f"hand{i + 1} {c:.2f}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_HAND, 2)
        if kp is not None:
            for a, b in SKELETON:
                if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                    cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[b, 0]), int(kp[b, 1])), COLOR_POSE, 2)
            for pt in kp:
                if pt[2] > 0.4:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
        return frame


def main():
    ap = argparse.ArgumentParser(description="板/手/骨架 检测可视化")
    ap.add_argument("--source", required=True, help="视频文件或图片")
    ap.add_argument("--out", required=True, help="输出路径（.mp4 或 .jpg）")
    ap.add_argument("--every", type=int, default=3, help="视频时每 N 帧处理一次")
    a = ap.parse_args()

    suite = DetectorSuite()
    img_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if a.source.lower().endswith(img_ext):
        f = cv2.imread(a.source)
        if f is None:
            raise SystemExit(f"读不到图片: {a.source}")
        board, hands, kp = suite.detect(f)
        suite.draw(f, board, hands, kp)
        cv2.imwrite(a.out, f)
        print(f"{a.out}  board={'Y' if board else 'N'} hands={len(hands)} pose={'Y' if kp is not None else 'N'}")
        return

    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / a.every, (W, H))
    n = nb = nh = np_ = 0
    c = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if c % a.every:
            c += 1
            continue
        board, hands, kp = suite.detect(f)
        nb += board is not None
        nh += len(hands)
        np_ += kp is not None
        n += 1
        w.write(suite.draw(f, board, hands, kp))
        c += 1
    cap.release(); w.release()
    print(f"{a.out}  处理 {n} 帧 | 板检出 {nb} ({nb/max(1,n):.2f}) | "
          f"手 {nh} 个 | 骨架 {np_} ({np_/max(1,n):.2f})")


if __name__ == "__main__":
    main()
