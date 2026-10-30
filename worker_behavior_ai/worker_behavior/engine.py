"""worker_ai.engine: per-frame perception + event state machine.

Per frame runs board / hand / pose (ultralytics), derives cues, and feeds a
latched event state machine emitting: normal | single_hand | board_drop | touch_face.
No SAM3 / no tracking beyond cheap box carry-over.
"""
import numpy as np

from . import config as C


class BehaviorEngine:
    def __init__(self, conf=C.CONF, board_imgsz=C.BOARD_IMGSZ,
                 hand_imgsz=C.HAND_IMGSZ, pose_imgsz=C.POSE_IMGSZ):
        from ultralytics import YOLO
        self.det = YOLO(C.BOARD_CKPT)
        self.hand = YOLO(C.HAND_CKPT)
        self.pose = YOLO(C.POSE_CKPT)
        self.conf = conf
        self.board_imgsz, self.hand_imgsz, self.pose_imgsz = board_imgsz, hand_imgsz, pose_imgsz
        self._prev_bot = None
        self._lost = 0
        self.reset()

    def reset(self):
        self._prev_bot = None
        self._lost = 0
        self.st = dict(low_run=0, single_run=0, hand_on_run=0, gone_run=0,
                       fall_off_run=0, face_run=0, event="normal")

    # ---------- cues ----------
    def step(self, frame, dt=1.0 / 24.0):
        H, W = frame.shape[:2]
        # board
        rd = self.det.predict(frame, imgsz=self.board_imgsz, conf=self.conf, verbose=False)[0]
        board = None
        if rd.boxes is not None and len(rd.boxes):
            i = int(rd.boxes.conf.cpu().numpy().argmax())
            board = rd.boxes.xyxy.cpu().numpy()[i].copy()
        # pose
        rp = self.pose.predict(frame, imgsz=self.pose_imgsz, conf=self.conf, verbose=False)[0]
        kp = None
        if rp.keypoints is not None and len(rp.keypoints.data):
            kp = rp.keypoints.data[0].cpu().numpy()

        cu = dict(board=False, low=False, Lo=False, Ro=False, released=False,
                  fall=False, face=False, nbo=0)
        if kp is not None:
            ankles = [k for k in (kp[15], kp[16]) if k[2] > 0.3]
            floorY = (min(a[1] for a in ankles) - 0.05 * H) if ankles else 0.93 * H
            for side, j in (("L", 9), ("R", 10)):
                if kp[j][2] <= 0.3:
                    continue
                w = kp[j]
                if board is not None:
                    x1, y1, x2, y2 = board
                    padx, pady = (x2 - x1) * 0.4, (y2 - y1) * 0.4
                    if x1 - padx <= w[0] <= x2 + padx and y1 - pady <= w[1] <= y2 + pady:
                        cu["Lo" if side == "L" else "Ro"] = True
            if board is not None:
                cu["low"] = board[3] >= floorY
                any_wrist = any(kp[j][2] > 0.3 for j in (9, 10))
                cu["released"] = bool(cu["low"] and not (cu["Lo"] or cu["Ro"]) and any_wrist)
            if board is not None and self._prev_bot is not None:
                v = (float(board[3]) - float(self._prev_bot)) / float(max(dt, 1e-3)) / float(H)
                if v > C.FALL_V:
                    cu["fall"] = True
        # hand boxes
        rh = self.hand.predict(frame, imgsz=self.hand_imgsz, conf=0.25, verbose=False)[0]
        hb = rh.boxes.xyxy.cpu().numpy() if (rh.boxes is not None and len(rh.boxes)) else []
        if board is not None:
            x1, y1, x2, y2 = board
            padx, pady = (x2 - x1) * 0.35, (y2 - y1) * 0.35
            cu["nbo"] = sum(1 for b in hb
                            if not (b[2] < x1 - padx or b[0] > x2 + padx
                                    or b[3] < y1 - pady or b[1] > y2 + pady))
        if kp is not None and kp[0][2] > 0.35:
            n = kp[0]
            sh = (np.hypot(*(kp[5][:2] - kp[6][:2]))
                  if kp[5][2] > 0.3 and kp[6][2] > 0.3 else max(W, H) * 0.2)
            hr = (n[0] - 0.65 * sh, n[1] - 0.65 * sh, n[0] + 0.65 * sh, n[1] + 0.65 * sh)
            cu["face"] = any(not (b[2] < hr[0] or b[0] > hr[2] or b[3] < hr[1] or b[1] > hr[3])
                             for b in hb)
        self._prev_bot = None if board is None else float(board[3])
        cu["board"] = board is not None
        return cu

    # ---------- state machine ----------
    def classify(self, cu, fps):
        st = self.st
        lo, ro, board = cu["Lo"], cu["Ro"], cu["board"]
        any_hand = lo or ro or cu["nbo"] > 0
        st["low_run"] = st["low_run"] + 1 if (cu["released"] and board) else 0
        st["single_run"] = st["single_run"] + 1 if (board and cu["nbo"] == 1 and not cu["released"]) else 0
        st["hand_on_run"] = st["hand_on_run"] + 1 if any_hand else 0
        st["gone_run"] = st["gone_run"] + 1 if not board else 0
        st["fall_off_run"] = st["fall_off_run"] + 1 if (board and cu["fall"] and not any_hand) else 0
        st["face_run"] = st["face_run"] + 1 if cu["face"] else 0
        drop_start = (st["low_run"] >= C.DROP_CONFIRM) or (st["fall_off_run"] >= 2)
        if st["event"] == "board_drop":
            if st["hand_on_run"] >= C.PICKUP_SEC * fps or st["gone_run"] > C.CARRY_FRAMES + 2:
                st["event"] = "normal"
            return st["event"]
        if st["event"] == "touch_face":
            if st["face_run"] == 0:
                st["event"] = "normal"
            return st["event"]
        if drop_start:
            st["event"] = "board_drop"
        elif st["face_run"] >= C.FACE_CONFIRM:
            st["event"] = "touch_face"
        elif st["single_run"] >= C.SINGLE_CONFIRM:
            st["event"] = "single_hand"
        else:
            st["event"] = "normal"
        return st["event"]
