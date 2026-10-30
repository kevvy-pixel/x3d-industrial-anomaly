"""Shared per-frame feature builder (board + hands + pose -> fixed vector).

Single source of truth used by BOTH offline training and online inference, so
the deployed model always sees exactly the features it was trained on.

Layout (81 dims, all normalised by frame size / shoulder width so they transfer
across camera distance):
    pose       17*(x,y,conf)                                    0..50
    pose_present, shoulder_w                                    51,52
    board      present,cx,cy,w,h,conf                           53..58
    hands      up to 2 sorted by x: present,cx,cy,w,h,conf      59..70
    hand->board dist, inside flag (x2)                          71..74
    hand->nose dist (x2)                                        75,76
    board_bottom_vs_ankle, board_vy, n_hands_on_board, board_area 77..80
"""
import numpy as np

D = 81

FEATURE_NAMES = (
    [f"pose{i}_{a}" for i in range(17) for a in ("x", "y", "c")]
    + ["pose_present", "shoulder_w"]
    + ["board_present", "board_cx", "board_cy", "board_w", "board_h", "board_conf"]
    + [f"hand{k}_{a}" for k in range(2) for a in ("present", "cx", "cy", "w", "h", "conf")]
    + [f"hb_dist{k}" for k in range(2)] + [f"hb_inside{k}" for k in range(2)]
    + [f"hn_dist{k}" for k in range(2)]
    + ["board_bottom_vs_ankle", "board_vy", "n_hands_on_board", "board_area"]
)


def frame_features(board, hands, kp, W, H, prev_bottom=None, dt=1.0 / 24.0):
    """Build the 81-dim feature vector for one frame.

    board : (xyxy, conf) or None
    hands : list of (xyxy, conf), up to 2 used
    kp    : (17,3) pose keypoints in PIXELS + conf, or None
    prev_bottom : previous board bottom (normalised y) or None
    Returns (x, board_bottom) so the caller can feed prev_bottom next frame.
    """
    x = np.zeros(D, np.float32)
    sh = 0.0
    nose = None
    if kp is not None:
        x[0:51] = np.concatenate([kp[:, 0] / W, kp[:, 1] / H, kp[:, 2]])
        x[51] = 1.0
        if kp[5, 2] > 0.3 and kp[6, 2] > 0.3:
            sh = float(np.hypot(*(kp[5][:2] - kp[6][:2]))) / W
        x[52] = sh
        if kp[0, 2] > 0.35:
            nose = (kp[0, 0] / W, kp[0, 1] / H)

    if board is not None:
        b, cf = board
        x[53] = 1.0
        x[54] = (b[0] + b[2]) / 2 / W
        x[55] = (b[1] + b[3]) / 2 / H
        x[56] = (b[2] - b[0]) / W
        x[57] = (b[3] - b[1]) / H
        x[58] = cf
        x[80] = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)

    hands = hands[:2]
    for k, (bb, cf) in enumerate(hands):
        o = 59 + k * 6
        x[o:o + 6] = [1.0, (bb[0] + bb[2]) / 2 / W, (bb[1] + bb[3]) / 2 / H,
                      (bb[2] - bb[0]) / W, (bb[3] - bb[1]) / H, cf]

    shn = sh if sh > 1e-6 else 0.2
    n_on = 0
    for k in range(2):
        o = 59 + k * 6
        if x[o] < 0.5:
            continue
        hcx, hcy = x[o + 1], x[o + 2]
        if board is not None:
            b = board[0]
            bcx, bcy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            x[71 + k] = float(np.hypot(hcx - bcx, hcy - bcy)) / shn
            inside = (b[0] / W <= hcx <= b[2] / W) and (b[1] / H <= hcy <= b[3] / H)
            x[73 + k] = 1.0 if inside else 0.0
            if inside:
                n_on += 1
        if nose is not None:
            x[75 + k] = float(np.hypot(hcx - nose[0], hcy - nose[1])) / shn
    x[79] = n_on

    bottom = None
    if board is not None:
        if kp is not None:
            ankles = [k[1] for k in (kp[15], kp[16]) if k[2] > 0.3]
            if ankles:
                x[77] = (board[0][3] / H) - (min(ankles) / H)
        bottom = board[0][3] / H
        if prev_bottom is not None:
            x[78] = (bottom - prev_bottom) / max(dt, 1e-3)
    return x, bottom


def contextualise(seq, i, ctx):
    """Concatenate frames i-ctx..i+ctx (edge-replicated) -> flat vector."""
    if ctx <= 0:
        return seq[i]
    lo, hi = max(0, i - ctx), min(len(seq), i + ctx + 1)
    parts = [seq[lo:hi]]
    pad_lo, pad_hi = ctx - (i - lo), ctx - (hi - i - 1)
    if pad_lo > 0:
        parts.insert(0, np.repeat(seq[lo:lo + 1], pad_lo, axis=0))
    if pad_hi > 0:
        parts.append(np.repeat(seq[hi - 1:hi], pad_hi, axis=0))
    return np.concatenate(parts, axis=0).reshape(-1)
