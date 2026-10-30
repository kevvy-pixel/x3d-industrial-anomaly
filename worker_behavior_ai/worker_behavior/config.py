"""worker_ai config: model paths + event thresholds (all env-overridable)."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
# weights shipped inside this repo (worker_ai/weights); override dir via MODEL_DIR
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(REPO, "models"))

def _p(name):
    p = os.path.join(MODEL_DIR, name)
    return p

BOARD_CKPT = _p("board.pt")
HAND_CKPT = _p("hand.pt")
POSE_CKPT = _p("pose.pt")

# model inference
BOARD_IMGSZ = 1024
HAND_IMGSZ = 640
POSE_IMGSZ = 640
CONF = 0.3

# event state machine
CARRY_FRAMES = 6
FALL_V = 0.22          # board bottom downward speed (frac H / s)
DROP_CONFIRM = 3       # consecutive low+released frames -> drop
SINGLE_CONFIRM = 4     # consecutive single-hand frames
PICKUP_SEC = 0.5       # hand back on board this long -> drop ends
FACE_CONFIRM = 3

# stream
MODE = os.environ.get("WORKER_AI_MODE", "learn")   # "learn" (temporal model) | "rule"
SAMPLE_EVERY = 2       # process every N-th frame by default
EVENT_MIN_SEC = 0.5    # ignore events shorter than this

# event -> color / priority
CLASSES = ("normal", "single_hand", "touch_face", "board_drop")  # matches the trained model
