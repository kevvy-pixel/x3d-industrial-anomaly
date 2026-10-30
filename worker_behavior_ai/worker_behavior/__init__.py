"""worker_behavior — 车间工人行为实时识别.

4 类状态：normal / single_hand / touch_face / board_drop。
感知(yolo26 板/手/骨架) → 81维共享特征 → MLP 逐帧分类 → Viterbi 状态机解码。
详见 README.md。
"""
from .engine import BehaviorEngine          # noqa: F401
from .temporal import TemporalModel         # noqa: F401
from .stream import run                     # noqa: F401
from . import config                        # noqa: F401

__all__ = ["BehaviorEngine", "TemporalModel", "run", "config"]
__version__ = "1.0.0"
