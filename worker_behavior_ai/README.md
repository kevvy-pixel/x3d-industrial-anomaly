# worker_behavior_ai — 车间工人行为实时识别

从实时视频流判定 4 类状态：**normal / single_hand（单手拿板）/ touch_face（摸脸）/ board_drop（板掉落）**。

轻量、无 SAM3，单卡实时可用。

## 方法

```
输入视频流
   ↓ 抽帧 (~8-10 fps)
① 感知层：3 个 YOLO（逐帧）
     board.pt   板检测      yolo26n
     hand.pt    手检测      yolo26n（18 段人工标注）
     pose.pt    人体骨架    yolo26l-pose (COCO 17 点)
   ↓
② 特征层 worker_behavior/features.py
     81 维/帧：姿态(17×x,y,conf) + 板框 + 手框(×2)
     + 关系量：手→板距离/是否在板内、手→鼻距离、
                板底 vs 脚踝、板纵向速度、手在板数量
     （全部按肩宽/画面归一化，抗机位与距离变化）
   ↓ 拼接 t±2 帧 → 405 维
③ 分类层 worker_behavior/temporal.py
     小型 MLP（256 隐层）→ 每帧 4 类概率
   ↓
④ 状态机层
     由训练集学到的状态转移矩阵 + Viterbi 解码（self-boost=30）
     → 状态持续、只走合法转移、抑制瞬时抖动
   ↓
⑤ 事件层：丢弃短于 0.5s 的事件（config.EVENT_MIN_SEC）
```

**为什么不是纯阈值规则**：规则要针对每个机位手调阈值，且对"手贴脸导致手检测漏检"
这类情况无解；学习模型同时利用骨架几何，实测在 126 号片段上规则 **0 检出**、
学习模型 **3 个事件全部与标注吻合**。

## 评测

数据：`data_full` 的 4 个数据源（data1–data4）共 1065 段，按 **(数据源 × 类别)** 分层划分
→ **train 851 段 / val 214 段**（每个源都出现在两侧）。

**帧级 macro-F1（val 214 段，Viterbi 解码）= 0.783**
（同一划分下，只在单一数据源上训练的旧模型只有 **0.485**）

**事件级**（与标注时间段按 IoU≥0.3 匹配，最短事件 0.5s）：

| 行为 | GT 事件 | P | R | F1 |
|---|---|---|---|---|
| single_hand | 37 | 0.75 | 0.97 | **0.85** |
| touch_face | 20 | 0.78 | 0.90 | **0.84** |
| board_drop | 16 | 1.00 | 0.88 | **0.93** |

**分数据源**：

| 源 | single_hand | touch_face | board_drop | 说明 |
|---|---|---|---|---|
| data1 | 0.67 | 0.82 | 0.92 | 最早一批数据，机位不同，最弱 |
| data2 | — | — | — | 纯 normal（65 段）：**零误报** |
| data3 | 0.33* | —* | 1.00* | 异常样本极少(3/0/1)，*数字仅为参考 |
| data4 | 0.83 | 0.88 | 0.93 | 最新一批，样本最多 |

## 前置检测模型（板 / 手 / 骨架）

三个检测器已随仓库封装好，可**独立使用**：

```python
from worker_behavior.detectors import DetectorSuite
det = DetectorSuite()
board, hands, kp = det.detect(frame_bgr)
# board : ((x1,y1,x2,y2), conf) | None           板，取置信度最高实例
# hands : [((x1,y1,x2,y2), conf), ...]           手，按 x 排序最多 2 只
# kp    : (17,3) float32 像素坐标+置信度 | None   人体骨架(COCO 17)
```

单独跑可视化（把三类检测画到视频/图片上）：

```bash
python -m worker_behavior.detectors --source demo.mp4 --out det.mp4 --every 3
python -m worker_behavior.detectors --source frame.jpg --out det.jpg
```
输出示例：`处理 90 帧 | 板检出 38 (0.42) | 手 165 个 | 骨架 90 (1.00)`

也可只实例化某一个：`BoardDetector()` / `HandDetector()` / `PoseDetector()`。

## 快速开始

```bash
pip install -r requirements.txt

python -m worker_behavior.stream --source 0                      # 摄像头
python -m worker_behavior.stream --source rtsp://user:pw@ip/8545/1   # RTSP

# 视频文件：输出标注视频 + 事件 json + 事件起始截图
python -m worker_behavior.stream \
    --source demo.mp4 --out annotated.mp4 --json events.json --snap-dir snaps
```

事件 json：`[{"label":"board_drop","t0":7.8,"t1":8.5,"snap":"..."}]`

编程调用：
```python
from worker_behavior import run
events = run("clip.mp4", json_path="events.json")
```

### 两种模式
`--mode learn`（默认）走 MLP + Viterbi；`--mode rule` 走 `engine.py` 的阈值状态机（兜底/对照）。

### 主要参数（`worker_behavior/config.py`，均可用环境变量覆盖）
| 变量 | 说明 |
|---|---|
| `MODEL_DIR` | 权重目录（默认 `./models`） |
| `WORKER_AI_MODE` | `learn` / `rule` |
| `SAMPLE_EVERY` | 每 N 帧处理一次 |
| `EVENT_MIN_SEC` | 丢弃短于该时长的事件（默认 0.5s） |

## 训练 / 重新训练

全链路脚本在 `tools/`，见 `tools/README.md`。典型流程：

```bash
# 检测器
python tools/01_make_board_dataset.py --out datasets/board_detect && python tools/02_train_board.py
python tools/05_extract_pose_features.py --out ../feats_pose_yolo
python tools/03_make_hand_dataset.py --out datasets/hand_detect && python tools/04_train_hand.py

# 行为模型（需要带 split 字段的 manifest 与已抽好的特征）
python tools/10_extract_features_v2.py --manifest ../behavior_manifest_full.json --out ../temporal/feats_full
python tools/15_train_on_split.py \
    --manifest ../behavior_manifest_full.json --feats ../temporal/feats_full \
    --out models/temporal_clf.pkl --self-boost 30      # 训+分源评测+导出
```

`tools/11_train_temporal.py` 用于在同一数据上做按 clip 的 K 折交叉验证（模型选型用）。

## 目录

```
worker_behavior_ai/
├── worker_behavior/        推理包
│   ├── detectors.py        前置检测：板/手/骨架（含独立可视化 CLI）
│   ├── features.py         81 维共享特征（训练=推理）
│   ├── temporal.py         MLP + 转移矩阵 + Viterbi
│   ├── engine.py           规则状态机
│   ├── stream.py           实时/replay 循环
│   └── config.py           路径与参数
├── models/                 board.pt · hand.pt · pose.pt · temporal_clf.pkl
├── tools/                  数据集/训练/特征/评测/导出
└── requirements.txt
```

## 已知限制
- 评测集覆盖 4 个数据源，但**都是同一类车间**；换产线/机位仍需补数据微调
- `single_hand`/`touch_face` 偏"宁可错报"（P≈0.75/0.78，R≈0.97/0.90）；`board_drop` 已较平衡
- `data3` 异常样本过少（3/0/1 个事件），无法有效验证该源
- 多人场景按检出置信度取 1 人 2 手，多人并发未专门处理
