# X3D-M 四分类数据与训练实验记录

## 数据规则

- 原始数据：`C:\Users\27356\Music\Desktop\工厂` 下的 data1/data2/data3/data4。
- 类别顺序：`normal=0, single_hand=1, board_drop=2, touch_face=3`。`no_gloves` 完全排除；`single_hand_or_drag` 按 `single_hand` 归并。
- 先以原始视频为单位固定切分，再生成样本。同一原视频（含所有 annotation 和窗口）只会出现在 train 或 val 一侧。
- 固定随机种子：20260911；按每个视频的标签组合分层，约 80% train / 20% val。验证清单见 `val_sources.csv`，后续不得用于训练、调参或重采样。
- 以实测 FPS 就近归入 16/24/30 档，对应窗口 1.0/0.8/0.7 秒；每个窗口在读取时均匀抽取 16 帧。
- 从 annotation 起点连续生成不重叠完整窗口，尾部余段丢弃。仅当整个片段没有完整窗口且长度大于 0.9 秒时，围绕该片段补足一个窗口；小于等于 0.9 秒不生成样本。normal 视频整段采用同样规则。
- 单标签冲突控制：若一个候选窗口有至少 50% 时长同时落入另一目标类别的 annotation，则该窗口作为歧义样本排除，避免同一画面给四分类 softmax 提供互相矛盾的标签。

## 固定切分与样本统计

- 原始视频总数：1065；train 852，val 213。
- 样本总数：9904；train 7921，val 1983。

| split | normal | single_hand | board_drop | touch_face | 合计 |
|---|---:|---:|---:|---:|---:|
| train | 6967 | 542 | 91 | 321 | 7921 |
| val | 1736 | 126 | 28 | 93 | 1983 |
| total | 8703 | 668 | 119 | 414 | 9904 |

## 排除与审计

- `ambiguous_board_drop`：10 个 annotation/视频区间未生成样本。
- `ambiguous_single_hand`：25 个 annotation/视频区间未生成样本。
- `ambiguous_touch_face`：35 个 annotation/视频区间未生成样本。
- `short_board_drop`：12 个 annotation/视频区间未生成样本。
- `short_normal`：1 个 annotation/视频区间未生成样本。
- `short_single_hand`：11 个 annotation/视频区间未生成样本。
- `short_touch_face`：5 个 annotation/视频区间未生成样本。
- 数据读取警告：0 条，详见 `audit_warnings.txt`。

## 训练实验设置

- 架构：PyTorchVideo X3D-M，Kinetics-400 预训练；最后分类层替换为 4 类。输入为 RGB 16 帧、224×224。
- 训练增强：保持宽高比缩放、随机裁剪、随机水平翻转；验证使用中心裁剪。归一化采用 Kinetics 均值/标准差。
- 类别均衡：训练每个 epoch 对四类使用相同配额，配额取训练集中最少类的样本数；各类均无放回采样，多数类每轮轮换子集。因此不会因某类原始样本多而在单轮多采。验证集保持原始自然分布，不做均衡采样。
- 优化：AdamW、交叉熵（label smoothing 0.1）、AMP、余弦学习率；按验证 macro-F1 保存最佳权重。具体超参数与每轮指标会由训练脚本追加到本文件，并另存 `config.json`、`history.csv`、`best_metrics.json`。
- 防泄漏约束：训练脚本启动时会再次断言 train/val 的 `source_id` 交集为空。

## 实际训练结果

- 完成轮数：30；batch size：8；梯度累积：2。
- 初始学习率：0.0001；weight decay：0.0001。
- 最佳 epoch：28；验证 loss：0.888699；accuracy：0.9259；macro-F1：0.8029。

### 最佳模型逐类指标

| 类别 | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| normal | 0.9957 | 0.9263 | 0.9597 | 1736 |
| single_hand | 0.5616 | 0.9762 | 0.7130 | 126 |
| board_drop | 0.6970 | 0.8214 | 0.7541 | 28 |
| touch_face | 0.7069 | 0.8817 | 0.7847 | 93 |

### 混淆矩阵（行是真值，列是预测）

| true\pred | normal | single_hand | board_drop | touch_face |
|---|---:|---:|---:|---:|
| normal | 1608 | 85 | 9 | 34 |
| single_hand | 2 | 123 | 1 | 0 |
| board_drop | 0 | 5 | 23 | 0 |
| touch_face | 5 | 6 | 0 | 82 |
