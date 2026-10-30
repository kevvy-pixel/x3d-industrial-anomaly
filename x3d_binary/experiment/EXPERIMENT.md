# X3D-M normal / touch_face 二分类实验

## 冻结划分与防泄漏

- 完全复用四分类实验的源视频级划分：`D:\CondaData\factory_x3d_m\experiment\source_split.csv`，没有重新随机划分。
- 划分文件 SHA-256：`7cdccd3c055b49fb4bad9f9585e3592d35ace3f8d87b7204a9e2cd4f21802059`。
- 原样本清单 SHA-256：`10e44ff367f417d9f8095b3838bc17d044f31f2ef56630274ecd3bb6d7d57620`。
- 唯一源视频：train 849，val 213；交集为 0。
- 验证集只用于每轮评估、最佳 epoch 选择和阈值分析，绝不进入梯度更新。

## 二分类定义

- `touch_face`：原四分类清单中的 touch_face 异常片段。
- `normal`：原 normal、single_hand、board_drop 样本。后两类作为不摸脸硬负样本，以降低其他手部动作被误报为摸脸的风险。
- `nogloves` 仍不使用。混合异常视频沿用原 JSON 注释窗口及其源视频 split。

## 样本统计

| split | normal（含硬负样本） | touch_face | 合计 |
|---|---:|---:|---:|
| train | 7600 | 321 | 7921 |
| val | 1890 | 93 | 1983 |

## 采样与训练约束

- 每个样本固定 16 帧；约 16 FPS 使用 1.0 秒窗口、24 FPS 使用 0.8 秒、30 FPS 使用 0.7 秒；沿用原清单的 `start_sec/end_sec`。
- 每个训练 epoch 两类严格等量、无放回采样；配额为较少类 touch_face 的全部训练样本数。负类使用跨 epoch 循环洗牌，以尽量覆盖全部硬负样本。
- 验证每轮使用全部冻结验证样本，不做均衡抽样。模型按验证 macro-F1 保存最佳权重。
- 初始化：载入原四分类最佳模型的 X3D-M 主干，丢弃四分类头，重新初始化二分类头，并对全模型微调。

以下结果来自已完成的 30 轮训练及冻结验证集评估。

## 训练结果

- 实际完成 30 个 epoch；最佳 epoch：13。
- 最佳验证 loss：0.358863；accuracy：0.9798；macro-F1：0.8957。
- ROC-AUC：0.9509；PR-AUC：0.8434。

| 类别 | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| normal | 0.9936 | 0.9852 | 0.9894 | 1890 |
| touch_face | 0.7431 | 0.8710 | 0.8020 | 93 |

### 混淆矩阵（行是真值，列是预测）

| true\pred | normal | touch_face |
|---|---:|---:|
| normal | 1862 | 28 |
| touch_face | 12 | 81 |

### 超参数

```json
{
  "experiment": "D:\\CondaData\\factory_x3d_m_touchface_binary\\experiment",
  "backbone": "D:\\CondaData\\factory_x3d_m\\experiment\\best.pt",
  "epochs": 30,
  "batch_size": 8,
  "grad_accum": 2,
  "lr": 5e-05,
  "weight_decay": 0.0001,
  "workers": 6,
  "seed": 20260915,
  "classes": [
    "normal",
    "touch_face"
  ],
  "torch": "2.0.1+cu118",
  "cuda": "11.8",
  "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "selection_metric": "macro_f1"
}
```

## 阈值分析与实时推理建议

阈值扫描使用最佳 epoch 的 1983 个冻结验证样本。该结果属于单窗口、平滑与连续确认之前的离线结果。

| touch_face 阈值 | precision | recall | F1 | FP | FN | 用途 |
|---:|---:|---:|---:|---:|---:|---|
| 0.500 | 0.7431 | 0.8710 | 0.8020 | 28 | 12 | 默认 argmax |
| 0.684 | 0.7938 | 0.8280 | 0.8105 | 20 | 16 | 验证集最佳 touch_face F1 / macro-F1 |
| 0.721 | 0.8000 | 0.7742 | 0.7869 | 18 | 21 | 更偏向控制误报 |

- 实时流初始推荐阈值：`touch_face: 0.68`，配合 3 窗口概率平滑和连续 2 个窗口确认。
- 如果现场更重视少漏报，可降至 `0.50`；如果更重视少误报，可提高至约 `0.72`。
- 最终阈值仍应使用甲方现场 RTSP 视频校准，因为连续视频窗口存在相关性，效果不会与独立验证样本完全相同。
- 完整扫描结果：`threshold_analysis.json`。

## 部署产物

- 训练检查点：`best.pt`；SHA-256：`261bda95f733ea2f4942ce88e88e5eecc907f51efd900ad646561f215a55549c`。
- TorchScript：`x3d_m_normal_touch_face.ts`；SHA-256：`b267edf7df41be50c4dc477d5d925670f9cf82f6cbccb189911e5ae3d8dedd94`。
- 输入：`float32 NCTHW [1,3,16,224,224]`，RGB，mean=`[0.45,0.45,0.45]`，std=`[0.225,0.225,0.225]`。
- 输出类别顺序：`[normal, touch_face]`。
- TorchScript 与原模型的导出校验最大误差：`0.0`。
- 部署元数据：`inference_metadata.json`。
