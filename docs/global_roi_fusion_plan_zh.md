# X3D-M 全局与 ROI 融合实验方案

## 目标

在不重新训练现有全画面与 ROI 四分类 X3D-M 的前提下，利用两种视图的互补性，提高正常识别能力，同时减少 `board_drop` 漏报。

## 数据与评价

- 类别：`normal`、`board_drop`、`single_hand`、`touch_face`
- 沿用相同的源视频级四折划分。
- 同一源视频的所有窗口只能进入同一个 train/validation/test 集合。
- 融合参数只能使用每一折的验证集学习，不能在 OOF 测试预测上搜索权重后再报告同一批测试结果。
- 报告窗口级、源视频级和正常/异常报警二分类指标。

## 对照方法

1. 全画面模型。
2. ROI 模型。
3. 固定 50/50 概率或 logit 融合。
4. 每折验证集搜索一个全局权重 `alpha`。
5. 每类别学习一个融合权重与偏置。

固定融合公式：

```text
fused = alpha * global + (1 - alpha) * roi
```

类别级融合公式：

```text
fused_logit[c] = alpha[c] * global_logit[c]
                + (1 - alpha[c]) * roi_logit[c]
                + bias[c]
alpha[c] = sigmoid(theta[c])
```

两个 X3D 主干保持冻结。类别级融合层仅包含 4 个权重和 4 个偏置，并使用将权重约束在 0.5 附近的正则项。

## 暂不采用的方案

当前只有 31 个独立源视频，不先训练复杂的逐窗口动态门控网络。待数据量增加后，可使用两个模型的概率、熵、置信度差和预测一致性作为输入，学习逐窗口动态权重。

## 模型选择原则

不能只优化总体准确率。优先满足验证集异常召回率要求，再最大化正常召回率，并以 Macro-F1 作为并列判据。

建议同时报告：

- Accuracy、Macro-F1、Balanced Accuracy；
- 各类别 Precision、Recall、F1；
- 正常误报率、异常漏报率；
- 源视频级 Accuracy 和 Macro-F1；
- 四折融合权重的均值和标准差。

## 当前探索性参考

现有 OOF 预测直接进行未调参的 50/50 融合时：

- 窗口准确率：72.88%；
- Macro-F1：0.612；
- `normal` Recall：71.9%；
- `board_drop` Recall：62.5%；
- `single_hand` Recall：25.0%；
- `touch_face` Recall：93.3%。

该结果仅用于证明两种视图具有互补性，不作为独立测试结论。
