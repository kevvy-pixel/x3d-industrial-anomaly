# X3D-M 工业作业异常动作识别实验

本仓库整理了项目从 UCF101 可行性验证、工业少样本五分类、困难正常窗口、固定 ROI，到扩充数据四分类和 80/20 逐轮评估的完整代码、配置与非敏感结果。目标是在工业固定工位视频中识别 `normal`、`board_drop`、`single_hand`、`touch_face` 四种状态。

> 数据集、JSON 标注、源视频名级预测、训练日志和模型权重均不上传。最佳第 13 轮权重及实时推理代码保存在单独的本地部署包中。

## 1. 项目来源与环境

- 初始参考代码：[ZJCV/X3D](https://github.com/ZJCV/X3D)
- 最终训练实现：PyTorchVideo 的 Kinetics-400 预训练 `x3d_m`
- 实验硬件：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）
- 输入：16 帧、2.666667 秒窗口、224×224 网络输入
- 评估预处理：短边缩放至 256，中心裁剪 224×224，均值/标准差为 0.45/0.225
- 训练增强：短边随机缩放、随机裁剪、水平翻转、时间抖动

建议 Python 3.11，并安装与本机 CUDA 匹配的 PyTorch，再运行：

```powershell
pip install -r requirements.txt
```

## 2. 数据要求

数据不在仓库内。配置文件使用以下相对目录；也可以把配置中的路径改为实际位置。

```text
data/
├─ ucf101_mirror/<ClassName>/*.avi
├─ ucfTrainTestlist/
├─ industrial_anomaly/
│  ├─ normal/*.mp4
│  ├─ board_drop/*.mp4
│  ├─ single_hand/*.mp4
│  ├─ touch_face/*.mp4
│  └─ single_hand+board_drop/*.mp4
└─ annotations/<同名类别>/*.json
```

工业异常 JSON 中每个事件至少包含 `start_sec`、`end_sec` 和 `label`。`single_hand+board_drop` 是源视频目录，不是第五类；其中每个标注区间按自己的 `label` 分配到 `single_hand` 或 `board_drop`。异常视频的标注区间外会抽取困难正常窗口，并在事件前后保留 1 秒安全边界。

所有划分均以源视频为组。一个事件可以生成一个窗口，同一源视频也可生成多个困难正常窗口，但同一源视频的所有窗口必须进入同一折或同一侧，禁止相邻窗口跨训练/验证/测试集造成泄漏。

## 3. 实验流程

UCF101：

```powershell
python src/ucf101/prepare_manifests.py --config configs/ucf101_297.yaml
python src/ucf101/train_x3d.py --config configs/ucf101_297.yaml

python src/ucf101/prepare_fewshot.py --config configs/ucf101_fewshot36.yaml
python src/ucf101/train_x3d_fewshot.py --config configs/ucf101_fewshot36.yaml
```

工业四折实验（将配置换成对应版本即可）：

```powershell
python src/industrial/prepare_industrial.py --config configs/industrial_expanded_fullframe_v2.yaml
python src/industrial/verify_4class_inputs.py --config configs/industrial_expanded_fullframe_v2.yaml
python src/industrial/train_industrial.py --config configs/industrial_expanded_fullframe_v2.yaml
```

工业 80/20 固定划分与逐轮评估：

```powershell
python src/industrial/prepare_holdout.py --config configs/industrial_holdout80_20_epoch_eval.yaml
python src/industrial/train_holdout_epoch_eval.py --config configs/industrial_holdout80_20_epoch_eval.yaml
```

训练分两阶段：先冻结骨干训练分类头 6 轮，再以较小学习率微调整网 10 轮。工业实验使用平衡采样、标签平滑和混合精度。所有准确率均为窗口级，除非明确写为源视频级。

## 4. 版本与结果更新

| # | 实验 | 数据与设置 | 主要结果 |
|---:|---|---|---|
| 01 | UCF101 紧凑子集 | 297 视频、5 类；197/53/47；5+10 轮 | 测试 Acc 100%，Macro-F1 1.000 |
| 02 | UCF101 少样本模拟 | 36 视频、5 类；21/7/8；10+15 轮 | 测试 Acc 100%，Macro-F1 1.000；仅 8 个测试样本，不能外推 |
| 03 | 工业五分类基线 | 36 源视频、39 事件窗口；源视频级 4 折；仅 normal 文件夹提供正常样本 | OOF Acc 64.10%，Macro-F1 0.415；源视频 Acc 61.11%；normal Recall 0% |
| 04 | 五分类 + 困难正常 | 36 源视频、66 窗口；增加 27 个异常视频非标注区间 | OOF Acc 50.00%，Macro-F1 0.331；困难正常 Recall 44.44% |
| 05 | 四分类全图 v1 | 暂停 no_gloves；`single_hand_or_drag` 统一为 `single_hand`；31 源、59 窗口 | OOF Acc 66.10%，Macro-F1 0.581；困难正常 Recall 52% |
| 06 | 四分类固定 ROI v1 | 与 05 同划分，按每个视频的 YOLO 格式矩形裁剪并扩展为方形 | OOF Acc 71.19%，Macro-F1 0.590；困难正常 Recall 72% |
| 07 | 扩充四分类全图 v2 | 160 源、237 窗口；含复合源视频的区间级标签；4 折 | OOF Acc 82.70%，Balanced Acc 82.29%，Macro-F1 0.825；源视频 Acc 80.63% |
| 08 | 扩充四分类 ROI v2 | 与 07 同数据和划分，仅输入 ROI | OOF Acc 77.22%，Balanced Acc 77.84%，Macro-F1 0.769；困难正常 Recall 82.76% |
| 09 | 全图 80/20，固定终轮 | 108/27 源视频、191/46 窗口；无验证集；固定训练 16 轮后评估 | 测试 Acc 82.61%，Balanced Acc 85.33%，Macro-F1 0.824 |
| 10 | 全图 80/20，逐轮评估 | 与 09 完全相同的固定划分；每轮评估，以 Macro-F1 选模型 | 第 13 轮最佳：Acc 86.96%，Balanced Acc 89.08%，Macro-F1 0.867；第 16 轮 Macro-F1 0.849 |

第 13 轮各类评估指标：

| 类别 | Precision | Recall/类准确率 | F1 | N |
|---|---:|---:|---:|---:|
| normal | 84.62% | 100.00% | 0.917 | 11 |
| board_drop | 66.67% | 85.71% | 0.750 | 7 |
| single_hand | 92.31% | 70.59% | 0.800 | 17 |
| touch_face | 100.00% | 100.00% | 1.000 | 11 |

这里的 20% 集合在每轮都被用于选最佳 epoch，因此它实际是开发/评估集，而不是完全独立、无偏的最终测试集。正式汇报或部署前，仍需锁定模型后增加一批从未参与调参的独立源视频。

## 5. 关键结论

- 数据扩充是最大增益来源：全图四折 Macro-F1 从 0.581 提升至 0.825。
- 固定 ROI 在早期小数据上改善明显，但扩充数据后全图整体分类更强；ROI 对困难正常的识别仍更好。
- `single_hand` 是当前最佳模型的主要短板（Recall 70.59%），后续适合尝试全图与手部/工位 ROI 双分支可学习融合。
- `no_gloves` 更偏静态局部外观判断，已从时序四分类中移除，建议单独使用手部检测与图像分类。
- 当前训练/测试使用标注事件窗口与困难正常窗口，不等于完整视频连续检测。部署验证还需要统计事件召回、时间 IoU、报警合并和每分钟误报数。

## 6. 仓库结构

```text
configs/       十个实验的可复现实验配置
src/ucf101/    UCF101 清单生成与训练代码
src/industrial/工业数据清单、四折训练、80/20 训练及审计代码
src/tools/     数据、解码、ROI 与训练/测试指标诊断工具
results/       脱敏后的汇总、分类报告、混淆矩阵和训练曲线
reports/       工业四分类详细审计报告
docs/          后续全图 + ROI 动态融合方案
```

详细类别指标、误判分析和指标定义见 [工业实验审计报告](reports/industrial_audit_report_zh.md)。原始结果 JSON/CSV 是最终数字的依据。

## 7. 结果解释边界

窗口准确率表示抽出的 2.67 秒窗口被正确分类的比例；Recall 与“类准确率”在单标签分类里是同一个量，即该真实类别中预测正确的比例。困难正常窗口是从异常源视频的所有异常标注区间之外抽取、且避开安全边界的正常片段。它们按源视频分组划分，不与该视频的异常窗口跨集合。

本项目仍处于研究验证阶段，不应直接用于现场自动停线或安全联锁。
