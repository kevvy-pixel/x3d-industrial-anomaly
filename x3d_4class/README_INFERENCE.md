# X3D-M 四分类推理说明

## 推荐模型

- `experiment/x3d_m_4class.ts`：推荐用于实时视频流，已包含网络结构和权重。
- 精简 PyTorch state_dict 和完整训练 checkpoint 保存在原实验目录，仓库仅包含实时推理所需的 TorchScript 文件。

类别顺序固定为：`normal=0, single_hand=1, board_drop=2, touch_face=3`。

## 视频流推理

在 PowerShell 中执行：

```powershell
python .\infer_stream.py `
  --model .\experiment\x3d_m_4class.ts `
  --source 0 `
  --stride-sec 0.3 `
  --smooth 3 `
  --display
```

`--source` 可使用摄像头编号、视频文件路径或 RTSP URL。例如：

```powershell
python .\infer_stream.py `
  --model .\experiment\x3d_m_4class.ts `
  --source "rtsp://user:password@host/stream" `
  --stride-sec 0.3 `
  --smooth 3
```

脚本按输入流 FPS 就近使用训练时的 1.0/0.8/0.7 秒因果窗口，均匀抽取 16 帧；默认每 0.3 秒预测一次，并对最近 3 次概率做移动平均。告警阈值为 board_drop 0.60、single_hand 0.60、touch_face 0.65。board_drop 采用最近 2 次中任意 1 次确认，另外两类要求连续 2 次确认；同类告警在 2 秒内合并。每次结果以 JSON 行输出。

如需保存带预测文字的视频，增加：

```text
--output-video .\stream_result.mp4
```

## 文件与实验记录

- `experiment/EXPERIMENT.md`：数据规则、固定切分、样本数量、均衡采样和最终指标。
- `experiment/samples.csv` 和 `experiment/val_sources.csv`：保存在原实验目录的样本及固定验证清单，仓库未包含。
- `experiment/history.csv`：30 轮训练历史。
- `experiment/deployment_smoke_test.json`：真实验证片段的部署验收结果。
