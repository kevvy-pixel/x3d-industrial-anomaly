# 工业作业异常动作识别

本仓库包含三个独立实现及其本地测试总结：

| 目录 | 任务 | 推理入口 |
|---|---|---|
| `x3d_4class/` | X3D-M 四分类：normal、single_hand、board_drop、touch_face | `infer_stream.py` |
| `x3d_binary/` | X3D-M 二分类：normal、touch_face | `rtsp_realtime/realtime_infer.py` |
| `worker_behavior_ai/` | 三个 YOLO 检测器 + 时序分类器的四分类系统 | `run_worker.ps1` 或 `python -m worker_behavior.delivery` |

测试指标、数据划分和实时推理差异见 [X3D 与 YOLO 模型测试及实时推理总结](docs/X3D与YOLO模型测试与实时推理总结.md)。各子目录的 README 和 `experiment/EXPERIMENT.md` 记录了原实验细节。

## 快速运行

按目标机器的 CPU/CUDA 环境先安装 PyTorch，再安装对应项目的依赖。三个项目应分别在其目录中运行。

```powershell
# X3D 四分类：使用已包含权重的 TorchScript 模型
cd x3d_4class
python -m pip install numpy opencv-python torch
python .\infer_stream.py --model .\experiment\x3d_m_4class.ts --source 0 --display

# X3D 二分类
cd ..\x3d_binary\rtsp_realtime
python -m pip install -r requirements.txt
python .\verify_model.py
python .\realtime_infer.py --source 0 --config .\config.yaml

# YOLO + 时序分类器
cd ..\..\worker_behavior_ai
python -m pip install -r requirements.txt
python -m worker_behavior.delivery --source 0 --display
```

四分类 TorchScript 模型位于 `x3d_4class/experiment/x3d_m_4class.ts`；二分类模型位于 `x3d_binary/experiment/x3d_m_normal_touch_face.ts`；YOLO 的板、手、姿态权重及分类器位于 `worker_behavior_ai/models/`。训练 checkpoint、视频数据、逐样本预测和现场录像未纳入仓库。训练脚本和实验元数据仍保留原机器上的绝对路径；在新机器训练时需调整数据与输出路径，二分类训练还需要四分类训练 checkpoint 作为初始权重。

权重文件来自本次指定的本地项目目录。第三方依赖和模型来源信息见 [THIRD_PARTY.md](THIRD_PARTY.md)。
