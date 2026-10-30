# X3D-M normal / touch_face RTSP 实时推理

本工程加载二分类 TorchScript 模型，类别顺序为 `normal, touch_face`，默认打开 OpenCV 实时画面窗口。

## 当前推理配置

- 每个窗口 16 帧；按 RTSP 报告帧率选择最近档位：16 FPS=`1.0s`、24 FPS=`0.8s`、30 FPS=`0.7s`。
- 推理步长 `0.3s`，3 个窗口概率平滑。
- `touch_face` 阈值 `0.68`，连续 2 个窗口确认，2 秒内事件合并。
- 报警时自动将触发窗口保存为 H.264 MP4 到 `outputs/alarms`，事件记录写入 `logs/events.jsonl`。
- RTSP 断流自动重连；画面尺寸改变时清空旧缓存，避免不同分辨率帧混合。

## 启动

在当前训练电脑上，最简单的方法是双击 `run_rtsp_local.cmd`，粘贴 RTSP 地址后按回车。

在本工程目录打开 CMD：

```cmd
run_rtsp.cmd "rtsp://用户名:密码@摄像机IP:554/路径"
```

或者在 PowerShell 中指定 Python：

```powershell
.\run_rtsp.ps1 `
  -RtspUrl "rtsp://用户名:密码@摄像机IP:554/路径" `
  -PythonExe "python"
```

如果该 Python 已在 PATH 中，可省略 `-PythonExe`。画面窗口中按 `Q` 或 `Esc` 退出。

直接调用主程序也可以：

```powershell
python .\realtime_infer.py --source "rtsp://用户名:密码@摄像机IP:554/路径"
```

不要加 `--headless`，否则不会显示实时窗口。若希望同时保存整段标注视频，可增加：

```powershell
--output-video outputs\annotated.mp4
```

## 验证安装

```powershell
python .\verify_model.py
```

模型是 TorchScript，不需要安装 `pytorchvideo`。CPU 环境可安装 CPU 版 PyTorch；有 NVIDIA GPU 时使用匹配的 CUDA 版 PyTorch。

## 可选告警回调

```powershell
python .\realtime_infer.py `
  --source "rtsp://用户名:密码@摄像机IP:554/路径" `
  --alarm-callback-url "http://127.0.0.1:9000/alarm/callback" `
  --device-id "camera01"
```

本地报警视频不依赖回调地址，默认始终保存。
