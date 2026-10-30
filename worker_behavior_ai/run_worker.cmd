@echo off
setlocal
if "%WORKER_RTSP_URL%"=="" (
  echo Set WORKER_RTSP_URL first.
  exit /b 2
)
python -m worker_behavior.delivery --source "%WORKER_RTSP_URL%" %*
