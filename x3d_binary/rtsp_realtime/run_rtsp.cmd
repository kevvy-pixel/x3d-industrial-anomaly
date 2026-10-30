@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  set /p "RTSP_URL=Enter RTSP URL: "
) else (
  set "RTSP_URL=%~1"
)
if not defined RTSP_URL exit /b 2
python realtime_infer.py --source "%RTSP_URL%" --config config.yaml
if errorlevel 1 pause
endlocal
