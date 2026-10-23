@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PROJECT_PYTHON=%USERPROFILE%\anaconda3\envs\yolov5\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem The local OpenCV and PyTorch packages bundle different Intel OpenMP DLLs.
rem Keep one inference section at a time and allow both runtimes to coexist.
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
rem OpenCV/FFmpeg must receive this before Python starts, otherwise some RTSP
rem streams fall back to UDP and open without ever returning a decodable frame.
set "OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp"

if not exist "%PROJECT_PYTHON%" (
    echo [错误] 未找到 yolov5 环境：%PROJECT_PYTHON%
    pause
    exit /b 1
)

echo 正在启动工程机械伤害检测 Web 平台...
echo 访问地址：http://127.0.0.1:8000
"%PROJECT_PYTHON%" server.py --serve --host 127.0.0.1 --port 8000

if errorlevel 1 (
    echo.
    echo [错误] Web 服务运行失败，退出代码：%ERRORLEVEL%
    pause
)
