@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PROJECT_PYTHON=%USERPROFILE%\anaconda3\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%PROJECT_PYTHON%" (
    echo [错误] 未找到本机 Anaconda Python：%PROJECT_PYTHON%
    echo 请确认 Anaconda 安装位置，或修改 run_gui.bat 中的 PROJECT_PYTHON。
    pause
    exit /b 1
)

"%PROJECT_PYTHON%" gui_app.py
if errorlevel 1 (
    echo.
    echo [错误] GUI 运行失败，退出代码：%ERRORLEVEL%
    pause
)
