@echo off
chcp 65001 >nul
title Qwen3-Embedding-0.6B API 服务
setlocal EnableDelayedExpansion

:: ==================== 配置区域 ====================
set "PROJECT_DIR=D:\embedding_service"
set "UV_PATH=C:\Users\Longray\.local\bin\uv.exe"
set "PYTHON_PATH=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT_PATH=%PROJECT_DIR%\src\qwen3_embedding_service\main.py"
set "PORT=18000"

:: ==================== 颜色定义 ====================
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

:: ==================== 启动画面 ====================
echo %BLUE%
echo ============================================
echo    Qwen3-Embedding-0.6B API 服务启动脚本
echo ============================================
echo %RESET%
echo.

:: ==================== 环境检查 ====================
echo %YELLOW%[检查] 项目环境...%RESET%

:: 检查 UV 是否存在
if not exist "%UV_PATH%" (
    echo %RED%[错误] 未找到 uv.exe: %UV_PATH%%RESET%
    echo 请确认 uv 已正确安装
    pause
    exit /b 1
)

:: 检查 Python 虚拟环境
if not exist "%PYTHON_PATH%" (
    echo %RED%[错误] 未找到 Python 解释器: %PYTHON_PATH%%RESET%
    echo 请确认虚拟环境已创建
    pause
    exit /b 1
)

:: 检查主脚本
if not exist "%SCRIPT_PATH%" (
    echo %RED%[错误] 未找到主脚本: %SCRIPT_PATH%%RESET%
    pause
    exit /b 1
)

echo %GREEN%[通过] 环境检查完成%RESET%
echo.

:: ==================== 端口检查 ====================
echo %YELLOW%[检查] 端口 %PORT% 占用情况...%RESET%

netstat -ano | findstr ":%PORT%" >nul
if %errorlevel% equ 0 (
    echo %RED%[警告] 端口 %PORT% 已被占用%RESET%
    echo 请关闭占用该端口的程序后重试
    echo 或使用命令查看: netstat -ano ^| findstr ":%PORT%"
    pause
    exit /b 1
)

echo %GREEN%[通过] 端口 %PORT% 可用%RESET%
echo.

:: ==================== 启动服务 ====================
echo %BLUE%[启动] 正在启动 Embedding 服务...%RESET%
echo %YELLOW%-------------------------------------------%RESET%
echo 模型路径: %PROJECT_DIR%\src\qwen3_embedding_service\models
echo 监听地址: http://0.0.0.0:%PORT%
echo 健康检查: http://localhost:%PORT%/health
echo %YELLOW%-------------------------------------------%RESET%
echo.

:: 切换到项目目录
cd /d "%PROJECT_DIR%"

:: 启动服务（使用 uv run）
"%UV_PATH%" run "%PYTHON_PATH%" "%SCRIPT_PATH%"

:: 如果服务异常退出
echo.
echo %RED%[停止] 服务已退出 (代码: %errorlevel%)%RESET%
echo 按任意键关闭窗口...
pause >nul