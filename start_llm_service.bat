@echo off
chcp 65001 >nul

title MiniCPM4-0.5B API 服务
setlocal EnableDelayedExpansion

:: ==================== 配置区域 ====================
set "PROJECT_DIR=D:\embedding_service"
set "PYTHON_PATH=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT_PATH=%PROJECT_DIR%\src\qwen3_embedding_service\start_llm.py"
set "PORT=18001"
set "MAX_BATCH_SIZE=32"        :: MiniCPM4-0.5B 推荐值：32-64（根据显存调整）
set "CACHE_SIZE=500"

:: ==================== 颜色定义 ====================
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

:: ==================== 启动画面 ====================
echo %BLUE%
echo ============================================
echo    MiniCPM4-0.5B API 服务启动脚本
echo ============================================
echo %RESET%
echo.

:: ==================== 环境检查 ====================
echo %YELLOW%[检查] 项目环境...%RESET%

:: 检查 Python 虚拟环境
if not exist "%PYTHON_PATH%" (
    echo %RED%[错误] 未找到 Python 解释器: %PYTHON_PATH%%RESET%
    echo 请先执行以下命令创建环境：
    echo   uv venv
    echo   uv pip install -r requirements.txt
    pause
    exit /b 1
)

:: 检查主脚本
if not exist "%SCRIPT_PATH%" (
    echo %RED%[错误] 未找到主脚本: %SCRIPT_PATH%%RESET%
    echo 请确认 llm.py 位于项目根目录（%PROJECT_DIR%）
    pause
    exit /b 1
)

:: 检查模型缓存（非强制，首次启动自动下载）
set "MODEL_CACHE_DIR=%PROJECT_DIR%\src\qwen3_embedding_service\models\models--OpenBMB--MiniCPM4-0.5B"
if not exist "%MODEL_CACHE_DIR%" (
    echo %YELLOW%[提示] 模型缓存未找到，首次启动将自动下载（约 1.0GB）%RESET%
    echo 模型路径: %MODEL_CACHE_DIR%
    echo.
) else (
    echo %GREEN%[通过] 模型缓存已存在%RESET%
)

echo %GREEN%[通过] 环境检查完成%RESET%
echo.

:: ==================== 端口检查 ====================
echo %YELLOW%[检查] 端口 %PORT% 占用情况...%RESET%

netstat -ano | findstr ":%PORT%" >nul
if !errorlevel! equ 0 (
    echo %RED%[警告] 端口 %PORT% 已被占用%RESET%
    echo 请关闭占用程序或修改 PORT 配置
    echo 查看命令: netstat -ano ^| findstr ":%PORT%"
    pause
    exit /b 1
)

echo %GREEN%[通过] 端口 %PORT% 可用%RESET%
echo.

:: ==================== 设置环境变量 ====================
set "HF_HOME=%PROJECT_DIR%\src\qwen3_embedding_service\models"
set "MAX_BATCH_SIZE=%MAX_BATCH_SIZE%"
set "CACHE_SIZE=%CACHE_SIZE%"

echo %BLUE%[配置] 运行参数%RESET%
echo -------------------------------------------
echo 项目目录   : %PROJECT_DIR%
echo 模型缓存   : %HF_HOME%
echo 监听端口   : %PORT%
echo 批量大小   : %MAX_BATCH_SIZE% (MiniCPM4-0.5B 推荐: 32-64)
echo 缓存大小   : %CACHE_SIZE%
echo -------------------------------------------
echo.

:: ==================== 启动服务 ====================
echo %BLUE%[启动] 正在启动 MiniCPM4-0.5B 服务...%RESET%
echo %YELLOW%-------------------------------------------%RESET%
echo 服务地址   : http://localhost:%PORT%
echo 健康检查   : http://localhost:%PORT%/health
echo API 文档   : http://localhost:%PORT%/docs
echo 指标统计   : http://localhost:%PORT%/stats
echo 模型列表   : http://localhost:%PORT%/v1/models
echo 对话接口   : http://localhost:%PORT%/v1/chat/completions
echo 生成接口   : http://localhost:%PORT%/generate
echo -------------------------------------------%RESET%
echo.

cd /d "%PROJECT_DIR%"

:: 启动服务（直接使用虚拟环境 Python）
"%PYTHON_PATH%" "%SCRIPT_PATH%"

:: 服务退出处理
echo.
if !errorlevel! equ 0 (
    echo %GREEN%[正常] 服务已正常退出%RESET%
) else (
    echo %RED%[错误] 服务异常退出 (代码: !errorlevel!)%RESET%
    echo 请检查控制台日志以排查问题
)
echo 按任意键关闭窗口...
pause >nul