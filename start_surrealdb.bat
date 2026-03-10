@echo off
chcp 65001 >nul

title SurrealDB Database Service
setlocal EnableDelayedExpansion

:: ==================== 配置区域 ====================
set "PROJECT_DIR=D:\embedding_service"
set "DATA_DIR=%PROJECT_DIR%\surrealdb_data"
set "SURREAL_PATH=C:\Users\Longray\AppData\Local\SurrealDB\surreal.exe"
set "PORT=8000"
set "HOST=0.0.0.0"
set "USERNAME=root"
set "PASSWORD=root"

:: ==================== 颜色定义 ====================
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "CYAN=[96m"
set "RESET=[0m"

:: ==================== 启动画面 ====================
echo %CYAN%
echo ============================================
echo       SurrealDB Database Service
echo ============================================
echo %RESET%
echo.

:: ==================== 环境检查 ====================
echo %YELLOW%[检查] SurrealDB 可执行文件...%RESET%

if not exist "%SURREAL_PATH%" (
    echo %RED%[错误] 未找到 surreal.exe: %SURREAL_PATH%%RESET%
    echo 请确认 SurrealDB 已正确安装
    pause
    exit /b 1
)

echo %GREEN%[通过] 环境检查完成%RESET%
echo.

:: ==================== 数据目录检查 ====================
echo %YELLOW%[检查] 数据目录...%RESET%

if not exist "%DATA_DIR%" (
    echo %YELLOW%[创建] 数据目录: %DATA_DIR%%RESET%
    mkdir "%DATA_DIR%"
)

echo %GREEN%[通过] 数据目录就绪%RESET%
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
echo %BLUE%[启动] 正在启动 SurrealDB 服务...%RESET%
echo %YELLOW%-------------------------------------------%RESET%
echo 数据目录: %DATA_DIR%
echo 监听地址: http://%HOST%:%PORT%
echo WebSocket: ws://localhost:%PORT%/rpc
echo %YELLOW%-------------------------------------------%RESET%
echo.

:: 切换到项目目录
cd /d "%PROJECT_DIR%"

:: 启动 SurrealDB
"%SURREAL_PATH%" start --bind %HOST%:%PORT% --user %USERNAME% --pass %PASSWORD% rocksdb://%DATA_DIR%

:: 如果服务异常退出
echo.
echo %RED%[停止] 服务已退出 (代码: %errorlevel%)%RESET%
echo 按任意键关闭窗口...
pause >nul