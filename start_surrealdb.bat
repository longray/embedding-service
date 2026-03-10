@echo off
chcp 65001 >nul

title SurrealDB 数据库服务
setlocal EnableDelayedExpansion

:: ==================== 配置区域 ====================
set "PROJECT_DIR=D:\embedding_service"
set "DATA_DIR=%PROJECT_DIR%\surrealdb_data"
set "LOG_FILE=%PROJECT_DIR%\surrealdb.log"
set "PORT=8000"
set "HOST=0.0.0.0"
set "USERNAME=root"
set "PASSWORD=root"

:: SurrealDB 可执行文件路径（按优先级）
set "SURREAL_PATH1=C:\Users\Longray\AppData\Local\SurrealDB\surreal.exe"
set "SURREAL_PATH2=C:\Program Files\SurrealDB\surreal.exe"
set "SURREAL_PATH="

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
echo       SurrealDB 数据库服务启动脚本
echo ============================================
echo %RESET%
echo.

:: ==================== 查找 SurrealDB ====================
echo %YELLOW%[检查] SurrealDB 可执行文件...%RESET%

if exist "%SURREAL_PATH1%" (
    set "SURREAL_PATH=%SURREAL_PATH1%"
    echo %GREEN%[找到] %SURREAL_PATH1%%RESET%
) else if exist "%SURREAL_PATH2%" (
    set "SURREAL_PATH=%SURREAL_PATH2%"
    echo %GREEN%[找到] %SURREAL_PATH2%%RESET%
) else (
    echo %RED%[错误] 未找到 surreal.exe%RESET%
    echo.
    echo 请安装 SurrealDB:
    echo   1. 访问 https://surrealdb.com/install
    echo   2. 或使用 winget install SurrealDB.SurrealDB
    echo.
    pause
    exit /b 1
)

:: ==================== 数据目录检查 ====================
echo %YELLOW%[检查] 数据目录...%RESET%

if not exist "%DATA_DIR%" (
    echo %YELLOW%[创建] 数据目录: %DATA_DIR%%RESET%
    mkdir "%DATA_DIR%"
)

echo %GREEN%[通过] 数据目录: %DATA_DIR%%RESET%
echo.

:: ==================== 端口检查 ====================
echo %YELLOW%[检查] 端口 %PORT% 占用情况...%RESET%

netstat -ano | findstr ":%PORT%" >nul
if %errorlevel% equ 0 (
    echo %RED%[警告] 端口 %PORT% 已被占用%RESET%
    echo.
    echo 可能已有 SurrealDB 实例在运行
    echo 请先关闭现有实例或更换端口
    echo.
    echo 查看占用进程:
    echo   netstat -ano ^| findstr ":%PORT%"
    echo.
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
echo 用户名:   %USERNAME%
echo 密码:     %PASSWORD%
echo 日志文件: %LOG_FILE%
echo %YELLOW%-------------------------------------------%RESET%
echo.
echo %CYAN%提示: 按 Ctrl+C 停止服务%RESET%
echo.

:: 切换到项目目录
cd /d "%PROJECT_DIR%"

:: 启动 SurrealDB（内存模式 + 文件持久化）
"%SURREAL_PATH%" start --bind "%HOST%:%PORT%" --user %USERNAME% --pass %PASSWORD% "file://%DATA_DIR%" 2>&1 | tee "%LOG_FILE%"

:: 如果服务异常退出
echo.
echo %RED%[停止] 服务已退出 (代码: %errorlevel%)%RESET%
echo 按任意键关闭窗口...
pause >nul