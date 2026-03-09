@echo off
chcp 65001
cls

echo ==========================================
echo  Embedding Service API 功能测试
echo ==========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装，请先安装 Python 3.11+
    pause
    exit /b 1
)

REM 检查 uv 是否安装
uv --version >nul 2>&1
if errorlevel 1 (
    echo ❌ uv 未安装，正在安装...
    pip install uv
)

REM 检查 httpx 是否安装
python -c "import httpx" >nul 2>&1
if errorlevel 1 (
    echo 📦 安装 httpx...
    uv pip install httpx
)

echo.
echo 🔍 检查服务状态...
python -c "import httpx; import sys; sys.exit(0 if httpx.get('http://localhost:18000/health', timeout=5).status_code == 200 else 1)" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  服务未启动，正在启动服务...
    echo.
    start "Embedding Service" uv run python start_services.py
    echo ⏳ 等待服务启动（15秒）...
    timeout /t 15 /nobreak >nul
) else (
    echo ✅ 服务已在运行
)

echo.
echo ==========================================
echo  开始 API 功能测试
echo ==========================================
echo.

python test_api_integration.py

if errorlevel 1 (
    echo.
    echo ❌ 测试失败
    pause
    exit /b 1
) else (
    echo.
    echo ✅ 测试全部通过！
    pause
    exit /b 0
)
