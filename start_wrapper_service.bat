@echo off
chcp 65001 >nul

title Wrapper Service (Minimal)
setlocal EnableDelayedExpansion

:: ==================== Configuration ====================
set "PROJECT_DIR=D:\embedding_service"
set "UV_PATH=C:\Users\Longray\.local\bin\uv.exe"
set "PYTHON_PATH=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "PORT=18008"
set "HEALTH_URL=http://localhost:%PORT%/health"
set "HEALTH_TIMEOUT=3"

:: ==================== Startup Banner ====================
echo ============================================
echo       Wrapper Service Startup (Minimal)
echo ============================================
echo.

:: ==================== Health Check ====================
echo [CHECK] Checking if service is already running...

curl -sf %HEALTH_URL% >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Service is already running and healthy!
    echo Health endpoint: %HEALTH_URL%
    echo.
    echo Press any key to close...
    pause >nul
    exit /b 0
)

echo [INFO] Service not running, will start new instance...
echo.

:: ==================== Environment Check ====================
echo [CHECK] Checking environment...

if not exist "%UV_PATH%" (
    echo [ERROR] uv.exe not found: %UV_PATH%
    echo Please ensure uv is installed correctly
    pause
    exit /b 1
)

if not exist "%PYTHON_PATH%" (
    echo [ERROR] Python interpreter not found: %PYTHON_PATH%
    echo Please ensure virtual environment is created
    pause
    exit /b 1
)

echo [PASS] Environment check completed
echo.

:: ==================== Port Check ====================
echo [CHECK] Checking port %PORT% availability...

netstat -ano | findstr ":%PORT%" >nul
if %errorlevel% equ 0 (
    echo [WARNING] Port %PORT% is already in use
    echo.
    echo Attempting health check on existing service...
    curl -sf %HEALTH_URL% >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Existing service is healthy. Using existing instance.
        echo Press any key to close...
        pause >nul
        exit /b 0
    )
    echo [INFO] Port occupied but service not healthy, killing process...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr LISTENING') do (
        echo [KILL] Terminating PID: %%a
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
    echo [OK] Port %PORT% cleared
)

echo [PASS] Port %PORT% is available
echo.

:: ==================== Start Service ====================
echo [START] Starting Wrapper Service...
echo -------------------------------------------
echo Listen Address: http://0.0.0.0:%PORT% 
echo Health Check: %HEALTH_URL%
echo -------------------------------------------
echo.

:: Change to project directory
cd /d "%PROJECT_DIR%"

:: Start service using uv run
"%UV_PATH%" run "%PYTHON_PATH%" -m wrapper.src.main

:: If service exits abnormally
echo.
echo [STOP] Service exited (code: %errorlevel%)
echo Press any key to close window...
pause >nul