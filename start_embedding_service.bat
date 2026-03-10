@echo off
chcp 65001 >nul

title Qwen3-Embedding-0.6B API Service
setlocal EnableDelayedExpansion

:: ==================== Configuration ====================
set "PROJECT_DIR=D:\embedding_service"
set "UV_PATH=C:\Users\Longray\.local\bin\uv.exe"
set "PYTHON_PATH=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT_PATH=%PROJECT_DIR%\src\qwen3_embedding_service\start_embedding.py"
set "PORT=18000"
set "HEALTH_URL=http://localhost:%PORT%/health"
set "HEALTH_TIMEOUT=3"
set MAX_BATCH_SIZE=256

:: Color definitions
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

:: ==================== Startup Banner ====================
echo %BLUE%
echo ============================================
echo    Qwen3-Embedding-0.6B API Service Startup
echo ============================================
echo %RESET%
echo.

:: ==================== Health Check ====================
echo %YELLOW%[CHECK] Checking if service is already running...%RESET%

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%HEALTH_URL%' -Method GET -TimeoutSec %HEALTH_TIMEOUT% -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1

if %errorlevel% equ 0 (
    echo %GREEN%[OK] Service is already running and healthy!%RESET%
    echo %CYAN%Health endpoint: %HEALTH_URL%%RESET%
    echo.
    echo %YELLOW%Service is already available. No need to restart.%RESET%
    echo.
    timeout /t 3 /nobreak >nul
    exit /b 0
)

echo %YELLOW%[INFO] Service not running, will start new instance...%RESET%
echo.

:: ==================== Environment Check ====================
echo %YELLOW%[CHECK] Checking environment...%RESET%

:: Check UV exists
if not exist "%UV_PATH%" (
    echo %RED%[ERROR] uv.exe not found: %UV_PATH%%RESET%
    echo Please ensure uv is installed correctly
    pause
    exit /b 1
)

:: Check Python virtual environment
if not exist "%PYTHON_PATH%" (
    echo %RED%[ERROR] Python interpreter not found: %PYTHON_PATH%%RESET%
    echo Please ensure virtual environment is created
    pause
    exit /b 1
)

:: Check main script
if not exist "%SCRIPT_PATH%" (
    echo %RED%[ERROR] Main script not found: %SCRIPT_PATH%%RESET%
    pause
    exit /b 1
)

echo %GREEN%[PASS] Environment check completed%RESET%
echo.

:: ==================== Port Check ====================
echo %YELLOW%[CHECK] Checking port %PORT% availability...%RESET%

netstat -ano | findstr ":%PORT%" >nul
if %errorlevel% equ 0 (
    echo %RED%[WARNING] Port %PORT% is already in use%RESET%
    echo Another process may be using this port.
    echo Check with: netstat -ano ^| findstr ":%PORT%"
    echo.
    echo %YELLOW%Attempting health check on existing service...%RESET%
    
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%HEALTH_URL%' -Method GET -TimeoutSec %HEALTH_TIMEOUT% -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    
    if %errorlevel% equ 0 (
        echo %GREEN%[OK] Existing service is healthy! Using existing instance.%RESET%
        timeout /t 2 /nobreak >nul
        exit /b 0
    ) else (
        echo %RED%[ERROR] Port %PORT% is in use but service is not healthy%RESET%
        echo Please stop the process using this port and try again.
        pause
        exit /b 1
    )
)

echo %GREEN%[PASS] Port %PORT% is available%RESET%
echo.

:: ==================== Start Service ====================
echo %BLUE%[START] Starting Embedding Service...%RESET%
echo %YELLOW%-------------------------------------------%RESET%
echo Model Path: %PROJECT_DIR%\src\qwen3_embedding_service\models
echo Listen Address: http://0.0.0.0:%PORT%
echo Health Check: %HEALTH_URL%
echo %YELLOW%-------------------------------------------%RESET%
echo.

:: Change to project directory
cd /d "%PROJECT_DIR%"

:: Start service using uv run
"%UV_PATH%" run "%PYTHON_PATH%" "%SCRIPT_PATH%"

:: If service exits abnormally
echo.
echo %RED%[STOP] Service exited (code: %errorlevel%)%RESET%
echo Press any key to close window...
pause >nul
