@echo off
chcp 65001 >nul

title SurrealDB Database Service
setlocal EnableDelayedExpansion

:: ==================== Configuration ====================
set "PROJECT_DIR=D:\embedding_service"
set "DATA_DIR=%PROJECT_DIR%\surrealdb_data"
set "SURREAL_PATH=C:\Users\Longray\AppData\Local\SurrealDB\surreal.exe"
set "PORT=8000"
set "HOST=0.0.0.0"
set "USERNAME=root"
set "PASSWORD=root"
set "WS_URL=ws://localhost:%PORT%/rpc"
set "HEALTH_TIMEOUT=3"

:: Color definitions
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "CYAN=[96m"
set "RESET=[0m"

:: ==================== Startup Banner ====================
echo %CYAN%
echo ============================================
echo       SurrealDB Database Service Startup
echo ============================================
echo %RESET%
echo.

:: ==================== Health Check ====================
echo %YELLOW%[CHECK] Checking if SurrealDB is already running...%RESET%

curl -sf http://localhost:%PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo %GREEN%[OK] SurrealDB is already running and healthy!%RESET%
    echo %CYAN%WebSocket endpoint: %WS_URL%%RESET%
    echo.
    echo %YELLOW%Press any key to close...%RESET%
    pause >nul
    exit /b 0
)

echo %YELLOW%[INFO] SurrealDB not running, will start new instance...%RESET%
echo.

:: ==================== Environment Check ====================
echo %YELLOW%[CHECK] Checking SurrealDB executable...%RESET%

if not exist "%SURREAL_PATH%" (
    echo %RED%[ERROR] surreal.exe not found: %SURREAL_PATH%%RESET%
    echo Please ensure SurrealDB is installed correctly
    pause
    exit /b 1
)

echo %GREEN%[PASS] Environment check completed%RESET%
echo.

:: ==================== Data Directory Check ====================
echo %YELLOW%[CHECK] Checking data directory...%RESET%

if not exist "%DATA_DIR%" (
    echo %YELLOW%[CREATE] Creating data directory: %DATA_DIR%%RESET%
    mkdir "%DATA_DIR%"
)

echo %GREEN%[PASS] Data directory ready%RESET%
echo.

:: ==================== Port Check ====================
echo %YELLOW%[CHECK] Checking port %PORT% availability...%RESET%

netstat -ano | findstr ":%PORT%" >nul
if %errorlevel% equ 0 (
    echo %RED%[WARNING] Port %PORT% is already in use%RESET%
    echo.
    echo %YELLOW%Attempting health check on existing service...%RESET%
    curl -sf http://localhost:%PORT%/health >nul 2>&1
    if !errorlevel! equ 0 (
        echo %GREEN%[OK] Existing service is healthy. Using existing instance.%RESET%
        echo %YELLOW%Press any key to close...%RESET%
        pause >nul
        exit /b 0
    )
    echo %YELLOW%[INFO] Port occupied but service not healthy, killing process...%RESET%
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr LISTENING') do (
        echo [KILL] Terminating PID: %%a
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
    echo %GREEN%[OK] Port %PORT% cleared%RESET%
)

echo %GREEN%[PASS] Port %PORT% is available%RESET%
echo.

:: ==================== Start Service ====================
echo %BLUE%[START] Starting SurrealDB Service...%RESET%
echo %YELLOW%-------------------------------------------%RESET%
echo Data Directory: %DATA_DIR%
echo Listen Address: http://%HOST%:%PORT%
echo WebSocket: %WS_URL%
echo %YELLOW%-------------------------------------------%RESET%
echo.

:: Change to project directory
cd /d "%PROJECT_DIR%"

:: Start SurrealDB
"%SURREAL_PATH%" start --bind %HOST%:%PORT% --user %USERNAME% --pass %PASSWORD% rocksdb://%DATA_DIR%

:: If service exits abnormally
echo.
echo %RED%[STOP] Service exited (code: %errorlevel%)%RESET%
echo Press any key to close window...
pause >nul
