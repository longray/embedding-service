@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

title Embedding Service - Smart Startup
echo.
echo ============================================
echo    Embedding Service - Smart Startup
echo ============================================
echo.

:: ==================== Configuration ====================
set "PROJECT_DIR=D:\embedding_service"
set "SURREAL_PATH=C:\Users\Longray\AppData\Local\SurrealDB\surreal.exe"
set "DATA_DIR=%PROJECT_DIR%\surrealdb_data"

set "SURREALDB_PORT=8000"
set "EMBEDDING_PORT=18000"
set "WRAPPER_PORT=17999"
set "MAX_WAIT=30"
set "MAX_WAIT_EMBED=120"

echo [INFO] Starting service health checks...
echo.

:: ==================== Check all services (curl) ====================
echo [CHECK] SurrealDB (port %SURREALDB_PORT%)...
curl -sf http://localhost:%SURREALDB_PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Already running
    set "SURREALDB_STATUS=running"
) else (
    echo   [--] Not running
    set "SURREALDB_STATUS=stopped"
)

echo [CHECK] Embedding Service (port %EMBEDDING_PORT%)...
curl -sf http://localhost:%EMBEDDING_PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Already running
    set "EMBEDDING_STATUS=running"
) else (
    echo   [--] Not running
    set "EMBEDDING_STATUS=stopped"
)

echo [CHECK] Wrapper Service (port %WRAPPER_PORT%)...
curl -sf http://localhost:%WRAPPER_PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Already running
    set "WRAPPER_STATUS=running"
) else (
    echo   [--] Not running
    set "WRAPPER_STATUS=stopped"
)

echo.

:: ==================== 1. Start SurrealDB ====================
if "!SURREALDB_STATUS!"=="running" goto skip_surrealdb

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
echo [START] Launching SurrealDB...
start "SurrealDB Database" "%SURREAL_PATH%" start --bind 0.0.0.0:%SURREALDB_PORT% --user root --pass root rocksdb://%DATA_DIR%

echo [WAIT] Waiting for SurrealDB...
set /a count=0
:wait_surrealdb
timeout /t 1 /nobreak >nul
set /a count+=1
curl -sf http://localhost:%SURREALDB_PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] SurrealDB is ready
    goto skip_surrealdb
)
if !count! lss %MAX_WAIT% goto wait_surrealdb
echo [ERROR] SurrealDB failed to start within %MAX_WAIT%s
pause
exit /b 1

:skip_surrealdb
echo.

:: ==================== 2. Start Embedding Service ====================
if "!EMBEDDING_STATUS!"=="running" goto skip_embedding

echo [START] Launching Embedding Service...
start "Qwen3-Embedding Service" "%PROJECT_DIR%\start_embedding_service.bat"

echo [WAIT] Waiting for Embedding Service (model loading)...
set /a count=0
:wait_embedding
timeout /t 3 /nobreak >nul
set /a count+=1
set /a elapsed=count*3
<nul set /p ="  !elapsed!s ..."
curl -sf http://localhost:%EMBEDDING_PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo [OK] Embedding Service is ready ^(!elapsed!s^)
    goto skip_embedding
)
if !count! lss %MAX_WAIT_EMBED% goto wait_embedding
echo [ERROR] Embedding Service failed to start within %MAX_WAIT_EMBED%s
pause
exit /b 1

:skip_embedding
echo.

:: ==================== 3. Start Wrapper Service ====================
if "!WRAPPER_STATUS!"=="running" goto skip_wrapper

echo [START] Launching Wrapper Service...
start "Wrapper Service" "%PROJECT_DIR%\start_wrapper_service.bat"

echo [WAIT] Waiting for Wrapper Service...
set /a count=0
:wait_wrapper
timeout /t 2 /nobreak >nul
set /a count+=1
set /a elapsed=count*2
<nul set /p ="  !elapsed!s ..."
curl -sf http://localhost:%WRAPPER_PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo [OK] Wrapper Service is ready ^(!elapsed!s^)
    goto skip_wrapper
)
if !count! lss %MAX_WAIT% goto wait_wrapper
echo [ERROR] Wrapper Service failed to start within %MAX_WAIT%s
pause
exit /b 1

:skip_wrapper
echo.

:: ==================== Final status ====================
echo ============================================
echo All services are ready!
echo ============================================
echo.
echo Endpoints:
echo   SurrealDB:         ws://localhost:%SURREALDB_PORT%/rpc
echo   Embedding Service: http://localhost:%EMBEDDING_PORT%/health
echo   Wrapper Service:   http://localhost:%WRAPPER_PORT%/health
echo   WebSocket:         ws://localhost:%WRAPPER_PORT%/ws/memories/live
echo.
echo Press any key to close this window...
pause >nul
