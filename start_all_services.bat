@echo off
chcp 65001 >nul

title Embedding Service - Smart Startup
echo.
echo ============================================
echo    Embedding Service - Smart Startup
echo ============================================
echo.

:: Configuration
set "SCRIPT_DIR=%~dp0"
set "EMBEDDING_PORT=18000"
set "SURREALDB_PORT=8000"
set "HEALTH_TIMEOUT=5"
set "MAX_WAIT=30"

:: Color definitions
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "CYAN=[96m"
set "RESET=[0m"

echo %BLUE%[INFO] Starting service health checks...%RESET%
echo.

:: Check Embedding Service Health
echo %YELLOW%[CHECK] Embedding Service (port %EMBEDDING_PORT%)...%RESET%

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%EMBEDDING_PORT%/health' -Method GET -TimeoutSec %HEALTH_TIMEOUT% -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1

if %errorlevel% equ 0 (
    echo %GREEN%[OK] Embedding Service is already running and healthy%RESET%
    set "EMBEDDING_STATUS=running"
) else (
    echo %YELLOW%[INFO] Embedding Service not running, will start...%RESET%
    set "EMBEDDING_STATUS=stopped"
)

echo.

:: Check SurrealDB Health
echo %YELLOW%[CHECK] SurrealDB (port %SURREALDB_PORT%)...%RESET%

powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('localhost', %SURREALDB_PORT%); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1

if %errorlevel% equ 0 (
    echo %GREEN%[OK] SurrealDB is already running and accepting connections%RESET%
    set "SURREALDB_STATUS=running"
) else (
    echo %YELLOW%[INFO] SurrealDB not running, will start...%RESET%
    set "SURREALDB_STATUS=stopped"
)

echo.
echo ============================================
echo %BLUE%[INFO] Service Status Summary:%RESET%
if "%EMBEDDING_STATUS%"=="running" (
    echo   - Embedding Service: %GREEN%RUNNING%RESET%
) else (
    echo   - Embedding Service: %YELLOW%STARTING...%RESET%
)
if "%SURREALDB_STATUS%"=="running" (
    echo   - SurrealDB: %GREEN%RUNNING%RESET%
) else (
    echo   - SurrealDB: %YELLOW%STARTING...%RESET%
)
echo ============================================
echo.

:: Start SurrealDB if needed
if "%SURREALDB_STATUS%"=="stopped" (
    echo %BLUE%[START] Launching SurrealDB...%RESET%
    start "SurrealDB Database" cmd /c "%SCRIPT_DIR%start_surrealdb.bat"
    
    echo %YELLOW%[WAIT] Waiting for SurrealDB to be ready...%RESET%
    set /a count=0
    :wait_surrealdb
    timeout /t 1 /nobreak >nul
    set /a count+=1
    
    powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('localhost', %SURREALDB_PORT%); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
    
    if %errorlevel% neq 0 (
        if %count% lss %MAX_WAIT% (
            goto wait_surrealdb
        ) else (
            echo %RED%[ERROR] SurrealDB failed to start within %MAX_WAIT% seconds%RESET%
            pause
            exit /b 1
        )
    )
    
    echo %GREEN%[OK] SurrealDB is ready%RESET%
    echo.
)

:: Start Embedding Service if needed
if "%EMBEDDING_STATUS%"=="stopped" (
    echo %BLUE%[START] Launching Embedding Service...%RESET%
    start "Qwen3-Embedding Service" cmd /c "%SCRIPT_DIR%start_embedding_service.bat"
    
    echo %YELLOW%[WAIT] Waiting for Embedding Service to be ready...%RESET%
    set /a count=0
    :wait_embedding
    timeout /t 1 /nobreak >nul
    set /a count+=1
    
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%EMBEDDING_PORT%/health' -Method GET -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    
    if %errorlevel% neq 0 (
        if %count% lss %MAX_WAIT% (
            goto wait_embedding
        ) else (
            echo %RED%[ERROR] Embedding Service failed to start within %MAX_WAIT% seconds%RESET%
            pause
            exit /b 1
        )
    )
    
    echo %GREEN%[OK] Embedding Service is ready%RESET%
    echo.
)

:: Final status
echo.
echo ============================================
echo %GREEN%All services are ready!%RESET%
echo ============================================
echo.
echo %CYAN%Available endpoints:%RESET%
echo   - Embedding Service: http://localhost:%EMBEDDING_PORT%/health
echo   - SurrealDB:         ws://localhost:%SURREALDB_PORT%/rpc
echo.
echo %YELLOW%Press any key to close this window...%RESET%
pause >nul
