@echo off
chcp 65001 >nul

title All Services Orchestrator
setlocal EnableDelayedExpansion

:: ==================== Configuration ====================
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=D:\embedding_service"
set "EMBEDDING_PORT=18000"
set "SURREALDB_PORT=8000"
set "WRAPPER_PORT=17999"
set "HEALTH_TIMEOUT=5"
set "MAX_WAIT=30"

:: ==================== Startup Banner ====================
echo.
echo ============================================
echo    All Services Orchestrator
echo ============================================
echo.

:: ==================== Health Checks ====================
echo [CHECK] Performing service health checks...
echo.

:: Check Wrapper Service Health
echo [CHECK] Checking Wrapper Service on port %WRAPPER_PORT%...

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%WRAPPER_PORT%/health' -Method GET -TimeoutSec %HEALTH_TIMEOUT% -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1

if %errorlevel% equ 0 (
    echo [OK] Wrapper Service is already running and healthy
    set "WRAPPER_STATUS=running"
) else (
    echo [INFO] Wrapper Service not running, will start new instance
    set "WRAPPER_STATUS=stopped"
)

echo.

:: Check Embedding Service Health
echo [CHECK] Checking Embedding Service on port %EMBEDDING_PORT%...

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%EMBEDDING_PORT%/health' -Method GET -TimeoutSec %HEALTH_TIMEOUT% -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1

if %errorlevel% equ 0 (
    echo [OK] Embedding Service is already running and healthy
    set "EMBEDDING_STATUS=running"
) else (
    echo [INFO] Embedding Service not running, will start new instance
    set "EMBEDDING_STATUS=stopped"
)

echo.

:: Check SurrealDB Health
echo [CHECK] Checking SurrealDB on port %SURREALDB_PORT%...

powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('localhost', %SURREALDB_PORT%); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1

if %errorlevel% equ 0 (
    echo [OK] SurrealDB is already running and accepting connections
    set "SURREALDB_STATUS=running"
) else (
    echo [INFO] SurrealDB not running, will start new instance
    set "SURREALDB_STATUS=stopped"
)

echo.

:: ==================== Status Summary ====================
echo ============================================
echo [INFO] Service Status Summary:
if "%WRAPPER_STATUS%"=="running" (
    echo   - Wrapper Service:   RUNNING
) else (
    echo   - Wrapper Service:   STARTING
)
if "%EMBEDDING_STATUS%"=="running" (
    echo   - Embedding Service: RUNNING
) else (
    echo   - Embedding Service: STARTING
)
if "%SURREALDB_STATUS%"=="running" (
    echo   - SurrealDB:         RUNNING
) else (
    echo   - SurrealDB:         STARTING
)
echo ============================================
echo.

:: ==================== Start SurrealDB if needed ====================
if "%SURREALDB_STATUS%"=="stopped" (
    echo [START] Launching SurrealDB...
    start "SurrealDB Database" cmd /c "%SCRIPT_DIR%start_surrealdb.bat"
    
    echo [WAIT] Waiting for SurrealDB to be ready, max %MAX_WAIT% seconds...
    set /a count=0
    :wait_surrealdb
    timeout /t 1 /nobreak >nul
    set /a count+=1
    
    powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('localhost', %SURREALDB_PORT%); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
    
    if !errorlevel! neq 0 (
        if !count! lss %MAX_WAIT% (
            goto wait_surrealdb
        ) else (
            echo [ERROR] SurrealDB failed to start within %MAX_WAIT% seconds
            exit /b 1
        )
    )
    
    echo [OK] SurrealDB is ready and accepting connections
    echo.
)

:: ==================== Start Embedding Service if needed ====================
if "%EMBEDDING_STATUS%"=="stopped" (
    echo [START] Launching Embedding Service...
    start "Qwen3-Embedding Service" cmd /c "%SCRIPT_DIR%start_embedding_service.bat"
    
    echo [WAIT] Waiting for Embedding Service to be ready, max %MAX_WAIT% seconds...
    set /a count=0
    :wait_embedding
    timeout /t 1 /nobreak >nul
    set /a count+=1
    
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%EMBEDDING_PORT%/health' -Method GET -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    
    if !errorlevel! neq 0 (
        if !count! lss %MAX_WAIT% (
            goto wait_embedding
        ) else (
            echo [ERROR] Embedding Service failed to start within %MAX_WAIT% seconds
            exit /b 1
        )
    )
    
    echo [OK] Embedding Service is ready and healthy
    echo.
)

:: ==================== Start Wrapper Service if needed ====================
if "%WRAPPER_STATUS%"=="stopped" (
    echo [START] Launching Wrapper Service...
    start "Wrapper Service" cmd /c "cd /d %PROJECT_DIR% && uv run python -m wrapper.src.main"
    
    echo [WAIT] Waiting for Wrapper Service to be ready, max %MAX_WAIT% seconds...
    set /a count=0
    :wait_wrapper
    timeout /t 1 /nobreak >nul
    set /a count+=1
    
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%WRAPPER_PORT%/health' -Method GET -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    
    if !errorlevel! neq 0 (
        if !count! lss %MAX_WAIT% (
            goto wait_wrapper
        ) else (
            echo [ERROR] Wrapper Service failed to start within %MAX_WAIT% seconds
            exit /b 1
        )
    )
    
    echo [OK] Wrapper Service is ready and healthy
    echo.
)

:: ==================== Final Status ====================
echo.
echo ============================================
echo [OK] All services are ready and operational
echo ============================================
echo.

:: ==================== Wrapper Health Summary ====================
echo [INFO] Wrapper Service Health Summary:
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%WRAPPER_PORT%/health' -Method GET -TimeoutSec 5 -UseBasicParsing; $j = $r.Content | ConvertFrom-Json; Write-Host ('  Status: ' + $j.status); Write-Host ('  Service: ' + $j.service); Write-Host ('  Version: ' + $j.version); Write-Host ('  Port: ' + $j.port); Write-Host ('  Embedding: ' + $j.embedding_service.status); Write-Host ('  SurrealDB: ' + $j.surrealdb.status); Write-Host ('  Cache: ' + $j.cache_stats.current_size + '/' + $j.cache_stats.max_size) } catch { Write-Host '  Failed to retrieve health status' }"

echo.
echo Available Endpoints:
echo   - Wrapper Service:   http://localhost:%WRAPPER_PORT%/health
echo   - Embedding Service: http://localhost:%EMBEDDING_PORT%/health
echo   - SurrealDB:         ws://localhost:%SURREALDB_PORT%/rpc
echo.
echo Window will remain open. Close manually when done.
echo.
echo Press any key to close window...
pause >nul