@echo off
chcp 65001 >nul

title Embedding Service - Test Runner
setlocal EnableDelayedExpansion

:: Configuration
set "PROJECT_DIR=D:\embedding_service"
set "UV_PATH=C:\Users\Longray\.local\bin\uv.exe"
set "EMBEDDING_PORT=18000"
set "WRAPPER_PORT=18008"
set "SURREALDB_PORT=18002"
set "HEALTH_TIMEOUT=3"

:: Color definitions
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "CYAN=[96m"
set "RESET=[0m"

echo.
echo ============================================
echo   Embedding Service - Test Runner
echo ============================================
echo.

:: ==================== Pre-checks ====================
echo %YELLOW%[CHECK] Checking services...%RESET%

set "ALL_OK=1"

:: Check SurrealDB
curl -sf http://localhost:%SURREALDB_PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo   %GREEN%[OK]%RESET% SurrealDB        (port %SURREALDB_PORT%)
) else (
    echo   %RED%[--]%RESET% SurrealDB        (port %SURREALDB_PORT%)
    set "ALL_OK=0"
)

:: Check Embedding Service
curl -sf http://localhost:%EMBEDDING_PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo   %GREEN%[OK]%RESET% Embedding Service (port %EMBEDDING_PORT%)
) else (
    echo   %RED%[--]%RESET% Embedding Service (port %EMBEDDING_PORT%)
    set "ALL_OK=0"
)

:: Check Wrapper Service
curl -sf http://localhost:%WRAPPER_PORT%/health >nul 2>&1
if !errorlevel! equ 0 (
    echo   %GREEN%[OK]%RESET% Wrapper Service   (port %WRAPPER_PORT%)
) else (
    echo   %RED%[--]%RESET% Wrapper Service   (port %WRAPPER_PORT%)
    set "ALL_OK=0"
)

echo.

if "%ALL_OK%"=="0" (
    echo %RED%[WARNING] Some services are not running.%RESET%
    echo Tests that depend on missing services will fail.
    echo.
    echo Start all services first: start_all_services.bat
    echo.
    set /p "CONTINUE=Continue anyway? (y/N): "
    if /i not "!CONTINUE!"=="y" (
        exit /b 1
    )
    echo.
)

:: ==================== Run Tests ====================
cd /d "%PROJECT_DIR%"

echo %BLUE%[TEST] Running API tests...%RESET%
echo ============================================
echo.

"%UV_PATH%" run python -m pytest tests/test_wrapper_api.py -v --tb=short

set "TEST_EXIT=%errorlevel%"
echo.

if %TEST_EXIT% equ 0 (
    echo %GREEN%[OK] All tests passed!%RESET%
) else (
    echo %RED%[FAIL] Some tests failed (exit code: %TEST_EXIT%)%RESET%
)

echo.
echo %YELLOW%Press any key to close window...%RESET%
pause >nul
