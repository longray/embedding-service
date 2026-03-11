@echo off
chcp 65001 >nul

title SurrealDB Schema Initialization
setlocal EnableDelayedExpansion

:: ==================== Configuration ====================
set "PROJECT_DIR=D:\embedding_service"
set "SURREAL_PATH=C:\Users\Longray\AppData\Local\SurrealDB\surreal.exe"
set "SCHEMA_FILE=%PROJECT_DIR%\scripts\init_surrealdb.surql"
set "PORT=18002"
set "HOST=localhost"
set "USERNAME=root"
set "PASSWORD=root"
set "NAMESPACE=default"
set "DATABASE=default"

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
echo      SurrealDB Schema Initialization
echo ============================================
echo %RESET%
echo.

:: ==================== Pre-checks ====================
echo %YELLOW%[CHECK] Checking SurrealDB executable...%RESET%

if not exist "%SURREAL_PATH%" (
    echo %RED%[ERROR] surreal.exe not found: %SURREAL_PATH%%RESET%
    echo Please ensure SurrealDB is installed correctly
    pause
    exit /b 1
)

echo %GREEN%[PASS] SurrealDB executable found%RESET%
echo.

echo %YELLOW%[CHECK] Checking schema file...%RESET%

if not exist "%SCHEMA_FILE%" (
    echo %RED%[ERROR] Schema file not found: %SCHEMA_FILE%%RESET%
    pause
    exit /b 1
)

echo %GREEN%[PASS] Schema file found%RESET%
echo.

:: ==================== SurrealDB Health Check ====================
echo %YELLOW%[CHECK] Checking if SurrealDB is running...%RESET%

curl -sf http://%HOST%:%PORT%/health >nul 2>&1

if %errorlevel% neq 0 (
    echo %RED%[ERROR] SurrealDB is not running on %HOST%:%PORT%%RESET%
    echo Please start SurrealDB first using: start_surrealdb.bat
    pause
    exit /b 1
)

echo %GREEN%[PASS] SurrealDB is running%RESET%
echo.

:: ==================== Import Schema ====================
echo %BLUE%[START] Importing schema...%RESET%
echo %YELLOW%-------------------------------------------%RESET%
echo Connection: http://%HOST%:%PORT%
echo Namespace:  %NAMESPACE%
echo Database:   %DATABASE%
echo Schema:     %SCHEMA_FILE%
echo %YELLOW%-------------------------------------------%RESET%
echo.

"%SURREAL_PATH%" import --conn http://%HOST%:%PORT% --user %USERNAME% --pass %PASSWORD% --ns %NAMESPACE% --db %DATABASE% "%SCHEMA_FILE%"

if %errorlevel% equ 0 (
    echo.
    echo %GREEN%[OK] Schema imported successfully!%RESET%
) else (
    echo.
    echo %RED%[ERROR] Schema import failed (code: %errorlevel%)%RESET%
)

echo.
echo Press any key to close window...
pause >nul
