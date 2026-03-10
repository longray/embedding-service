@echo off
chcp 65001 >nul

title SurrealDB Database Service
setlocal EnableDelayedExpansion

:: ==================== Configuration ====================
set "PROJECT_DIR=D:\embedding_service"
set "DATA_DIR=%PROJECT_DIR%\surrealdb_data"
set "LOG_FILE=%PROJECT_DIR%\surrealdb.log"
set "PORT=8000"
set "HOST=0.0.0.0"
set "USERNAME=root"
set "PASSWORD=root"

:: SurrealDB executable paths (priority order)
set "SURREAL_PATH1=C:\Users\Longray\AppData\Local\SurrealDB\surreal.exe"
set "SURREAL_PATH2=C:\Program Files\SurrealDB\surreal.exe"
set "SURREAL_PATH="

:: ==================== Color Definitions ====================
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "CYAN=[96m"
set "RESET=[0m"

:: ==================== Banner ====================
echo %CYAN%
echo ============================================
echo       SurrealDB Database Service
echo ============================================
echo %RESET%
echo.

:: ==================== Find SurrealDB ====================
echo %YELLOW%[Check] SurrealDB executable...%RESET%

if exist "%SURREAL_PATH1%" (
    set "SURREAL_PATH=%SURREAL_PATH1%"
    echo %GREEN%[Found] %SURREAL_PATH1%%RESET%
) else if exist "%SURREAL_PATH2%" (
    set "SURREAL_PATH=%SURREAL_PATH2%"
    echo %GREEN%[Found] %SURREAL_PATH2%%RESET%
) else (
    echo %RED%[Error] surreal.exe not found%RESET%
    echo.
    echo Please install SurrealDB:
    echo   1. Visit https://surrealdb.com/install
    echo   2. Or run: winget install SurrealDB.SurrealDB
    echo.
    pause
    exit /b 1
)

:: ==================== Data Directory Check ====================
echo %YELLOW%[Check] Data directory...%RESET%

if not exist "%DATA_DIR%" (
    echo %YELLOW%[Create] Data directory: %DATA_DIR%%RESET%
    mkdir "%DATA_DIR%"
)

echo %GREEN%[Pass] Data directory: %DATA_DIR%%RESET%
echo.

:: ==================== Port Check ====================
echo %YELLOW%[Check] Port %PORT% availability...%RESET%

netstat -ano | findstr ":%PORT%" >nul
if %errorlevel% equ 0 (
    echo %RED%[Warning] Port %PORT% is already in use%RESET%
    echo.
    echo A SurrealDB instance may already be running.
    echo Please stop the existing instance or change the port.
    echo.
    echo Check with: netstat -ano ^| findstr ":%PORT%"
    echo.
    pause
    exit /b 1
)

echo %GREEN%[Pass] Port %PORT% is available%RESET%
echo.

:: ==================== Start Service ====================
echo %BLUE%[Start] Starting SurrealDB service...%RESET%
echo %YELLOW%-------------------------------------------%RESET%
echo Data Dir:    %DATA_DIR%
echo Listen:      http://%HOST%:%PORT%
echo WebSocket:   ws://localhost:%PORT%/rpc
echo Username:    %USERNAME%
echo Password:    %PASSWORD%
echo Log File:    %LOG_FILE%
echo %YELLOW%-------------------------------------------%RESET%
echo.
echo %CYAN%Tip: Press Ctrl+C to stop the service%RESET%
echo.

:: Change to project directory
cd /d "%PROJECT_DIR%"

:: Start SurrealDB (file persistence mode)
"%SURREAL_PATH%" start --bind "%HOST%:%PORT%" --user %USERNAME% --pass %PASSWORD% "file://%DATA_DIR%"

:: Service exit handler
echo.
echo %RED%[Stop] Service exited (code: %errorlevel%)%RESET%
echo Press any key to close...
pause >nul