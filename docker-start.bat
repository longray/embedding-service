@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo.
echo ============================================
echo    Docker - Embedding Service Startup
echo ============================================
echo.

:: Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found. Please install Docker Desktop.
    pause
    exit /b 1
)

:: Create data directories
if not exist "docker-data\surrealdb" mkdir docker-data\surrealdb
if not exist "docker-data\meilisearch" mkdir docker-data\meilisearch
if not exist "docker-data\models" mkdir docker-data\models

:: Check .env
if not exist ".env" (
    echo [WARN] .env not found, using defaults
)

echo [INFO] Starting services...
echo.

:: Start base services (no GPU)
docker-compose up -d surrealdb meilisearch wrapper

echo [WAIT] Waiting for services to be healthy...
echo.

:: Wait for SurrealDB
set /a count=0
:wait_surrealdb
timeout /t 2 /nobreak >nul
set /a count+=1
curl -sf http://localhost:28002/health >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] SurrealDB (28002)
    goto surrealdb_ready
)
if !count! lss 15 goto wait_surrealdb
echo [WARN] SurrealDB health check timeout, continuing...
:surrealdb_ready

:: Wait for Meilisearch
set /a count=0
:wait_meili
timeout /t 2 /nobreak >nul
set /a count+=1
curl -sf http://localhost:28003/health >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Meilisearch (28003)
    goto meili_ready
)
if !count! lss 15 goto wait_meili
echo [WARN] Meilisearch health check timeout, continuing...
:meili_ready

:: Wait for Wrapper
set /a count=0
:wait_wrapper
timeout /t 2 /nobreak >nul
set /a count+=1
curl -sf http://localhost:27999/health >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Wrapper (27999)
    goto wrapper_ready
)
if !count! lss 15 goto wait_wrapper
echo [WARN] Wrapper health check timeout, continuing...
:wrapper_ready

echo.
echo ============================================
echo Services ready!
echo ============================================
echo.
echo Endpoints:
echo   Wrapper:     http://localhost:27999
echo   SurrealDB:   ws://localhost:28002/rpc
echo   Meilisearch: http://localhost:28003
echo.
echo GPU services (Embedding/LLM):
echo   docker-compose --profile gpu up -d
echo.
