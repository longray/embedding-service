@echo off
chcp 65001 >nul

title Health Check Test
echo.
echo ============================================
echo    Health Check Test Script
echo ============================================
echo.

set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "CYAN=[96m"
set "RESET=[0m"

:: Test 1: Embedding Service Health
echo %BLUE%[TEST 1] Embedding Service Health Check%RESET%
echo   URL: http://localhost:18000/health

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:18000/health' -Method GET -TimeoutSec 3 -UseBasicParsing; Write-Host \"Status Code: $($r.StatusCode)\"; Write-Host \"Response: $($r.Content)\"; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { Write-Host \"Error: $_\"; exit 1 }"

if %errorlevel% equ 0 (
    echo %GREEN%   [PASS] Embedding Service is healthy%RESET%
) else (
    echo %RED%   [FAIL] Embedding Service is not responding%RESET%
)
echo.

:: Test 2: SurrealDB Health (TCP Socket)
echo %BLUE%[TEST 2] SurrealDB TCP Socket Check%RESET%
echo   Host: localhost:8000

powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('localhost', 8000); Write-Host \"Connected successfully\"; $c.Close(); exit 0 } catch { Write-Host \"Error: $_\"; exit 1 }"

if %errorlevel% equ 0 (
    echo %GREEN%   [PASS] SurrealDB is accepting connections%RESET%
) else (
    echo %RED%   [FAIL] SurrealDB is not responding%RESET%
)
echo.

:: Test 3: Port Check
echo %BLUE%[TEST 3] Port Status Check%RESET%
echo.

echo   Port 18000 (Embedding):
netstat -ano | findstr ":18000" >nul
if %errorlevel% equ 0 (
    echo   %GREEN%   [OCCUPIED] Port is in use%RESET%
) else (
    echo   %YELLOW%   [FREE] Port is available%RESET%
)

echo   Port 8000 (SurrealDB):
netstat -ano | findstr ":8000" >nul
if %errorlevel% equ 0 (
    echo   %GREEN%   [OCCUPIED] Port is in use%RESET%
) else (
    echo   %YELLOW%   [FREE] Port is available%RESET%
)
echo.

echo ============================================
echo %CYAN%Test completed.%RESET%
echo.
echo %YELLOW%Press any key to exit...%RESET%
pause >nul
