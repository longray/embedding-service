@echo off
chcp 65001 >nul
echo.
echo ============================================
echo    Docker - Embedding Service Shutdown
echo ============================================
echo.

docker-compose down

echo.
echo All services stopped.
echo Data preserved in docker-data/
echo.
