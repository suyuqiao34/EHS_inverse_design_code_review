@echo off
chcp 65001 >nul
echo ============================================================
echo 🚀 Starting Eutectic Hydrated Salt Intelligent Design System
echo ============================================================
echo.

cd /d %~dp0

echo Checking Python environment...
python --version
if errorlevel 1 (
    echo ❌ Error: Python not found, please install Python first
    pause
    exit /b 1
)

echo.
echo Starting Web service...
echo.
echo 📍 Access URL: http://localhost:5000
echo 📍 Or use: http://127.0.0.1:5000
echo.
echo 💡 Note: Press Ctrl+C to stop the service
echo ============================================================
echo.

python app.py

pause
