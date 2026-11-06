@echo off
cd /d %~dp0

echo ============================================================
echo  Eutectic hydrated salt design system
echo ============================================================

python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.11 first.
    pause
    exit /b 1
)

if "%DASHSCOPE_API_KEY%"=="" (
    echo Warning: DASHSCOPE_API_KEY is not set. LLM agents will be unavailable.
    echo Set it first, e.g.  set DASHSCOPE_API_KEY=your_key
)

echo Open http://127.0.0.1:5000 in a browser. Press Ctrl+C to stop.
python app.py

pause
