@echo off
cd /d E:\ai_music\backend

echo ========================================
echo   AI Music - Score Generator
echo ========================================
echo.

set PYTHONPATH=E:\lib\site-packages;%PYTHONPATH%
set HF_ENDPOINT=https://hf-mirror.com

echo Checking dependencies...
python -c "import uvicorn,fastapi,librosa,demucs,soundfile,funasr; print('All dependencies OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies, please wait...
    python -m pip install --target=E:\lib\site-packages -r requirements.txt
    if %errorlevel% neq 0 (
        echo Dependency installation failed. Check your network.
        pause
        exit /b 1
    )
)

echo Starting server...
echo.
echo Open: http://localhost:8020
echo Press Ctrl+C to stop
echo ========================================
echo.

start http://localhost:8020
python -m uvicorn main:app --host 0.0.0.0 --port 8020

pause
