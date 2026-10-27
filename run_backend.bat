@echo off
setlocal
title Image Captioning - Backend
cd /d "%~dp0"

echo ========================================================
echo        Starting Image Captioning Backend (FastAPI)
echo ========================================================
echo.

:: Check for virtual environment python
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found! Please create .venv or install Python.
        echo.
        pause
        exit /b 1
    )
    set "PYTHON_EXEC=python"
    echo [!] Virtual environment not found. Using system python.
)

echo [*] Starting Uvicorn server on http://127.0.0.1:8000 ...
echo.

"%PYTHON_EXEC%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1 %*

if errorlevel 1 (
    echo.
    echo [ERROR] Backend exited with error code %errorlevel%.
    pause
)
