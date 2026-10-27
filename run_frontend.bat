@echo off
setlocal
title Image Captioning - Frontend
cd /d "%~dp0"

echo ========================================================
echo        Starting Image Captioning Frontend (PySide6)
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

echo [*] Launching Frontend with: %PYTHON_EXEC%
echo.

"%PYTHON_EXEC%" -m frontend.main %*

if errorlevel 1 (
    echo.
    echo [ERROR] Frontend exited with error code %errorlevel%.
    pause
)
