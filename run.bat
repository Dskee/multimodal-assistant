@echo off
rem ============================================
rem  Multimodal Voice Assistant - Launcher
rem  Uses the project-local venv (inherits Anaconda
rem  base deps; typing_extensions fixed inside).
rem  NOTE: keep this file pure ASCII (English only)
rem  so cmd parses it correctly on any code page.
rem ============================================
cd /d %~dp0

set "PY=venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] venv not found: %PY%
    echo.
    echo Please create it first:
    echo   python -m venv --system-site-packages venv
    echo   venv\Scripts\python.exe -m pip install typing_extensions==4.16.0
    echo.
    pause
    exit /b 1
)

"%PY%" main.py
if errorlevel 1 (
    echo.
    echo [ERROR] The program exited with an error. See messages above.
    pause
)
