@echo off
REM ============================================================
REM PRISM Engine Launcher (Windows)
REM ============================================================
REM Double-click this file to launch!
REM ============================================================

title PRISM Engine Launcher

REM Get directory where this script lives
set SCRIPT_DIR=%~dp0
REM Go up one level to project root
cd /d "%SCRIPT_DIR%.."
set PROJECT_ROOT=%cd%

echo ========================================
echo   PRISM Engine Launcher
echo ========================================
echo Project root: %PROJECT_ROOT%
echo.

REM Try to find and activate venv
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    echo [OK] Activating venv...
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
    python --version
) else if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    echo [OK] Activating .venv...
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
    python --version
) else (
    echo [!] No venv found! Creating one...
    python -m venv "%PROJECT_ROOT%\venv"
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
    
    if exist "%PROJECT_ROOT%\requirements.txt" (
        pip install -r "%PROJECT_ROOT%\requirements.txt"
    )
)

echo.
echo ========================================
echo   Ready! Choose an option:
echo ========================================
echo   1) Open Python REPL
echo   2) Run Jupyter Notebook  
echo   3) Run main.py
echo   4) Just open command prompt here
echo   Q) Quit
echo.

set /p choice="Choice: "

if "%choice%"=="1" python
if "%choice%"=="2" jupyter notebook
if "%choice%"=="3" python "%PROJECT_ROOT%\main.py"
if "%choice%"=="4" cmd /k
if /i "%choice%"=="q" exit

REM Keep window open if something went wrong
pause
