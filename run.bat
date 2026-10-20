@echo off
rem ---------------------------------------------------------------
rem  ASCII-only on purpose: cmd.exe parses .bat files with the OEM
rem  codepage, so non-ASCII characters here become garbage commands.
rem ---------------------------------------------------------------
setlocal
chcp 65001 >nul

cd /d "%~dp0"

rem ---- locate python: project venv first, then PATH ----
set "PY_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PY_EXE if exist "%~dp0venv\Scripts\python.exe" set "PY_EXE=%~dp0venv\Scripts\python.exe"
if not defined PY_EXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PY_EXE set "PY_EXE=%%i"
    )
)

if not defined PY_EXE (
    echo.
    echo [ERROR] Python not found.
    echo   1. Install Python 3.10+ from https://www.python.org/downloads/
    echo      and tick "Add python.exe to PATH".
    echo   2. Or create a virtual env in this folder:
    echo        python -m venv .venv
    echo        .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem ---- prefer pythonw so the GUI starts without a console window ----
set "PYW_EXE=%PY_EXE:python.exe=pythonw.exe%"
if not exist "%PYW_EXE%" set "PYW_EXE=%PY_EXE%"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

rem ---- console mode: run.bat --console keeps the classic CLI workflow ----
if "%~1"=="--console" (
    shift
    "%PY_EXE%" "%~dp0main.py" %*
    echo.
    echo Workflow exited. Press any key to close this window.
    pause >nul
    exit /b 0
)

rem ---- default: visual console, no black cmd window ----
start "" "%PYW_EXE%" "%~dp0gui.py"
exit /b 0
