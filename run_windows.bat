@echo off
setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

%PYTHON_EXE% start.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Projection Mapping Tool exited with code %EXIT_CODE%.
    echo If this is your first run, install dependencies with:
    echo     pip install -r requirements.txt
    echo.
    pause
)

endlocal
exit /b %EXIT_CODE%
