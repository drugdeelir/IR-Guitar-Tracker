@echo off
echo Starting Projection Mapper for Windows...

:: Check if requirements are installed
echo Checking dependencies...
pip install -r requirements.txt

:: Launch the application
echo Launching...
python main.py

if %errorlevel% neq 0 (
    echo Error: Application failed to start.
    pause
)
