@echo off
:: IR Guitar Tracker — Windows launcher
:: Double-click this file to start the application.
:: The working directory is forced to the folder containing this script
:: so that relative asset paths (style.qss, logo.mkv, settings.json) resolve correctly.

cd /d "%~dp0"
python start.py
pause
