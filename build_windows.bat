@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

pyinstaller --noconfirm --windowed --name "ProjectionMapper" --add-data "style.qss;." --add-data "logo.png;." main.py

echo Build complete. EXE is in dist\ProjectionMapper
