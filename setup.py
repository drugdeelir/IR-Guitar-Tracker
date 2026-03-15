"""
Legacy build helper.

This project is now Windows-first and uses PyInstaller for packaging.

Usage (Windows cmd.exe):
    pyinstaller --noconfirm --name ProjectionMapper --windowed --add-data "style.qss;." --add-data "logo.png;." main.py
"""

from pathlib import Path
import subprocess
import sys


def main():
    project_root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        "ProjectionMapper",
        "--windowed",
        "--add-data",
        "style.qss;.",
        "--add-data",
        "logo.png;.",
        "main.py",
    ]

    print("Building Windows executable with PyInstaller...")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
