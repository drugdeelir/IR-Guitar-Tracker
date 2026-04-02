import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES_TO_VALIDATE = ["main.py", "worker.py", "splash.py"]

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_python_version():
    if sys.version_info < (3, 9):
        print(f"\nError: Python 3.9 or newer is required.")
        print(f"       Running Python {sys.version}")
        print("       Download the latest Python from https://python.org\n")
        input("Press Enter to exit.")
        sys.exit(1)
    if sys.platform != 'win32':
        print("\nWarning: This application is designed for Windows 10/11.")
        print(f"         Running on {sys.platform}. Some features may not work.\n")


def check_camera():
    """Warn if no camera is accessible (non-fatal — camera may be off at launch)."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        cap.release()
        if not opened:
            print("\nWarning: No camera detected on index 0.")
            print("         Connect the IR camera before starting calibration.\n")
    except Exception:
        pass  # opencv not yet installed — skip this check


def check_codec():
    """Warn if H.264 write support is unavailable (cue videos may not play back)."""
    try:
        import cv2, tempfile, os
        tmp = tempfile.mktemp(suffix='.mp4')
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), 1, (2, 2))
        ok = writer.isOpened()
        writer.release()
        try:
            os.remove(tmp)
        except OSError:
            pass
        if not ok:
            print("\nWarning: Video codec (mp4v) unavailable.")
            print("         Cue video playback may fail. Install K-Lite Codec Pack or VLC.\n")
    except Exception:
        pass


def check_screens():
    """Warn if only one screen is detected (projector likely not connected)."""
    try:
        import cv2  # noqa — ensure OpenCV env is ready before importing Qt
        from PyQt5.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication([])
        n = len(_app.screens())
        if n < 2:
            print(f"\nWarning: Only {n} screen detected.")
            print("         Connect the projector before starting for best results.\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Existing validation helpers
# ---------------------------------------------------------------------------

def contains_diff_markers(path: Path) -> bool:
    try:
        first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:4]
    except OSError:
        return False
    markers = ("diff --git", "index ", "--- a/", "+++ b/")
    return any(line.startswith(markers) for line in first)


def run(cmd: list) -> int:
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    check_python_version()

    for rel in FILES_TO_VALIDATE:
        path = ROOT / rel
        if contains_diff_markers(path):
            print(f"\nError: {rel} appears to contain git diff text, not runnable source code.")
            print("Please re-download a clean copy of the repo (or restore the file) and try again.\n")
            return 1

    print("Ensuring Python requirements are installed...")
    pip_code = run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if pip_code != 0:
        print("\nWarning: dependency install failed. Attempting to start anyway...\n")

    check_code = run([sys.executable, "-m", "py_compile", *FILES_TO_VALIDATE])
    if check_code != 0:
        print("\nSyntax validation failed. One of the Python source files is invalid.\n")
        return 1

    # Run hardware pre-flight checks after dependencies are confirmed installed
    check_camera()
    check_codec()
    check_screens()

    return run([sys.executable, "main.py"])


if __name__ == "__main__":
    raise SystemExit(main())
