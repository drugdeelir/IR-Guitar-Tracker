import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES_TO_VALIDATE = ["main.py", "worker.py", "splash.py"]


def contains_diff_markers(path: Path) -> bool:
    try:
        first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:4]
    except OSError:
        return False
    markers = ("diff --git", "index ", "--- a/", "+++ b/")
    return any(line.startswith(markers) for line in first)


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
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

    return run([sys.executable, "main.py"])


if __name__ == "__main__":
    raise SystemExit(main())
