# How to Launch on Windows

You can launch the Projection Mapper on Windows in two ways:

## Option 1: Run directly with Python (Recommended for Setup)

1.  **Install Python:** Make sure you have Python 3.10 or newer installed from [python.org](https://www.python.org/).
2.  **Open Terminal:** Open `PowerShell` or `Command Prompt` in the project folder.
3.  **Install Dependencies:** Run the following command:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch:** Run the application:
    ```bash
    python main.py
    ```
    *Alternatively, double-click the `run_windows.bat` file provided in the folder.*

---

## Option 2: Build and Run as a Standalone .exe

If you want a single file to carry on a USB stick or to simplify launching on stage:

1.  **Open Terminal** in the project folder.
2.  **Build the EXE:** Run the build script:
    ```bash
    python build_win.py
    ```
3.  **Find the EXE:** Once finished, look in the `dist` folder.
4.  **Run:** Double-click `ProjectionMapper.exe`.
    *Note: All assets (styles, logos) are bundled inside the exe.*

---

## Troubleshooting

-   **Camera not opening:** Ensure no other app (like Zoom or Teams) is using your webcam/IR camera.
-   **MIDI not showing up:** Install **rtpMIDI** to enable network MIDI over Ethernet (see `HELP_MAINSTAGE.md`).
-   **Audio Reactivity:** Grant the application permission to use your Microphone in Windows Privacy Settings.
