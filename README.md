
# Real-Time Projection Mapping Tool

This is a comprehensive tool for creating real-time projection mapping effects, designed for live stage performances. It uses a camera to track infrared (IR) markers and warps video content onto a dynamic, moving mask.

## Features

*   **Real-Time IR Tracking:** Tracks configurable IR marker constellations with smoothing and dropout recovery.
*   **Dynamic Mask Warping:** Warps a video source to a mask defined by the live positions of the IR trackers.
*   **Custom Mask Creation:** An interactive mode to draw a custom polygon mask directly on the video feed.
*   **Cue System:** A list-based system to manage and trigger different video cues.
*   **Projector Keystone Correction:** A four-point warping system to align the final output perfectly to any projection surface.
*   **Depth Estimation:** A system to create a 3D "zoom" effect by scaling the mask based on the distance between trackers.
*   **Adaptive Thresholding:** Switch between manual threshold and auto (Otsu) threshold for varying stage lighting.
*   **Live Diagnostics:** Real-time FPS and frame-time stats in the control panel.
*   **Stage Timing Breakdown:** Live detect/match/warp/render timings to quickly spot bottlenecks.
*   **Session Persistence:** Remembers thresholds, display selections, and warp calibration between launches.
*   **Multi-threaded Performance:** A modern architecture that separates video processing from the UI to ensure a responsive and fast experience.
*   **Windows-first Workflow:** Optimized for Windows 10 laptop use with practical defaults for live performance.

## Installation (for Development)

These instructions are for running the application directly from the source code.

1.  **Install Python:** Ensure you have Python 3 installed on your Windows 10 laptop. You can download it from [python.org](https://www.python.org/).

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Install Dependencies:** It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *(Dependencies are listed in `requirements.txt`.)*

4.  **Run the Application:**
    ```bash
    python main.py
    ```

## Package as a Windows Executable (Optional)

If you want to run this without opening a terminal each time, you can build a `.exe` with PyInstaller:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build a one-folder executable:
   ```bash
   pyinstaller --noconfirm --name ProjectionMapper --windowed --add-data "style.qss;." --add-data "logo.png;." main.py
   ```
   or:
   ```bash
   python setup.py
   ```

3. Run from `dist/ProjectionMapper/ProjectionMapper.exe`.

## How to Use

1.  **Startup:** On launch, you'll see a brief splash screen with the logo video, and then the main control window and a black projector output window will appear.
2.  **Select Devices:**
    *   Use the **Camera** dropdown to select your IR camera.
    *   Use the **Projector Display** dropdown to select the screen or projector you want to use for the output. The projector window will automatically move to that screen and go fullscreen.
3.  **Add a Video Cue:**
    *   Click **"Add Video Cue"** and select a video file. It will appear in the "Cues" list.
4.  **Calibrate IR Tracking:**
    *   Adjust the **IR Threshold** slider until the application reliably detects your IR markers. The "Trackers detected" label will show you how many points it sees. You should also see red circles drawn over the trackers in the main video display.
5.  **Create a Mask:**
    *   Select the video cue you want to associate the mask with from the list.
    *   Click **"Create Mask"**.
    *   Click on the main video feed to draw the vertices of your custom mask shape.
    *   When you are done, click **"Finish Mask"**. The mask points are now associated with the selected video cue.
6.  **Link Trackers:**
    *   Click **"Select Guitar Markers"**, capture a still frame, and click the marker positions directly on the image in order.
    *   Click **"Confirm Markers"**, then click **"Link Mask to Markers"** for the selected cue.
7.  **Calibrate Projector:**
    *   Click **"Enable Warping"**. You will see four red dots on the projector output window.
    *   Drag these dots to the corners of your real-world projection surface to correct for keystone distortion.
    *   Click **"Disable Warping"** when you are done.
8.  **Calibrate Depth:**
    *   Position your guitar at a neutral, middle-distance from the camera.
    *   Click **"Calibrate Depth"**. This sets the baseline distance for the 3D effect.
    *   Now, as you move the guitar closer or further away, the mask will scale in size. Use the **Sensitivity** slider to adjust the strength of the effect.

## Performance Note for SSD Users

Running this application from an SSD is highly recommended. It will significantly improve the loading speed of your video cue files, resulting in smoother transitions and a more reliable performance during a live show.


## Windows 10 Optimization Notes

This app now auto-applies Windows-focused performance defaults:

* DirectShow camera backend for more stable webcam startup/latency on Windows.
* Camera defaults tuned for laptop performance (960x540 @ 30 FPS).
* Slightly lower detection scale to reduce CPU usage while tracking IR markers.

These defaults are stability-first for live use.

The app also attempts multiple camera backends on Windows (DirectShow -> Media Foundation -> Any) so it works with a wider range of webcams and capture devices.
