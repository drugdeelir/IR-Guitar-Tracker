
# Real-Time Projection Mapping Tool

This is a comprehensive tool for creating real-time projection mapping effects, designed for live stage performances. It uses a camera to track infrared (IR) markers and warps video content onto a dynamic, moving mask.

## Features

*   **Real-Time IR Tracking:** Tracks up to 4 IR markers simultaneously.
*   **Dynamic Mask Warping:** Warps a video source to a mask defined by the live positions of the IR trackers.
*   **Custom Mask Creation:** An interactive mode to draw a custom polygon mask directly on the video feed.
*   **Cue System:** A list-based system to manage and trigger different video cues.
*   **Projector Keystone Correction:** A four-point warping system to align the final output perfectly to any projection surface.
*   **Depth Estimation:** A system to create a 3D "zoom" effect by scaling the mask based on the distance between trackers.
*   **Multi-threaded Performance:** A modern architecture that separates video processing from the UI to ensure a responsive and fast experience.
*   **Standalone Application:** Includes a setup script to package the tool into a native macOS (`.app`) application.

## Installation (for Development)

These instructions are for running the application directly from the source code.

1.  **Install Python:** Ensure you have Python 3 installed on your Mac. You can download it from [python.org](https://www.python.org/).

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Install Dependencies:** It's recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` will need to be generated, but for now, the dependencies are `opencv-python`, `numpy`, and `PyQt5`)*

4.  **Run the Application:**
    ```bash
    python3 main.py
    ```

## How to Package for macOS (`.app` Bundle)

This project uses `py2app` to create a standalone macOS application. This is the recommended way to use the tool for live performances.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Clean Previous Builds:**
    ```bash
    rm -rf build dist
    ```

3.  **Build the Application:**
    ```bash
    python3 setup.py py2app
    ```

3.  **Run the App:** A `Projection Mapper.app` file will be created in the `dist/` directory. You can drag this to your Applications folder and run it like any other Mac app. No more terminal commands are needed!

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
    *   In the "Trackers" input field, enter the four indices of the detected trackers that correspond to the four corners of your mask (e.g., `0,1,2,3`).
    *   Click **"Link Trackers"**.
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
