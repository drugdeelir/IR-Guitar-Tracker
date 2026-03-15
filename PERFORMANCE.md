
# Performance Recommendations for Your Windows 10 Laptop

This application has been architected with a multi-threaded design to maximize performance by separating the user interface from the heavy video processing. This is the most significant optimization for ensuring a smooth, real-time experience.

However, here are several additional factors you can control to get the best possible performance on your Windows 10 laptop.


### 0. Windows 10 Defaults Included in App

This project now applies Windows-aware runtime defaults automatically when it detects Windows:

* Uses the **DirectShow camera backend** (`cv2.CAP_DSHOW`) for faster and more stable webcam open/close behavior on Windows 10.
* Starts with a **lighter camera workload** (960x540 at 30 FPS) to reduce dropped frames on laptop CPUs while preserving reliable IR tracking.
* Uses a slightly more aggressive detection downscale to reduce per-frame CPU cost.

If your machine is more powerful, you can raise resolution/FPS in code later, but these defaults are tuned for stability-first live performance.

### 1. Video Resolution is Key

The most performance-intensive tasks in the application are directly related to the resolution of the video frames being processed.

*   **Camera Input:** If your IR camera allows, set it to a lower resolution (e.g., 640x480 or 1280x720). Higher resolutions like 1080p or 4K will significantly increase CPU load.
*   **Video Cue Files:** Pre-process your video files to match your projector's output resolution (e.g., 1920x1080). If you use very high-resolution video files (like 4K), the application has to downscale them in real-time, which consumes unnecessary CPU cycles.

### 2. Choose Efficient Video Codecs

*   Use standard, efficiently decoded video codecs like **H.264** (commonly found in `.mp4` files). Avoid exotic or very high-bitrate professional codecs unless necessary, as they can be more demanding to decode.

### 3. System Environment

*   **Close Unnecessary Applications:** Before a live performance, close all other applications (web browsers, email clients, etc.) to free up CPU, RAM, and GPU resources for the projection mapping tool.
*   **Use Task Manager:** Use the Windows Task Manager to check if any other processes are unexpectedly consuming a large amount of CPU or memory.
*   **Power Connection:** Ensure your Windows 10 laptop is plugged into a power source. Running on a UPS (if applicable) in a power-saving mode can throttle performance.

### 4. Lighting and IR Tracking

*   **Consistent Lighting:** The more consistent and controlled the lighting is on your stage, the easier it will be for the computer vision algorithm to isolate the IR trackers.
*   **Optimize IR Threshold:** Spend time finding the optimal IR threshold value in the app's settings. A clean threshold means the app spends less time filtering out noise and finding contours, which reduces CPU load.

By keeping these points in mind, you can ensure the application runs as smoothly and efficiently as possible during your live performances.
