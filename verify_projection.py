"""After the app is running, capture a frame from the webcam to verify
what's being projected onto the guitar."""
import cv2
import numpy as np
import os, sys, time

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# Open camera 0 (same as the app uses)
# Wait a bit to let the app release it if needed
print("Attempting to capture verification frame...")

# Try camera 1 (secondary) if camera 0 is in use by the app
for cam_idx in [1, 0]:
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)
    if cap.isOpened():
        print(f"Opened camera {cam_idx}")
        break
    cap.release()

if not cap.isOpened():
    print("Could not open any camera for verification")
    sys.exit(1)

# Let auto-exposure settle
for _ in range(30):
    cap.read()
    time.sleep(0.033)

# Capture
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture frame")
    sys.exit(1)

cv2.imwrite("verify_projection.png", frame)
print(f"Saved verify_projection.png ({frame.shape[1]}x{frame.shape[0]})")
