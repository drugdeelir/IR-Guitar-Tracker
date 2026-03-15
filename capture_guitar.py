"""Capture clean frame (projector black) to see guitar position accurately."""
import cv2
import numpy as np
import os, json

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

cap = cv2.VideoCapture(0, cv2.CAP_ANY)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set to same resolution as the app uses
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Black projector window
black = np.zeros((1080, 1920, 3), dtype=np.uint8)
cv2.namedWindow("proj", cv2.WINDOW_NORMAL)
cv2.moveWindow("proj", 2560, 0)
cv2.setWindowProperty("proj", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("proj", black)
cv2.waitKey(100)

# Wait for auto-exposure
print("Waiting for camera to adjust...")
for _ in range(60):
    cap.read()
    cv2.waitKey(33)

# Average several frames
frames = []
for _ in range(8):
    ret, f = cap.read()
    if ret:
        frames.append(f.astype(np.float64))
    cv2.waitKey(33)

cap.release()
cv2.destroyAllWindows()

if not frames:
    print("Failed")
    exit(1)

frame = np.mean(frames, axis=0).astype(np.uint8)
h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

print(f"Frame: {w}x{h}, gray max={gray.max()}")

# Save raw frame and an enhanced version
cv2.imwrite("guitar_raw.png", frame)

# Enhance contrast to see the guitar clearly
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)
cv2.imwrite("guitar_enhanced.png", enhanced)

# Also save with grid overlay for coordinate reference
debug = frame.copy()
for x in range(0, w, 50):
    cv2.line(debug, (x, 0), (x, h), (40, 40, 40), 1)
    cv2.putText(debug, str(x), (x+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
for y in range(0, h, 50):
    cv2.line(debug, (0, y), (w, y), (40, 40, 40), 1)
    cv2.putText(debug, str(y), (2, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
cv2.imwrite("guitar_grid.png", debug)

print("Saved: guitar_raw.png, guitar_enhanced.png, guitar_grid.png")
