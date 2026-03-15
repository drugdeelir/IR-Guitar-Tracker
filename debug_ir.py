"""Capture frame, find IR markers with brightness-weighted centroids, save annotated image."""
import cv2
import numpy as np
import os, json

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

cap = cv2.VideoCapture(0, cv2.CAP_ANY)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
for _ in range(15):
    cap.read()
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
h, w = gray.shape

# Blob detector
params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.minThreshold = 10
params.maxThreshold = 255
params.thresholdStep = 5
params.filterByArea = True
params.minArea = 15
params.maxArea = 200000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minDistBetweenBlobs = 8
params.minRepeatability = 2
det = cv2.SimpleBlobDetector_create(params)

inverted = cv2.bitwise_not(blurred)
kps = det.detect(inverted)

# Get brightness-weighted centroids for bright markers
markers = []
for kp in kps:
    cx, cy = int(kp.pt[0]), int(kp.pt[1])
    r = max(6, int(kp.size / 2) + 2)
    y1, y2 = max(0, cy - r), min(h, cy + r)
    x1, x2 = max(0, cx - r), min(w, cx + r)
    roi = gray[y1:y2, x1:x2].astype(np.float64)
    peak = float(roi.max())

    if peak < 200:
        continue

    # Brightness-weighted centroid within ROI
    # Weight = (pixel - threshold)^2 to strongly favor brightest pixels
    thresh = peak * 0.7
    weights = np.maximum(roi - thresh, 0) ** 2
    total_w = weights.sum()
    if total_w > 0:
        ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        wcx = float((xs * weights).sum() / total_w) + x1
        wcy = float((ys * weights).sum() / total_w) + y1
    else:
        wcx, wcy = float(cx), float(cy)

    markers.append((peak, wcx, wcy, kp.size))
    print(f"Marker: blob=({cx},{cy}) weighted=({wcx:.1f},{wcy:.1f}) peak={peak:.0f} size={kp.size:.1f}")

markers.sort(key=lambda x: -x[0])
print(f"\nFound {len(markers)} IR markers")

# Draw
debug = frame.copy()
for peak, wcx, wcy, size in markers:
    # Red filled circle at weighted centroid
    cv2.circle(debug, (int(wcx), int(wcy)), 8, (0, 0, 255), -1)
    # Green circle at blob boundary
    cv2.circle(debug, (int(wcx), int(wcy)), max(5, int(size/2)), (0, 255, 0), 2)
    cv2.putText(debug, f"({int(wcx)},{int(wcy)}) p={int(peak)}",
                (int(wcx)+12, int(wcy)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)

# Save marker positions for mask creation
marker_positions = [(wcx, wcy) for _, wcx, wcy, _ in markers]
with open("marker_positions.json", "w") as f:
    json.dump(marker_positions, f)

cv2.imwrite("debug_markers.png", debug)
print("Saved: debug_markers.png, marker_positions.json")
