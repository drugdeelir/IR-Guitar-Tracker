"""Capture multiple frames, average marker positions, save for mask creation."""
import cv2
import numpy as np
import os, json

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

cap = cv2.VideoCapture(0, cv2.CAP_ANY)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

# Warm up
for _ in range(15):
    cap.read()

all_markers = []  # list of lists of (x,y) per frame
last_frame = None

for frame_idx in range(30):
    ret, frame = cap.read()
    if not ret:
        continue
    last_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    inverted = cv2.bitwise_not(blurred)
    kps = det.detect(inverted)

    frame_markers = []
    for kp in kps:
        cx, cy = kp.pt
        r = max(6, int(kp.size / 2) + 3)
        y1, y2 = max(0, int(cy) - r), min(h, int(cy) + r)
        x1, x2 = max(0, int(cx) - r), min(w, int(cx) + r)
        roi = gray[y1:y2, x1:x2].astype(np.float64)
        peak = float(roi.max())
        if peak < 200:
            continue
        # Weighted centroid
        thresh = peak * 0.65
        weights = np.maximum(roi - thresh, 0) ** 2
        total_w = weights.sum()
        if total_w > 0:
            ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
            wcx = float((xs * weights).sum() / total_w) + x1
            wcy = float((ys * weights).sum() / total_w) + y1
        else:
            wcx, wcy = float(cx), float(cy)
        frame_markers.append((wcx, wcy, peak))

    all_markers.append(frame_markers)

cap.release()

# Find the most common count of markers
counts = [len(m) for m in all_markers]
from collections import Counter
count_freq = Counter(counts)
print(f"Marker count distribution: {dict(count_freq)}")
target_count = count_freq.most_common(1)[0][0]
print(f"Most common count: {target_count}")

# Average positions from frames with the target count
positions_sum = None
n_frames = 0
for markers in all_markers:
    if len(markers) != target_count:
        continue
    # Sort by y then x to maintain consistent ordering
    sorted_m = sorted(markers, key=lambda m: (m[1], m[0]))
    pts = np.array([(m[0], m[1]) for m in sorted_m])
    if positions_sum is None:
        positions_sum = pts
    else:
        positions_sum += pts
    n_frames += 1

if n_frames > 0:
    avg_positions = positions_sum / n_frames
    print(f"\nAveraged over {n_frames} frames:")
    for i, (x, y) in enumerate(avg_positions):
        print(f"  Marker {i}: ({x:.1f}, {y:.1f})")

    # Save
    marker_list = avg_positions.tolist()
    with open("marker_positions.json", "w") as f:
        json.dump(marker_list, f)

    # Draw on last frame
    if last_frame is not None:
        debug = last_frame.copy()
        for x, y in avg_positions:
            cv2.circle(debug, (int(x), int(y)), 10, (0, 0, 255), -1)
            cv2.circle(debug, (int(x), int(y)), 15, (0, 255, 0), 2)

        # Draw guitar bounding box with margin
        xs = avg_positions[:, 0]
        ys = avg_positions[:, 1]
        margin_x = 30
        margin_y = 30
        x_min = max(0, int(xs.min() - margin_x))
        x_max = min(w, int(xs.max() + margin_x))
        y_min = max(0, int(ys.min() - margin_y))
        y_max = min(h, int(ys.max() + margin_y))
        cv2.rectangle(debug, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
        cv2.putText(debug, "Guitar mask area", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        cv2.imwrite("debug_avg_markers.png", debug)
        print(f"\nGuitar bounding box: ({x_min},{y_min}) to ({x_max},{y_max})")
        print(f"Frame size: {w}x{h}")
        print("Saved: debug_avg_markers.png")
else:
    print("No consistent frames found!")
