"""Create test videos: bright magenta for guitar, dark blue for background."""
import cv2
import numpy as np

# Guitar test video: bright animated magenta/cyan pattern - impossible to miss
w, h = 640, 480
fps = 30
frames = fps * 5  # 5 second loop

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Guitar: animated bright colored checkerboard
out = cv2.VideoWriter("test_guitar.mp4", fourcc, fps, (w, h))
for i in range(frames):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    t = i / fps
    # Animated diagonal stripes - bright magenta and cyan
    for y in range(h):
        for x in range(0, w, 1):
            val = int((x + y + t * 100) / 30) % 2
            if val == 0:
                frame[y, x] = (255, 0, 255)  # magenta
            else:
                frame[y, x] = (0, 255, 255)  # cyan
    # Add text
    cv2.putText(frame, "GUITAR MASK", (w//2 - 120, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, f"Frame {i}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out.write(frame)
out.release()
print("Created test_guitar.mp4")

# Background: subtle dark blue
out = cv2.VideoWriter("test_background.mp4", fourcc, fps, (w, h))
for i in range(frames):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    t = i / fps
    # Slow animated dark blue gradient
    for y in range(h):
        blue = int(30 + 20 * np.sin(2 * np.pi * (y / h + t * 0.2)))
        frame[y, :] = (blue, 0, 0)  # dark blue (BGR)
    cv2.putText(frame, "BG", (w//2 - 20, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 30, 0), 2)
    out.write(frame)
out.release()
print("Created test_background.mp4")
