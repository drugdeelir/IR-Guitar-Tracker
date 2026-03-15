"""Create test pattern videos for mask projection testing."""
import cv2
import numpy as np
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

def create_color_cycle_video(path, width=640, height=480, fps=30, duration=10):
    """Red/green/blue cycling gradient video for guitar mask."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    total_frames = fps * duration
    for i in range(total_frames):
        t = i / total_frames
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Cycling color gradient
        phase = t * 6  # 6 full cycles
        r = int(127 + 127 * np.sin(phase * 2 * np.pi))
        g = int(127 + 127 * np.sin(phase * 2 * np.pi + 2.094))
        b = int(127 + 127 * np.sin(phase * 2 * np.pi + 4.189))
        # Radial gradient from center
        cy, cx = height // 2, width // 2
        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        factor = (1.0 - dist / max_dist).clip(0, 1)
        frame[:, :, 0] = (b * factor).astype(np.uint8)
        frame[:, :, 1] = (g * factor).astype(np.uint8)
        frame[:, :, 2] = (r * factor).astype(np.uint8)
        # Add "GUITAR" text
        cv2.putText(frame, "GUITAR", (width//2 - 80, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        out.write(frame)
    out.release()
    print(f"Created: {path} ({total_frames} frames)")

def create_background_video(path, width=640, height=480, fps=30, duration=10):
    """Slow-moving starfield/ambient background."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    total_frames = fps * duration
    # Generate random stars
    np.random.seed(42)
    num_stars = 200
    star_x = np.random.randint(0, width, num_stars)
    star_y = np.random.randint(0, height, num_stars)
    star_brightness = np.random.randint(100, 255, num_stars)
    star_speed = np.random.uniform(0.5, 2.0, num_stars)

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Dark blue background with gradient
        frame[:, :, 0] = 40  # blue channel
        frame[:, :, 1] = 10
        frame[:, :, 2] = 5
        # Moving stars
        for s in range(num_stars):
            sx = int((star_x[s] + i * star_speed[s]) % width)
            sy = star_y[s]
            b = int(star_brightness[s])
            cv2.circle(frame, (sx, sy), 1, (b, b, b), -1)
        # Add "BG" watermark
        cv2.putText(frame, "BACKGROUND", (width//2 - 120, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 80), 2)
        out.write(frame)
    out.release()
    print(f"Created: {path} ({total_frames} frames)")

guitar_path = os.path.join(output_dir, "test_guitar.mp4")
bg_path = os.path.join(output_dir, "test_background.mp4")

create_color_cycle_video(guitar_path)
create_background_video(bg_path)
print("\nDone! Videos ready for projection testing.")
