
import cv2
import numpy as np
import time
from worker import Worker, CameraThread, TrackingThread
from PyQt5.QtCore import QCoreApplication
import sys

def test_threading():
    app = QCoreApplication(sys.argv)
    worker = Worker()

    print("Checking threads...")
    if not isinstance(worker.camera_thread, CameraThread):
        print("FAIL: camera_thread is not CameraThread")
        return False
    if not isinstance(worker.tracking_thread, TrackingThread):
        print("FAIL: tracking_thread is not TrackingThread")
        return False

    print("Threads initialized. Starting test...")

    # Simulate some frames being "captured" if camera fails (which it will in sandbox)
    # Actually, let's just check if they are running.
    print(f"Camera thread running: {worker.camera_thread.isRunning()}")
    print(f"Tracking thread running: {worker.tracking_thread.isRunning()}")

    # Check if we can manually trigger a frame update and see if it propagates
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_frame, (320, 240), 50, (255, 255, 255), -1)

    from PyQt5.QtCore import QMutexLocker
    with QMutexLocker(worker.latest_raw_frame_mutex):
        worker.latest_raw_frame = test_frame
        worker.latest_raw_frame_id = 1

    print("Injected test frame. Waiting for tracking...")
    time.sleep(1.0)

    with QMutexLocker(worker.tracking_mutex):
        print(f"Tracking frame count: {worker.tracking_frame_count}")
        # Even if it found 0 points, the count should increase if the thread is working
        if worker.tracking_frame_count > 0:
            print("SUCCESS: Tracking thread is processing frames.")
        else:
            print("FAIL: Tracking thread did not process injected frame.")
            # Maybe it's because it's too fast or it needs more time
            time.sleep(2.0)
            if worker.tracking_frame_count > 0:
                print("SUCCESS: Tracking thread is processing frames (after delay).")
            else:
                print("FAIL: Tracking thread still hasn't processed frame.")

    worker.stop()
    print("Test complete.")
    return True

if __name__ == "__main__":
    test_threading()
