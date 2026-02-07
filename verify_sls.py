
import cv2
import numpy as np
import time
from worker import Worker
from PyQt5.QtCore import QCoreApplication, QThread, QMutexLocker
import sys

def test_sls_sequence():
    app = QCoreApplication(sys.argv)
    worker = Worker()
    worker.projector_width = 1280
    worker.projector_height = 720

    # Start rendering thread
    render_thread = QThread()
    worker.moveToThread(render_thread)
    render_thread.started.connect(worker.process_video)
    render_thread.start()

    print("Starting Room Scan simulation...")
    worker.run_room_scan()

    start_time = time.time()
    patterns_generated = False
    while time.time() - start_time < 5.0:
        if worker._sls_step == 0 and len(worker._sls_patterns_x) > 0:
            print(f"SUCCESS: Patterns generated in worker thread. Total X: {len(worker._sls_patterns_x)}")
            patterns_generated = True
            break
        time.sleep(0.1)

    if not patterns_generated:
        print(f"FAIL: Patterns not generated. SLS Step: {worker._sls_step}")

    # Now simulate resolution stability
    print(f"Current camera res: {worker._current_camera_res}, Requested: {worker.requested_camera_res}")
    # Manually set current res to match requested to simulate stability
    worker._current_camera_res = worker.requested_camera_res

    # Inject a frame to trigger capture
    locker = QMutexLocker(worker.latest_main_frame_mutex)
    worker.latest_main_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    worker.latest_main_frame_id += 1
    del locker # Release lock

    print("Simulated resolution stability and frame update. Waiting for capture...")
    # Wait for multiple frames since _sls_wait_frames is 45
    for _ in range(60):
        time.sleep(0.1)
        if worker._sls_step > 0:
            break

    if worker._sls_step > 0:
        print(f"SUCCESS: Scan advanced to step {worker._sls_step}")
    else:
        print(f"FAIL: Scan did not advance. SLS Step: {worker._sls_step}, Wait count: {worker._sls_curr_wait}")

    worker.stop()
    render_thread.quit()
    render_thread.wait()
    print("Test complete.")

if __name__ == "__main__":
    test_sls_sequence()
