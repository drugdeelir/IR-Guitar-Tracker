
import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtGui import QImage

class Worker(QObject):
    frame_ready = pyqtSignal(QImage)
    projector_frame_ready = pyqtSignal(QImage)
    trackers_detected = pyqtSignal(int)
    camera_error = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.video_source = 0
        self.warp_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.masks = []
        self.video_captures = {}
        self.ir_threshold = 200
        self._camera_changed = True # Start with true to initialize camera
        self.baseline_distance = 0
        self.depth_sensitivity = 1.0
        self._calibrate_depth_flag = False

    def calibrate_depth(self):
        self._calibrate_depth_flag = True

    def set_depth_sensitivity(self, value):
        self.depth_sensitivity = value

    def set_ir_threshold(self, value):
        self.ir_threshold = value

    def set_video_source(self, source):
        self.video_source = source
        self._camera_changed = True

    def process_video(self):
        main_cap = None
        
        while self._running:
            if self._camera_changed:
                if main_cap:
                    main_cap.release()
                main_cap = cv2.VideoCapture(self.video_source)
                if not main_cap.isOpened():
                    self.camera_error.emit(self.video_source)
                    main_cap = None
                self._camera_changed = False

            if main_cap is None:
                QThread.msleep(100)
                continue

            ret, main_frame = main_cap.read()
            ret, main_frame = main_cap.read()
            if not ret:
                self.camera_error.emit(self.video_source)
                QThread.msleep(500) # Wait before trying again
                continue

            h, w, _ = main_frame.shape
            projector_output = np.zeros((h, w, 3), dtype=np.uint8)


            # IR Tracking
            gray_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_frame, self.ir_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            tracked_points = []
            for contour in contours:
                # Filter contours by area to remove noise
                if cv2.contourArea(contour) > 20:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        tracked_points.append((cX, cY))
            
            self.trackers_detected.emit(len(tracked_points))

            # Depth calibration
            if self._calibrate_depth_flag and len(tracked_points) >= 2:
                self.baseline_distance = np.linalg.norm(np.array(tracked_points[0]) - np.array(tracked_points[1]))
                self._calibrate_depth_flag = False
                print(f"Depth calibrated with baseline distance: {self.baseline_distance}")

            # Draw trackers on main display feed
            for point in tracked_points:
                cv2.circle(main_frame, point, 5, (0, 0, 255), -1)

            # Process cues and warp masks
            for mask in self.masks:
                if mask.type == 'dynamic' and len(tracked_points) >= 4:
                    if mask.video_path not in self.video_captures:
                        self.video_captures[mask.video_path] = cv2.VideoCapture(mask.video_path)
                    
                    cap = self.video_captures[mask.video_path]
                    ret_cue, frame_cue = cap.read()
                    
                    if not ret_cue:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_cue, frame_cue = cap.read()

                    if ret_cue and mask.source_points:
                        cue_h, cue_w, _ = frame_cue.shape
                        
                        # The source points for the transform are the corners of the custom mask
                        src_pts = np.float32(mask.source_points)

                        # The destination points are the live tracker positions
                        dst_pts_raw = np.float32([tracked_points[i] for i in mask.tracker_ids[:4]])
                        
                        # Apply depth scaling
                        if self.baseline_distance > 0 and len(tracked_points) >= 2:
                            current_distance = np.linalg.norm(np.array(tracked_points[0]) - np.array(tracked_points[1]))
                            scale_factor = (current_distance / self.baseline_distance - 1.0) * self.depth_sensitivity + 1.0
                            
                            center = np.mean(dst_pts_raw, axis=0)
                            dst_pts = (dst_pts_raw - center) * scale_factor + center
                        else:
                            dst_pts = dst_pts_raw

                        # Create a transform from the mask's original shape to the live tracker quad
                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        warped_cue = cv2.warpPerspective(frame_cue, matrix, (w, h))

                        # Create a mask to apply the warped video only within the warped polygon
                        mask_image = np.zeros_like(projector_output)
                        cv2.fillConvexPoly(mask_image, np.int32(dst_pts), (255, 255, 255))
                        
                        # Composite the result
                        projector_output = cv2.bitwise_and(projector_output, cv2.bitwise_not(mask_image))
                        projector_output = cv2.add(projector_output, cv2.bitwise_and(warped_cue, mask_image))


            # Main display processing (shows the raw camera feed with trackers)
            rgb_image_main = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
            qt_image_main = QImage(rgb_image_main.data, w, h, w * 3, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image_main)

            # Projector output processing
            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_points = np.float32([[p[0] * w, p[1] * h] for p in self.warp_points])
            
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_output = cv2.warpPerspective(projector_output, matrix, (w, h))

            rgb_image_proj = cv2.cvtColor(warped_output, cv2.COLOR_BGR2RGB)
            qt_image_proj = QImage(rgb_image_proj.data, w, h, w * 3, QImage.Format_RGB888)
            self.projector_frame_ready.emit(qt_image_proj)

            QThread.msleep(30)

        main_cap.release()
        for cap in self.video_captures.values():
            cap.release()

    def stop(self):
        self._running = False

    def set_video_source(self, source):
        self.video_source = source
    
    def set_warp_points(self, points):
        self.warp_points = points
    
    def set_masks(self, masks):
        self.masks = masks
        # Clean up old video captures
        current_cues = {mask.video_path for mask in self.masks}
        for cue in list(self.video_captures.keys()):
            if cue not in current_cues:
                self.video_captures[cue].release()
                del self.video_captures[cue]
