
import cv2
import numpy as np
import time
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker
from PyQt5.QtGui import QImage
from itertools import combinations

class VideoPlayer(QThread):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._running = True
        self.latest_frame = None
        self.mutex = QMutex()
        self.cap = cv2.VideoCapture(video_path)

    def run(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            with QMutexLocker(self.mutex):
                self.latest_frame = frame

            # Control playback speed roughly (30fps)
            time.sleep(1/30.0)

    def get_frame(self):
        with QMutexLocker(self.mutex):
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self._running = False
        self.wait()
        self.cap.release()

class Worker(QObject):
    frame_ready = pyqtSignal(QImage)
    projector_frame_ready = pyqtSignal(QImage)
    still_frame_ready = pyqtSignal(QImage)
    trackers_detected = pyqtSignal(int)
    camera_error = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.video_source = 0
        self.warp_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.masks = []
        self.video_players = {}
        self.ir_threshold = 200
        self._camera_changed = True
        self.baseline_distance = 0
        self.depth_sensitivity = 1.0
        self._calibrate_depth_flag = False
        self._capture_still_frame_flag = False
        self.marker_config = None
        self.bpm = 120.0
        self.last_tracked_points = None
        self.roi_padding = 50

        # Crossfade management
        self.fades = {} # tag: {'prev_path': path, 'start_time': time}
        self.fade_duration = 1.0

    def set_marker_points(self, points):
        self.marker_config = [ (p.x(), p.y()) for p in points]
        if len(self.marker_config) > 1:
            distances = []
            for p1, p2 in combinations(self.marker_config, 2):
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                distances.append(dist)
            self.marker_fingerprint = sorted(distances)
        else:
            self.marker_fingerprint = []
        self.last_tracked_points = None

    def clear_marker_config(self):
        self.marker_config = None
        self.marker_fingerprint = []
        self.last_tracked_points = None

    def capture_still_frame(self):
        self._capture_still_frame_flag = True

    def calibrate_depth(self):
        self._calibrate_depth_flag = True

    def set_depth_sensitivity(self, value):
        self.depth_sensitivity = value

    def set_ir_threshold(self, value):
        self.ir_threshold = value

    def set_video_source(self, source):
        self.video_source = source
        self._camera_changed = True

    def set_bpm(self, bpm):
        self.bpm = bpm

    def switch_video(self, tag, video_path):
        for mask in self.masks:
            if mask.tag == tag:
                if mask.video_path and mask.video_path != video_path:
                    self.fades[tag] = {
                        'prev_path': mask.video_path,
                        'start_time': time.time()
                    }
                mask.video_path = video_path

    def toggle_mask(self, tag, visible):
        for mask in self.masks:
            if mask.tag == tag:
                mask.visible = visible

    def set_fx(self, tag, fx_name, enabled):
        for mask in self.masks:
            if mask.tag == tag:
                if enabled and fx_name not in mask.active_fx:
                    mask.active_fx.append(fx_name)
                elif not enabled and fx_name in mask.active_fx:
                    mask.active_fx.remove(fx_name)

    def apply_fx(self, frame, mask):
        if not mask.active_fx:
            return frame
        processed = frame.copy()
        if 'strobe' in mask.active_fx:
            period = 60.0 / self.bpm
            if (time.time() % period) < (period / 2.0):
                processed = np.zeros_like(processed)
        if 'blur' in mask.active_fx:
            processed = cv2.GaussianBlur(processed, (15, 15), 0)
        if 'invert' in mask.active_fx:
            processed = cv2.bitwise_not(processed)
        if 'edges' in mask.active_fx:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if 'tint' in mask.active_fx:
            tint = np.full_like(processed, mask.tint_color)
            processed = cv2.addWeighted(processed, 0.7, tint, 0.3, 0)
        return processed

    def get_tracked_points(self, frame):
        h, w = frame.shape[:2]
        roi_x, roi_y, roi_w, roi_h = 0, 0, w, h
        if self.last_tracked_points is not None:
            pts = np.array(self.last_tracked_points)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            roi_x = max(0, int(min_x - self.roi_padding))
            roi_y = max(0, int(min_y - self.roi_padding))
            roi_w = min(w - roi_x, int(max_x - min_x + 2 * self.roi_padding))
            roi_h = min(h - roi_y, int(max_y - min_y + 2 * self.roi_padding))

        roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.ir_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_points = []
        for contour in contours:
            if cv2.contourArea(contour) > 20:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"]) + roi_x
                    cY = int(M["m01"] / M["m00"]) + roi_y
                    detected_points.append((cX, cY))

        if self.marker_config and len(self.marker_config) > 1 and len(detected_points) >= len(self.marker_config):
            points_to_check = detected_points[:20]
            num_markers = len(self.marker_config)
            for point_combo in combinations(points_to_check, num_markers):
                current_distances = []
                for p1, p2 in combinations(point_combo, 2):
                    current_distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
                current_fingerprint = sorted(current_distances)

                if len(current_fingerprint) == len(self.marker_fingerprint):
                    is_match = True
                    for i in range(len(current_fingerprint)):
                        if not np.isclose(current_fingerprint[i], self.marker_fingerprint[i], rtol=0.15):
                            is_match = False
                            break
                    if is_match:
                        src_pts = np.float32(self.marker_config).reshape(-1, 1, 2)
                        dst_pts = np.float32(point_combo).reshape(-1, 1, 2)
                        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if matrix is not None:
                            transformed_src = cv2.perspectiveTransform(src_pts, matrix)
                            ordered_points = []
                            remaining_dst = list(point_combo)
                            for i in range(num_markers):
                                pred = transformed_src[i][0]
                                closest = min(remaining_dst, key=lambda p: np.linalg.norm(np.array(p) - pred))
                                ordered_points.append(closest)
                                remaining_dst.remove(closest)
                            self.last_tracked_points = ordered_points
                            return ordered_points

        if self.last_tracked_points is not None:
            self.last_tracked_points = None
            return self.get_tracked_points(frame)
        return detected_points

    def process_video(self):
        main_cap = None
        while self._running:
            if self._camera_changed:
                if main_cap: main_cap.release()
                main_cap = cv2.VideoCapture(self.video_source)
                if not main_cap.isOpened():
                    self.camera_error.emit(self.video_source)
                    main_cap = None
                self._camera_changed = False

            if main_cap is None:
                QThread.msleep(100); continue

            ret, main_frame = main_cap.read()
            if not ret:
                self.camera_error.emit(self.video_source)
                QThread.msleep(500); continue

            h, w = main_frame.shape[:2]
            projector_output = np.zeros((h, w, 3), dtype=np.uint8)

            tracked_points = self.get_tracked_points(main_frame)
            self.trackers_detected.emit(len(tracked_points))

            if self._calibrate_depth_flag and len(tracked_points) >= 2:
                self.baseline_distance = np.linalg.norm(np.array(tracked_points[0]) - np.array(tracked_points[1]))
                self._calibrate_depth_flag = False

            for point in tracked_points:
                cv2.circle(main_frame, point, 5, (0, 0, 255), -1)

            sorted_masks = sorted(self.masks, key=lambda m: 0 if m.tag == 'background' else 1)
            for mask in sorted_masks:
                if not mask.visible or not mask.video_path: continue

                # Ensure players are running
                if mask.video_path not in self.video_players:
                    player = VideoPlayer(mask.video_path)
                    player.start()
                    self.video_players[mask.video_path] = player

                player = self.video_players[mask.video_path]
                frame_cue = player.get_frame()

                # Handle Crossfade
                if mask.tag in self.fades:
                    fade_info = self.fades[mask.tag]
                    elapsed = time.time() - fade_info['start_time']
                    if elapsed < self.fade_duration:
                        if fade_info['prev_path'] in self.video_players:
                            prev_player = self.video_players[fade_info['prev_path']]
                            prev_frame = prev_player.get_frame()
                            if prev_frame is not None and frame_cue is not None:
                                alpha = elapsed / self.fade_duration
                                frame_cue = cv2.addWeighted(prev_frame, 1.0 - alpha, frame_cue, alpha, 0)
                    else:
                        del self.fades[mask.tag]

                if frame_cue is not None:
                    frame_cue = self.apply_fx(frame_cue, mask)
                    if mask.type == 'dynamic' and mask.linked_marker_count == len(tracked_points):
                        src_pts = np.float32(mask.source_points)
                        dst_pts_raw = np.float32(tracked_points)
                        if self.baseline_distance > 0 and len(tracked_points) >= 2:
                            current_distance = np.linalg.norm(np.array(tracked_points[0]) - np.array(tracked_points[1]))
                            scale_factor = (current_distance / self.baseline_distance - 1.0) * self.depth_sensitivity + 1.0
                            center = np.mean(dst_pts_raw, axis=0)
                            dst_pts = (dst_pts_raw - center) * scale_factor + center
                        else:
                            dst_pts = dst_pts_raw
                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        warped_cue = cv2.warpPerspective(frame_cue, matrix, (w, h))
                        mask_img = np.zeros_like(projector_output)
                        cv2.fillConvexPoly(mask_img, np.int32(dst_pts), (255, 255, 255))
                        projector_output = cv2.bitwise_and(projector_output, cv2.bitwise_not(mask_img))
                        projector_output = cv2.add(projector_output, cv2.bitwise_and(warped_cue, mask_img))
                    elif mask.type == 'static':
                        if not mask.source_points:
                            projector_output = cv2.resize(frame_cue, (w, h))
                        else:
                            src_pts = np.float32([[0, 0], [frame_cue.shape[1], 0], [frame_cue.shape[1], frame_cue.shape[0]], [0, frame_cue.shape[0]]])
                            dst_pts = np.float32(mask.source_points)
                            matrix, _ = cv2.findHomography(src_pts, dst_pts)
                            warped_cue = cv2.warpPerspective(frame_cue, matrix, (w, h))
                            mask_img = np.zeros_like(projector_output)
                            cv2.fillConvexPoly(mask_img, np.int32(dst_pts), (255, 255, 255))
                            projector_output = cv2.bitwise_and(projector_output, cv2.bitwise_not(mask_img))
                            projector_output = cv2.add(projector_output, cv2.bitwise_and(warped_cue, mask_img))

            if self._capture_still_frame_flag:
                rgb = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                self.still_frame_ready.emit(QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888))
                self._capture_still_frame_flag = False

            rgb_main = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(QImage(rgb_main.data, w, h, w * 3, QImage.Format_RGB888))

            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_pts = np.float32([[p[0] * w, p[1] * h] for p in self.warp_points])
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_output = cv2.warpPerspective(projector_output, matrix, (w, h))
            rgb_proj = cv2.cvtColor(warped_output, cv2.COLOR_BGR2RGB)
            self.projector_frame_ready.emit(QImage(rgb_proj.data, w, h, w * 3, QImage.Format_RGB888))
            QThread.msleep(30)

        main_cap.release()
        for player in self.video_players.values(): player.stop()

    def stop(self): self._running = False
    def set_warp_points(self, points): self.warp_points = points
    def set_masks(self, masks):
        self.masks = masks
        # Modified to NOT stop players that are still needed for fades
        current_cues = {mask.video_path for mask in self.masks if mask.video_path}
        fade_cues = {f['prev_path'] for f in self.fades.values()}

        needed_cues = current_cues.union(fade_cues)
        for path in list(self.video_players.keys()):
            if path not in needed_cues:
                self.video_players[path].stop()
                del self.video_players[path]
