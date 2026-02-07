import cv2
import numpy as np
from itertools import combinations
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtGui import QImage


class Worker(QObject):
    frame_ready = pyqtSignal(QImage)
    projector_frame_ready = pyqtSignal(QImage)
    still_frame_ready = pyqtSignal(QImage)
    trackers_detected = pyqtSignal(int)
    camera_error = pyqtSignal(int)
    room_scan_ready = pyqtSignal(list)
    status_update = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.video_source = 0
        self.warp_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.masks = []
        self.video_captures = {}
        self.ir_threshold = 200
        self._camera_changed = True
        self.baseline_distance = 0
        self.depth_sensitivity = 1.0
        self._calibrate_depth_flag = False
        self._capture_still_frame_flag = False
        self._scan_room_flag = False
        self.marker_config = None
        self.marker_fingerprint = []
        self.last_tracked_points = []
        self.active_cue_index = -1

    def set_marker_points(self, points):
        self.marker_config = [(p.x(), p.y()) for p in points]
        if len(self.marker_config) > 1:
            distances = []
            for p1, p2 in combinations(self.marker_config, 2):
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                distances.append(dist)
            self.marker_fingerprint = sorted(distances)
        else:
            self.marker_fingerprint = []
        self.last_tracked_points = []

    def clear_marker_config(self):
        self.marker_config = None
        self.marker_fingerprint = []
        self.last_tracked_points = []

    def capture_still_frame(self):
        self._capture_still_frame_flag = True

    def calibrate_depth(self):
        self._calibrate_depth_flag = True

    def scan_room(self):
        self._scan_room_flag = True

    def set_depth_sensitivity(self, value):
        self.depth_sensitivity = value

    def set_ir_threshold(self, value):
        self.ir_threshold = value

    def set_video_source(self, source):
        self.video_source = source
        self._camera_changed = True

    def set_warp_points(self, points):
        self.warp_points = points

    def set_active_cue(self, index):
        self.active_cue_index = index

    def _setup_camera(self, cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _scan_room_from_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = gray.shape
        min_area = (w * h) * 0.08
        best_quad = None
        best_area = 0

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > min_area and area > best_area:
                    best_quad = approx.reshape(4, 2)
                    best_area = area

        if best_quad is None:
            return None

        pts = np.array(best_quad, dtype=np.float32)
        center = np.mean(pts, axis=0)
        sorted_pts = sorted(pts.tolist(), key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        ordered = np.array(sorted_pts, dtype=np.float32)
        top = ordered[np.argsort(ordered[:, 1])[:2]]
        bottom = ordered[np.argsort(ordered[:, 1])[2:]]
        top_left, top_right = top[np.argsort(top[:, 0])]
        bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]

        normalized = [
            [float(top_left[0] / w), float(top_left[1] / h)],
            [float(top_right[0] / w), float(top_right[1] / h)],
            [float(bottom_right[0] / w), float(bottom_right[1] / h)],
            [float(bottom_left[0] / w), float(bottom_left[1] / h)],
        ]
        return normalized

    def _ordered_tracking(self, detected_points):
        if not self.marker_config:
            return detected_points

        num_markers = len(self.marker_config)
        if num_markers == 0 or len(detected_points) < num_markers:
            return []

        points_to_check = detected_points[:16]

        best_match = None
        if self.marker_fingerprint and num_markers > 1:
            for point_combo in combinations(points_to_check, num_markers):
                current_distances = []
                for p1, p2 in combinations(point_combo, 2):
                    current_distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
                current_fingerprint = sorted(current_distances)
                if len(current_fingerprint) != len(self.marker_fingerprint):
                    continue
                if all(np.isclose(a, b, rtol=0.18) for a, b in zip(current_fingerprint, self.marker_fingerprint)):
                    src_pts = np.float32(self.marker_config).reshape(-1, 1, 2)
                    dst_pts = np.float32(point_combo).reshape(-1, 1, 2)
                    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if matrix is None:
                        continue
                    transformed_src = cv2.perspectiveTransform(src_pts, matrix)
                    ordered = [None] * num_markers
                    remaining = list(point_combo)
                    for i in range(num_markers):
                        predicted = transformed_src[i][0]
                        closest = min(remaining, key=lambda p: np.linalg.norm(np.array(p) - predicted))
                        ordered[i] = closest
                        remaining.remove(closest)
                    best_match = ordered
                    break

        if best_match is None:
            if self.last_tracked_points and len(self.last_tracked_points) == num_markers:
                remaining = detected_points.copy()
                ordered = []
                for prev in self.last_tracked_points:
                    closest = min(remaining, key=lambda p: np.linalg.norm(np.array(p) - np.array(prev)))
                    ordered.append(closest)
                    remaining.remove(closest)
                best_match = ordered
            else:
                best_match = detected_points[:num_markers]

        if self.last_tracked_points and len(self.last_tracked_points) == len(best_match):
            smoothed = []
            alpha = 0.65
            for i, pt in enumerate(best_match):
                prev = np.array(self.last_tracked_points[i], dtype=np.float32)
                cur = np.array(pt, dtype=np.float32)
                mix = alpha * cur + (1 - alpha) * prev
                smoothed.append((int(mix[0]), int(mix[1])))
            best_match = smoothed

        self.last_tracked_points = best_match.copy()
        return best_match

    def _generate_frame(self, generator_type, shape, tick):
        h, w = shape
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        t = tick / 30.0
        if generator_type == 'plasma':
            x = np.linspace(0, np.pi * 4, w, dtype=np.float32)
            y = np.linspace(0, np.pi * 4, h, dtype=np.float32)
            xx, yy = np.meshgrid(x, y)
            v = (np.sin(xx + t) + np.sin(yy + t * 1.2) + np.sin((xx + yy) * 0.7 + t * 1.7)) / 3.0
            norm = ((v + 1.0) * 127.5).astype(np.uint8)
            frame[..., 0] = norm
            frame[..., 1] = cv2.applyColorMap(norm, cv2.COLORMAP_OCEAN)[..., 1]
            frame[..., 2] = cv2.applyColorMap(norm, cv2.COLORMAP_HSV)[..., 2]
        elif generator_type == 'stripes':
            for i in range(0, w, 24):
                color = (int((i + tick * 5) % 255), int((255 - i + tick * 3) % 255), int((i * 2) % 255))
                cv2.rectangle(frame, (i, 0), (i + 12, h), color, -1)
        else:
            frame[:, :] = (20, 20, 20)
        return frame

    def process_video(self):
        main_cap = None
        tick = 0

        while self._running:
            if self._camera_changed:
                if main_cap:
                    main_cap.release()
                main_cap = cv2.VideoCapture(self.video_source)
                if not main_cap.isOpened():
                    self.camera_error.emit(self.video_source)
                    main_cap = None
                else:
                    self._setup_camera(main_cap)
                    self.status_update.emit(f"Camera {self.video_source} opened")
                self._camera_changed = False

            if main_cap is None:
                QThread.msleep(100)
                continue

            ret, main_frame = main_cap.read()
            if not ret:
                self.camera_error.emit(self.video_source)
                self._camera_changed = True
                QThread.msleep(200)
                continue

            h, w, _ = main_frame.shape
            projector_output = np.zeros((h, w, 3), dtype=np.uint8)

            gray_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
            denoised = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            _, thresh = cv2.threshold(denoised, self.ir_threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_points = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 8 < area < 1200:
                    m = cv2.moments(contour)
                    if m['m00'] != 0:
                        cx = int(m['m10'] / m['m00'])
                        cy = int(m['m01'] / m['m00'])
                        detected_points.append((cx, cy))

            detected_points.sort(key=lambda p: gray_frame[p[1], p[0]], reverse=True)
            tracked_points = self._ordered_tracking(detected_points)
            self.trackers_detected.emit(len(tracked_points))

            if self._scan_room_flag:
                scan_points = self._scan_room_from_frame(main_frame)
                if scan_points:
                    self.warp_points = scan_points
                    self.room_scan_ready.emit(scan_points)
                    self.status_update.emit("Room scan completed")
                else:
                    self.status_update.emit("Room scan failed. Improve edge contrast and retry.")
                self._scan_room_flag = False

            if self._calibrate_depth_flag and len(tracked_points) >= 2:
                self.baseline_distance = np.linalg.norm(np.array(tracked_points[0]) - np.array(tracked_points[1]))
                self._calibrate_depth_flag = False

            for point in tracked_points:
                cv2.circle(main_frame, point, 5, (0, 0, 255), -1)

            masks_to_render = []
            if 0 <= self.active_cue_index < len(self.masks):
                masks_to_render = [self.masks[self.active_cue_index]]
            else:
                masks_to_render = self.masks

            for mask in masks_to_render:
                if mask.type != 'dynamic' or mask.linked_marker_count != len(tracked_points):
                    continue
                if len(mask.source_points) != len(tracked_points) or len(mask.source_points) < 4:
                    continue

                if mask.cue_type == 'generator':
                    frame_cue = self._generate_frame(mask.generator_type, (h, w), tick)
                    ret_cue = True
                else:
                    if mask.video_path not in self.video_captures:
                        self.video_captures[mask.video_path] = cv2.VideoCapture(mask.video_path)
                    cap = self.video_captures[mask.video_path]
                    ret_cue, frame_cue = cap.read()
                    if not ret_cue:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_cue, frame_cue = cap.read()

                if not ret_cue:
                    continue

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
                warped_cue = cv2.warpPerspective(frame_cue, matrix, (w, h), flags=cv2.INTER_LINEAR)

                mask_image = np.zeros_like(projector_output)
                cv2.fillConvexPoly(mask_image, np.int32(dst_pts), (255, 255, 255))
                projector_output = cv2.bitwise_and(projector_output, cv2.bitwise_not(mask_image))
                projector_output = cv2.add(projector_output, cv2.bitwise_and(warped_cue, mask_image))

            if self._capture_still_frame_flag:
                rgb_image_still = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                qt_image_still = QImage(rgb_image_still.data, w, h, w * 3, QImage.Format_RGB888)
                self.still_frame_ready.emit(qt_image_still)
                self._capture_still_frame_flag = False

            rgb_image_main = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
            qt_image_main = QImage(rgb_image_main.data, w, h, w * 3, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image_main)

            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_points = np.float32([[p[0] * w, p[1] * h] for p in self.warp_points])
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_output = cv2.warpPerspective(projector_output, matrix, (w, h), flags=cv2.INTER_LINEAR)

            rgb_image_proj = cv2.cvtColor(warped_output, cv2.COLOR_BGR2RGB)
            qt_image_proj = QImage(rgb_image_proj.data, w, h, w * 3, QImage.Format_RGB888)
            self.projector_frame_ready.emit(qt_image_proj)

            tick += 1
            QThread.msleep(10)

        if main_cap:
            main_cap.release()
        for cap in self.video_captures.values():
            cap.release()

    def stop(self):
        self._running = False

    def set_masks(self, masks):
        self.masks = masks
        current_cues = {mask.video_path for mask in self.masks if mask.video_path}
        for cue in list(self.video_captures.keys()):
            if cue not in current_cues:
                self.video_captures[cue].release()
                del self.video_captures[cue]
