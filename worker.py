import time
from itertools import combinations

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage


class Worker(QObject):
    frame_ready = pyqtSignal(QImage)
    projector_frame_ready = pyqtSignal(QImage)
    still_frame_ready = pyqtSignal(QImage)
    trackers_detected = pyqtSignal(int)
    camera_error = pyqtSignal(int)
    performance_updated = pyqtSignal(float, float, float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.video_source = 0
        self.warp_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.masks = []
        self.video_captures = {}
        self.ir_threshold = 200
        self.threshold_mode = "manual"
        self._camera_changed = True
        self.baseline_distance = 0
        self.depth_sensitivity = 1.0
        self._calibrate_depth_flag = False
        self._capture_still_frame_flag = False
        self.active_cue_index = -1

        self.marker_config = None
        self.marker_fingerprint = []
        self.max_points_to_check = 12
        self.max_combinations_to_check = 1200
        self._dynamic_combination_budget = self.max_combinations_to_check

        self._noise_kernel = np.ones((3, 3), np.uint8)
        self._transform_cache = {}

        self.smoothed_points = []
        self.smoothing_alpha = 0.35
        self.tracking_lost_frames = 0
        self.max_lost_tracking_frames = 5

        self._projector_output_buffer = None
        self._mask_buffer = None
        self._buffer_shape = None

        self._target_fps = 30.0
        self._detection_scale = 0.5
        self._recent_frame_times = []
        self._calibration_distances = []

        self._camera_width = 1280
        self._camera_height = 720
        self._camera_fps = 30

    def set_marker_points(self, points):
        self.marker_config = [(p.x(), p.y()) for p in points]

        if len(self.marker_config) > 1:
            distances = []
            for p1, p2 in combinations(self.marker_config, 2):
                distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
            self.marker_fingerprint = sorted(distances)
        else:
            self.marker_fingerprint = []

    def clear_marker_config(self):
        self.marker_config = None
        self.marker_fingerprint = []

    def capture_still_frame(self):
        self._capture_still_frame_flag = True

    def calibrate_depth(self):
        self._calibrate_depth_flag = True
        self._calibration_distances = []

    def set_depth_sensitivity(self, value):
        self.depth_sensitivity = value

    def set_ir_threshold(self, value):
        self.ir_threshold = value

    def set_threshold_mode(self, mode):
        self.threshold_mode = mode if mode in {"manual", "auto"} else "manual"

    def set_video_source(self, source):
        self.video_source = source
        self._camera_changed = True

    def retry_camera(self):
        self._camera_changed = True

    def set_active_cue_index(self, index):
        self.active_cue_index = index

    def _ensure_buffers(self, h, w):
        if self._buffer_shape == (h, w):
            return
        self._projector_output_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        self._mask_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        self._buffer_shape = (h, w)

    def _extract_detected_points(self, main_frame):
        gray_full = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
        if self._detection_scale < 1.0:
            gray_frame = cv2.resize(
                gray_full,
                None,
                fx=self._detection_scale,
                fy=self._detection_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            gray_frame = gray_full

        if self.threshold_mode == "auto":
            _, thresh = cv2.threshold(
                gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, thresh = cv2.threshold(
                gray_frame, self.ir_threshold, 255, cv2.THRESH_BINARY
            )

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._noise_kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_candidates = []
        scale_back = 1.0 / self._detection_scale if self._detection_scale < 1.0 else 1.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 12:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.3:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = int((moments["m10"] / moments["m00"]) * scale_back)
            cy = int((moments["m01"] / moments["m00"]) * scale_back)
            contour_candidates.append((area, (cx, cy)))

        contour_candidates.sort(key=lambda item: item[0], reverse=True)
        return [point for _, point in contour_candidates]

    def _match_marker_configuration(self, detected_points):
        if not (
            self.marker_config
            and len(self.marker_config) > 1
            and len(detected_points) >= len(self.marker_config)
        ):
            return detected_points

        points_to_check = detected_points[: self.max_points_to_check]
        num_markers = len(self.marker_config)
        src_pts = np.float32(self.marker_config).reshape(-1, 1, 2)

        combinations_checked = 0
        for point_combo in combinations(points_to_check, num_markers):
            combinations_checked += 1
            if combinations_checked > self._dynamic_combination_budget:
                break

            current_distances = []
            for p1, p2 in combinations(point_combo, 2):
                current_distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
            current_fingerprint = sorted(current_distances)

            if len(current_fingerprint) != len(self.marker_fingerprint):
                continue

            if not all(
                np.isclose(value, self.marker_fingerprint[i], rtol=0.15)
                for i, value in enumerate(current_fingerprint)
            ):
                continue

            dst_pts = np.float32(point_combo).reshape(-1, 1, 2)
            matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if matrix is None:
                continue

            transformed_src = cv2.perspectiveTransform(src_pts, matrix)
            ordered_points = [None] * num_markers
            remaining_dst = list(point_combo)

            for i in range(num_markers):
                predicted_pt = transformed_src[i][0]
                closest_actual_pt = min(
                    remaining_dst,
                    key=lambda p: np.linalg.norm(np.array(p) - predicted_pt),
                )
                ordered_points[i] = closest_actual_pt
                remaining_dst.remove(closest_actual_pt)

            return ordered_points

        return []

    def _stabilize_tracked_points(self, tracked_points):
        if not tracked_points:
            self.tracking_lost_frames += 1
            if self.smoothed_points and self.tracking_lost_frames <= self.max_lost_tracking_frames:
                return self.smoothed_points
            self.smoothed_points = []
            return []

        self.tracking_lost_frames = 0
        if len(self.smoothed_points) != len(tracked_points):
            self.smoothed_points = [tuple(p) for p in tracked_points]
            return self.smoothed_points

        stabilized = []
        for prev, curr in zip(self.smoothed_points, tracked_points):
            sx = int(prev[0] * (1 - self.smoothing_alpha) + curr[0] * self.smoothing_alpha)
            sy = int(prev[1] * (1 - self.smoothing_alpha) + curr[1] * self.smoothing_alpha)
            stabilized.append((sx, sy))

        self.smoothed_points = stabilized
        return stabilized

    def _calculate_destination_points(self, tracked_points):
        dst_pts_raw = np.float32(tracked_points)
        if self.baseline_distance > 0 and len(tracked_points) >= 2:
            current_distance = np.linalg.norm(
                np.array(tracked_points[0]) - np.array(tracked_points[1])
            )
            scale_factor = (
                (current_distance / self.baseline_distance - 1.0)
                * self.depth_sensitivity
                + 1.0
            )
            center = np.mean(dst_pts_raw, axis=0)
            return (dst_pts_raw - center) * scale_factor + center
        return dst_pts_raw

    def _compute_transform(self, src_pts, dst_pts):
        if len(src_pts) == 4:
            return cv2.getPerspectiveTransform(src_pts, dst_pts)

        matrix, _ = cv2.findHomography(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2))
        return matrix

    def _get_cached_source_points(self, mask):
        key = id(mask)
        signature = tuple(tuple(p) for p in mask.source_points)
        cache = self._transform_cache.get(key)
        if cache and cache["signature"] == signature:
            return cache["src_pts"]

        src_pts = np.float32(mask.source_points)
        self._transform_cache[key] = {"signature": signature, "src_pts": src_pts}
        return src_pts

    def _is_default_warp(self):
        default = [[0, 0], [1, 0], [1, 1], [0, 1]]
        for p, d in zip(self.warp_points, default):
            if abs(p[0] - d[0]) > 1e-6 or abs(p[1] - d[1]) > 1e-6:
                return False
        return True

    def _open_camera(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
        cap.set(cv2.CAP_PROP_FPS, self._camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def process_video(self):
        main_cap = None
        fps_counter = 0
        fps_window_start = time.perf_counter()

        while self._running:
            frame_start = time.perf_counter()

            if self._camera_changed:
                if main_cap:
                    main_cap.release()
                main_cap = self._open_camera()
                if main_cap is None:
                    self.camera_error.emit(self.video_source)
                self._camera_changed = False

            if main_cap is None:
                QThread.msleep(100)
                continue

            t0 = time.perf_counter()
            ret, main_frame = main_cap.read()
            capture_ms = (time.perf_counter() - t0) * 1000.0
            if not ret:
                self.camera_error.emit(self.video_source)
                QThread.msleep(200)
                continue

            h, w, _ = main_frame.shape
            self._ensure_buffers(h, w)
            projector_output = self._projector_output_buffer
            projector_output.fill(0)

            t0 = time.perf_counter()
            all_detected_points = self._extract_detected_points(main_frame)
            detect_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            tracked_points = self._match_marker_configuration(all_detected_points)
            tracked_points = self._stabilize_tracked_points(tracked_points)
            match_ms = (time.perf_counter() - t0) * 1000.0

            self.trackers_detected.emit(len(tracked_points))

            if self._calibrate_depth_flag and len(tracked_points) >= 2:
                self._calibration_distances.append(
                    np.linalg.norm(np.array(tracked_points[0]) - np.array(tracked_points[1]))
                )
                if len(self._calibration_distances) >= 20:
                    self.baseline_distance = float(np.median(self._calibration_distances))
                    self._calibrate_depth_flag = False
                    self._calibration_distances = []

            for point in tracked_points:
                cv2.circle(main_frame, point, 5, (0, 0, 255), -1)

            t0 = time.perf_counter()
            for i, mask in enumerate(self.masks):
                if self.active_cue_index >= 0 and i != self.active_cue_index:
                    continue
                if mask.type != "dynamic" or mask.linked_marker_count != len(tracked_points):
                    continue

                if mask.video_path not in self.video_captures:
                    self.video_captures[mask.video_path] = cv2.VideoCapture(mask.video_path)

                cap = self.video_captures[mask.video_path]
                ret_cue, frame_cue = cap.read()

                if not ret_cue:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_cue, frame_cue = cap.read()

                if not (ret_cue and mask.source_points):
                    continue

                if len(tracked_points) != len(mask.source_points):
                    continue

                src_pts = self._get_cached_source_points(mask)
                dst_pts = self._calculate_destination_points(tracked_points)
                matrix = self._compute_transform(src_pts, dst_pts)
                if matrix is None:
                    continue

                warped_cue = cv2.warpPerspective(frame_cue, matrix, (w, h))

                mask_image = self._mask_buffer
                mask_image.fill(0)
                cv2.fillPoly(mask_image, [np.int32(dst_pts)], (255, 255, 255))

                projector_output[:] = cv2.bitwise_and(
                    projector_output, cv2.bitwise_not(mask_image)
                )
                projector_output[:] = cv2.add(
                    projector_output, cv2.bitwise_and(warped_cue, mask_image)
                )
            warp_compose_ms = (time.perf_counter() - t0) * 1000.0

            if self._capture_still_frame_flag:
                rgb_image_still = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                qt_image_still = QImage(
                    rgb_image_still.data, w, h, w * 3, QImage.Format_RGB888
                )
                self.still_frame_ready.emit(qt_image_still)
                self._capture_still_frame_flag = False

            rgb_image_main = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
            qt_image_main = QImage(rgb_image_main.data, w, h, w * 3, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image_main)

            t0 = time.perf_counter()
            if self._is_default_warp():
                warped_output = projector_output
            else:
                src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                dst_points = np.float32([[p[0] * w, p[1] * h] for p in self.warp_points])
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                warped_output = cv2.warpPerspective(projector_output, matrix, (w, h))

            rgb_image_proj = cv2.cvtColor(warped_output, cv2.COLOR_BGR2RGB)
            qt_image_proj = QImage(rgb_image_proj.data, w, h, w * 3, QImage.Format_RGB888)
            self.projector_frame_ready.emit(qt_image_proj)
            projector_ms = (time.perf_counter() - t0) * 1000.0

            frame_time_ms = (time.perf_counter() - frame_start) * 1000.0
            self._recent_frame_times.append(frame_time_ms)
            if len(self._recent_frame_times) > 20:
                self._recent_frame_times.pop(0)

            avg_frame = sum(self._recent_frame_times) / len(self._recent_frame_times)
            if avg_frame > 40:
                self._dynamic_combination_budget = max(250, self._dynamic_combination_budget - 100)
            elif avg_frame < 25:
                self._dynamic_combination_budget = min(
                    self.max_combinations_to_check,
                    self._dynamic_combination_budget + 50,
                )

            fps_counter += 1
            elapsed = time.perf_counter() - fps_window_start
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                self.performance_updated.emit(
                    fps,
                    frame_time_ms,
                    detect_ms,
                    match_ms,
                    warp_compose_ms,
                    projector_ms + capture_ms,
                )
                fps_counter = 0
                fps_window_start = time.perf_counter()

            target_ms = 1000.0 / self._target_fps
            sleep_ms = max(0.0, target_ms - frame_time_ms)
            if sleep_ms > 0:
                QThread.msleep(int(sleep_ms))

        if main_cap:
            main_cap.release()
        for cap in self.video_captures.values():
            cap.release()

    def stop(self):
        self._running = False

    def set_warp_points(self, points):
        self.warp_points = points

    def set_masks(self, masks):
        self.masks = masks
        current_cues = {mask.video_path for mask in self.masks}

        valid_mask_ids = {id(mask) for mask in self.masks}
        for cache_id in list(self._transform_cache.keys()):
            if cache_id not in valid_mask_ids:
                del self._transform_cache[cache_id]

        for cue in list(self.video_captures.keys()):
            if cue not in current_cues:
                self.video_captures[cue].release()
                del self.video_captures[cue]
