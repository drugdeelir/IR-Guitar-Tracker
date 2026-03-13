import time
from itertools import combinations
import os
import logging
import platform

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

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
    camera_info_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.video_source = 0
        self.warp_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.masks = []
        self.video_captures = {}
        self.ir_threshold = 215
        self.threshold_mode = "auto"
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
        self._is_windows = platform.system().lower() == "windows"
        self.camera_mode = "native"
        self.show_mask_overlays = True
        self.logger = logging.getLogger("Worker")
        self._last_debug_emit = 0.0

        if self._is_windows:
            self._target_fps = 30.0
            self._detection_scale = 0.45

    def _camera_backends(self):
        if not self._is_windows:
            return [cv2.CAP_ANY]

        preferred = ["CAP_DSHOW", "CAP_ANY"]
        backends = []
        for name in preferred:
            backend = getattr(cv2, name, None)
            if backend is not None and backend not in backends:
                backends.append(backend)
        return backends or [cv2.CAP_ANY]

    @staticmethod
    def _open_capture_with_backend(source, backend):
        try:
            return cv2.VideoCapture(source, backend)
        except TypeError:
            return cv2.VideoCapture(source)

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

    def set_camera_mode(self, mode):
        self.camera_mode = mode if mode in {"native", "performance", "hd"} else "native"
        self._camera_changed = True

    def set_show_mask_overlays(self, enabled):
        self.show_mask_overlays = bool(enabled)

    def _ensure_buffers(self, h, w):
        if self._buffer_shape == (h, w):
            return
        self._projector_output_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        self._mask_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        self._buffer_shape = (h, w)

    def _nms_points(self, scored_points, min_distance=28, limit=20):
        selected = []
        for score, point in sorted(scored_points, key=lambda item: item[0], reverse=True):
            if all((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 >= min_distance ** 2 for _, p in selected):
                selected.append((score, point))
            if len(selected) >= limit:
                break
        return [point for _, point in selected]

    def _extract_detected_points(self, main_frame):
        gray_full = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.GaussianBlur(gray_full, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_full = clahe.apply(gray_full)

        if self._detection_scale < 1.0:
            enhanced = cv2.resize(
                enhanced_full,
                None,
                fx=self._detection_scale,
                fy=self._detection_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            enhanced = enhanced_full

        # Adaptive mask catches dim markers; bright mask catches very bright large blobs.
        if self.threshold_mode == "auto":
            percentile_threshold = int(np.percentile(enhanced, 99.6))
            _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, pct = cv2.threshold(enhanced, percentile_threshold, 255, cv2.THRESH_BINARY)
            adaptive = cv2.bitwise_or(otsu, pct)
        else:
            _, adaptive = cv2.threshold(enhanced, self.ir_threshold, 255, cv2.THRESH_BINARY)

        bright_threshold = 245 if self.threshold_mode == "auto" else max(int(self.ir_threshold), 220)
        _, bright = cv2.threshold(gray_full, bright_threshold, 255, cv2.THRESH_BINARY)
        if self._detection_scale < 1.0:
            bright = cv2.resize(bright, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_AREA)

        thresh = cv2.bitwise_or(adaptive, bright)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._noise_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self._noise_kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_candidates = []
        scale_back = 1.0 / self._detection_scale if self._detection_scale < 1.0 else 1.0
        frame_area = float(enhanced.shape[0] * enhanced.shape[1])
        min_area = max(6.0, frame_area * 0.00001)
        max_area = max(8000.0, frame_area * 0.22)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= min_area or area >= max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.2:
                continue

            mask = np.zeros(enhanced.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            peak = float(cv2.minMaxLoc(enhanced, mask=mask)[1])
            mean_intensity = float(cv2.mean(enhanced, mask=mask)[0])
            if self.threshold_mode == "auto" and peak < 145 and mean_intensity < 90:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx_scaled = int(moments["m10"] / moments["m00"])
            cy_scaled = int(moments["m01"] / moments["m00"])
            cx = int(cx_scaled * scale_back)
            cy = int(cy_scaled * scale_back)

            score = peak * 2.8 + mean_intensity * 1.0 + circularity * 90.0 + min(area, 2500.0) * 0.05
            contour_candidates.append((score, (cx, cy)))

        return self._nms_points(contour_candidates, min_distance=26, limit=24)

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
        best_ordered_points = []
        best_error = float("inf")
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

            reprojection_error = float(
                np.mean(
                    [
                        np.linalg.norm(np.array(ordered_points[i]) - transformed_src[i][0])
                        for i in range(num_markers)
                    ]
                )
            )

            if reprojection_error < best_error:
                best_error = reprojection_error
                best_ordered_points = ordered_points

        return best_ordered_points

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
        mode_profiles = {
            "native": (None, None, None),
            "performance": (960, 540, 30),
            "hd": (1280, 720, 30),
        }
        req_w, req_h, req_fps = mode_profiles.get(self.camera_mode, mode_profiles["native"])

        for backend in self._camera_backends():
            cap = self._open_capture_with_backend(self.video_source, backend)
            if not cap.isOpened():
                cap.release()
                continue

            if req_w and req_h:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
            if req_fps:
                cap.set(cv2.CAP_PROP_FPS, req_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(cv2, "CAP_PROP_ZOOM"):
                cap.set(cv2.CAP_PROP_ZOOM, 0)

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            mode_name = self.camera_mode
            self.camera_info_updated.emit(
                f"Camera mode: {mode_name} | Actual: {actual_w}x{actual_h} @ {actual_fps:.1f} fps"
            )
            return cap

        return None

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

            if self.show_mask_overlays:
                for mask in self.masks:
                    if not mask.source_points:
                        continue
                    pts = np.array(mask.source_points, dtype=np.int32).reshape((-1, 1, 2))
                    if len(pts) >= 3:
                        cv2.polylines(main_frame, [pts], True, (0, 255, 255), 2)

            t0 = time.perf_counter()
            for i, mask in enumerate(self.masks):
                if self.active_cue_index >= 0 and i != self.active_cue_index:
                    continue
                if not mask.source_points:
                    continue

                cue_path = mask.get_active_video_path() if hasattr(mask, "get_active_video_path") else mask.video_path

                if not cue_path:
                    if not self.show_mask_overlays:
                        continue
                    dst_pts = np.float32(mask.source_points)
                    if len(dst_pts) < 3:
                        continue
                    overlay_color = (30, 30, 30) if mask.name.lower() == "background" else (0, 120, 255)
                    cv2.fillPoly(projector_output, [np.int32(dst_pts)], overlay_color)
                    cv2.polylines(projector_output, [np.int32(dst_pts)], True, (255, 255, 255), 2)
                    continue

                if mask.type == "dynamic":
                    if mask.linked_marker_count != len(tracked_points):
                        continue
                    if len(tracked_points) != len(mask.source_points):
                        continue
                    src_pts = self._get_cached_source_points(mask)
                    dst_pts = self._calculate_destination_points(tracked_points)
                else:
                    if len(mask.source_points) < 4:
                        continue
                    dst_pts = np.float32(mask.source_points)
                    fh, fw = 720, 1280
                    src_pts = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
                    if len(dst_pts) != 4:
                        continue

                if cue_path not in self.video_captures:
                    self.video_captures[cue_path] = cv2.VideoCapture(cue_path)

                cap = self.video_captures[cue_path]
                ret_cue, frame_cue = cap.read()

                if not ret_cue:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_cue, frame_cue = cap.read()

                if not ret_cue:
                    continue

                if mask.type != "dynamic":
                    fh, fw = frame_cue.shape[:2]
                    src_pts = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])

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
                ).copy()
                self.still_frame_ready.emit(qt_image_still)
                self.logger.info("still_frame_ready emitted (%dx%d)", w, h)
                self._capture_still_frame_flag = False

            rgb_image_main = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
            qt_image_main = QImage(
                rgb_image_main.data, w, h, w * 3, QImage.Format_RGB888
            ).copy()
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
            qt_image_proj = QImage(
                rgb_image_proj.data, w, h, w * 3, QImage.Format_RGB888
            ).copy()
            self.projector_frame_ready.emit(qt_image_proj)
            projector_ms = (time.perf_counter() - t0) * 1000.0

            frame_time_ms = (time.perf_counter() - frame_start) * 1000.0
            self._recent_frame_times.append(frame_time_ms)

            now = time.perf_counter()
            if now - self._last_debug_emit > 2.0:
                self.logger.info(
                    "frame %.1fms | detected=%d tracked=%d masks=%d threshold=%s:%d",
                    frame_time_ms,
                    len(all_detected_points),
                    len(tracked_points),
                    len(self.masks),
                    self.threshold_mode,
                    self.ir_threshold,
                )
                self._last_debug_emit = now
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
        current_cues = set()
        for mask in self.masks:
            if getattr(mask, "cues", None):
                current_cues.update([c for c in mask.cues if c])
            elif mask.video_path:
                current_cues.add(mask.video_path)

        valid_mask_ids = {id(mask) for mask in self.masks}
        for cache_id in list(self._transform_cache.keys()):
            if cache_id not in valid_mask_ids:
                del self._transform_cache[cache_id]

        for cue in list(self.video_captures.keys()):
            if cue not in current_cues:
                self.video_captures[cue].release()
                del self.video_captures[cue]
