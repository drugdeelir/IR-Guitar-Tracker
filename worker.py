import time
import threading
from itertools import combinations
import os
import logging
import platform

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
# Note: DO NOT disable MSMF — it's the only backend that works for camera 0 on this machine

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage


# ---------------------------------------------------------------------------
# IR detection tuning constants
# Adjust these for different camera/projector/IR-marker hardware setups.
# ---------------------------------------------------------------------------
_BLOB_MIN_HITS = 8          # frames a blob must appear before being reported
_BLOB_MAX_AGE = 15          # frames a blob survives without a new match
_BLOB_HISTORY_RADIUS = 40   # px radius for associating blobs across frames
_MARKER_NMS_RADIUS = 28     # px minimum separation between candidate markers
_MIN_BLOB_BRIGHTNESS = 40   # absolute brightness floor for blob candidates
_DIFF_BRIGHTNESS_FLOOR = 10 # minimum differential brightness (post-calibration mode)
_SCORE_DIFF_WEIGHT = 5.0    # weight for brightness difference in scoring
_SCORE_SIZE_WEIGHT = 3.0    # weight for blob size in scoring
_SAT_PENALTY_THRESHOLD = 60 # saturation level above which a colour penalty applies
_SAT_PENALTY_SCALE = 0.2    # scale factor for saturation penalty
_CALIBRATION_FRAMES = 30    # default frames per calibration phase
_WARMUP_FRAMES_REQUIRED = 90  # good frames needed before auto-calibration fires (~3 s at 30 fps)
# ---------------------------------------------------------------------------


class Worker(QObject):
    frame_ready = pyqtSignal(QImage)
    projector_frame_ready = pyqtSignal(QImage)
    still_frame_ready = pyqtSignal(QImage)
    trackers_detected = pyqtSignal(int)
    camera_error = pyqtSignal(int)
    performance_updated = pyqtSignal(float, float, float, float, float, float)
    camera_info_updated = pyqtSignal(str)
    markers_calibrated = pyqtSignal(list)  # emits list of (x,y) marker positions

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
        self._marker_config_lock = threading.Lock()
        self.max_points_to_check = 12
        self.max_combinations_to_check = 1200
        self._dynamic_combination_budget = self.max_combinations_to_check

        self._noise_kernel = np.ones((5, 5), np.uint8)
        self._transform_cache = {}
        self._blob_detector = self._create_blob_detector()

        self.smoothed_points = []
        self.smoothing_alpha = 0.05   # EMA fallback (unused if Kalman active)
        self.tracking_lost_frames = 0
        self.max_lost_tracking_frames = 15
        self.expected_marker_count = 4  # number of IR markers on the guitar

        # Kalman filters: one per marker point for ultra-smooth tracking
        self._kalman_filters = []  # list of cv2.KalmanFilter objects
        self._kalman_initialized = False

        # Temporal blob tracking: keep history of blob positions across frames
        # Only report blobs that are consistently present
        self._blob_history = []       # list of (cx, cy, peak, age, hit_count)
        self._blob_history_radius = _BLOB_HISTORY_RADIUS
        self._blob_min_hits = _BLOB_MIN_HITS
        self._blob_max_age = _BLOB_MAX_AGE

        # Auto-calibration: save initial stable marker positions
        self._calibrated = False
        self._calibration_frames = []
        self._calibration_target = _CALIBRATION_FRAMES
        self._warmup_frames = 0        # count frames with correct marker count
        self._warmup_required = _WARMUP_FRAMES_REQUIRED

        # Three-phase calibration: dark reference → illuminate → detect in diff image
        self._calib_phase = "idle"       # "idle" (waiting) → "dark" → "illuminate" → "detect" → "proj_scan"
        self._calib_frame_count = 0      # frames in current phase
        self._calib_dark_frames = []     # accumulate dark frames for reference
        self._calib_dark_ref = None      # averaged dark reference (grayscale)
        self._calib_illum_frames = []    # accumulate illuminated frames
        self._calib_illum_ref = None     # averaged illuminated reference (grayscale)
        self._exposure_locked = False    # True during calibration to prevent auto-exposure shifts

        # Inter-marker distance constraint: learned after calibration
        self._marker_distances = None  # sorted list of pairwise distances
        self._distance_tolerance = 0.25  # 25% tolerance on distances

        # Post-calibration local tracking: search near known positions
        self._calibrated_positions = None  # list of (x,y) marker centers
        self._local_search_radius = 50    # pixels around each marker to search

        self._projector_output_buffer = None
        self._mask_buffer = None
        self._buffer_shape = None

        # Projector coverage in camera pixel coords: (x, y, w, h)
        # Computed from diff image during calibration. Used to crop the
        # composited output so only the projector-visible region is sent.
        self._proj_camera_rect = None
        self.debug_solid_colors = True  # Show solid colors instead of video for mask debugging

        # Camera→Projector homography: computed during proj_scan calibration phase.
        # Maps camera pixel coords to projector pixel coords.
        self._cam_to_proj_H = None
        self._proj_resolution = (1920, 1080)  # projector native resolution
        self._pending_markers = None  # markers stored during detect, emitted after proj_scan
        self._guitar_polygon = None   # actual guitar polygon in camera pixel coords (list of (x,y) tuples)

        # Actual camera frame dimensions (set once camera starts)
        self.frame_width = 0
        self.frame_height = 0

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
        self._debug_mode = False  # set True to write calibration debug images to disk

        if self._is_windows:
            self._target_fps = 30.0
            self._detection_scale = 0.45

    def _camera_backends(self):
        """Return backends to try for sustained camera capture (not probing).
        On Windows, MSMF is preferred over DSHOW for continuous reads;
        main.py's _get_camera_backends() uses DSHOW for one-shot index probing."""
        if not self._is_windows:
            return [cv2.CAP_ANY]

        preferred = ["CAP_MSMF", "CAP_ANY", "CAP_DSHOW"]
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
        config = [(p.x(), p.y()) for p in points]
        if len(config) > 1:
            distances = [
                np.linalg.norm(np.array(p1) - np.array(p2))
                for p1, p2 in combinations(config, 2)
            ]
            fingerprint = sorted(distances)
        else:
            fingerprint = []
        with self._marker_config_lock:
            self.marker_config = config
            self.marker_fingerprint = fingerprint

    def clear_marker_config(self):
        with self._marker_config_lock:
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

    def start_calibration(self):
        """Trigger the calibration sequence: dark → illuminate → detect → proj_scan."""
        self.logger.info("Calibration triggered by user")
        self._calibrated = False
        self._calib_phase = "dark"
        self._calib_frame_count = 0
        self._calib_dark_frames = []
        self._calib_dark_ref = None
        self._calib_illum_frames = []
        self._calib_illum_ref = None
        self._cam_to_proj_H = None
        self._guitar_polygon = None
        self._pending_markers = None
        self._blob_history = []

    def _lock_camera_exposure(self, cap):
        """Lock camera exposure at a moderate level for calibration.
        Uses a brightness that can see projector illumination while still being
        consistent between dark and illuminate phases."""
        if cap is None:
            return
        # Read current auto-exposure value
        current_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        # Set manual exposure mode
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode
        # Use a brighter exposure than dark-room auto (-6 is too dark)
        # -4 is a good middle ground: dim enough for dark ref but bright enough
        # to see projector illumination and marker responses
        target_exp = max(current_exp + 2, -4.0)  # at least 2 stops brighter, at least -4
        cap.set(cv2.CAP_PROP_EXPOSURE, target_exp)
        actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        self._exposure_locked = True
        self.logger.info("Camera exposure locked: was=%s, target=%s, actual=%s",
                         current_exp, target_exp, actual_exp)

    def _unlock_camera_exposure(self, cap):
        """Restore camera auto-exposure after calibration."""
        if cap is None:
            return
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto
        self._exposure_locked = False
        self.logger.info("Camera exposure unlocked (auto-exposure restored)")

    def _ensure_buffers(self, h, w):
        if self._buffer_shape == (h, w):
            return
        self._projector_output_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        self._mask_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        self._buffer_shape = (h, w)

    @staticmethod
    def _create_blob_detector():
        """Create a SimpleBlobDetector tuned for large bright IR markers."""
        params = cv2.SimpleBlobDetector_Params()
        params.blobColor = 0
        params.minThreshold = 10
        params.maxThreshold = 255
        params.thresholdStep = 5
        params.filterByArea = True
        params.minArea = 30  # IR markers should be larger, but keep low for distance
        params.maxArea = 200000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minDistBetweenBlobs = 8
        params.minRepeatability = 2
        return cv2.SimpleBlobDetector_create(params)

    def _nms_points(self, scored_points, min_distance=_MARKER_NMS_RADIUS, limit=20):
        selected = []
        for score, point in sorted(scored_points, key=lambda item: item[0], reverse=True):
            if all((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 >= min_distance ** 2 for _, p in selected):
                selected.append((score, point))
            if len(selected) >= limit:
                break
        return [point for _, point in selected]

    def _brightness_weighted_centroid(self, gray, cx, cy, radius):
        """Compute sub-pixel centroid weighted by brightness squared.
        This gives much more accurate center for overexposed IR blobs."""
        h, w = gray.shape
        r = max(6, radius + 3)
        y1, y2 = max(0, int(cy) - r), min(h, int(cy) + r)
        x1, x2 = max(0, int(cx) - r), min(w, int(cx) + r)
        roi = gray[y1:y2, x1:x2].astype(np.float64)
        if roi.size == 0:
            return cx, cy, 0.0

        peak = float(roi.max())
        # Weight by (pixel - threshold)^2 to strongly favor brightest pixels
        thresh = peak * 0.65
        weights = np.maximum(roi - thresh, 0) ** 2
        total_w = weights.sum()
        if total_w > 0:
            ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
            wcx = float((xs * weights).sum() / total_w) + x1
            wcy = float((ys * weights).sum() / total_w) + y1
        else:
            wcx, wcy = float(cx), float(cy)

        return wcx, wcy, peak

    def _update_blob_history(self, new_blobs):
        """Update temporal blob tracking. new_blobs = list of (cx, cy, score).
        Returns list of (cx, cy) for blobs that have been stable for enough frames."""
        radius = self._blob_history_radius
        matched_history = [False] * len(self._blob_history)
        matched_new = [False] * len(new_blobs)

        # Match new blobs to existing history entries
        for ni, (ncx, ncy, nscore) in enumerate(new_blobs):
            best_hi = -1
            best_dist = radius
            for hi, (hcx, hcy, hpeak, hage, hhits) in enumerate(self._blob_history):
                if matched_history[hi]:
                    continue
                d = ((ncx - hcx) ** 2 + (ncy - hcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_hi = hi
            if best_hi >= 0:
                # Update existing history entry with smoothed position
                hcx, hcy, hpeak, hage, hhits = self._blob_history[best_hi]
                alpha = 0.15  # moderate smoothing on history — Kalman handles the rest
                new_hcx = hcx * (1 - alpha) + ncx * alpha
                new_hcy = hcy * (1 - alpha) + ncy * alpha
                self._blob_history[best_hi] = (new_hcx, new_hcy, nscore, 0, hhits + 1)
                matched_history[best_hi] = True
                matched_new[ni] = True

        # Age unmatched history entries, remove old ones
        original_len = len(matched_history)
        updated = []
        for hi in range(original_len):
            hcx, hcy, hpeak, hage, hhits = self._blob_history[hi]
            if not matched_history[hi]:
                hage += 1
                if hage > self._blob_max_age:
                    continue  # remove stale blob
                self._blob_history[hi] = (hcx, hcy, hpeak, hage, hhits)
            updated.append(self._blob_history[hi])

        # Add unmatched new blobs to history
        for ni, (ncx, ncy, nscore) in enumerate(new_blobs):
            if not matched_new[ni]:
                updated.append((ncx, ncy, nscore, 0, 1))

        self._blob_history = updated

        # Return stable blobs sorted by score
        stable = []
        for hcx, hcy, hpeak, hage, hhits in self._blob_history:
            if hhits >= self._blob_min_hits:
                stable.append((hpeak, hcx, hcy))
        stable.sort(key=lambda x: -x[0])
        return stable

    def _extract_detected_points_local(self, main_frame):
        """Post-calibration: search for each marker near its last known position.
        Much more robust than global search when the scene has many bright spots."""
        gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(main_frame, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]
        frame_h, frame_w = gray.shape[:2]

        # Use smoothed_points (Kalman-filtered) as search centers,
        # fall back to calibrated positions
        search_centers = self.smoothed_points if self.smoothed_points else self._calibrated_positions
        if not search_centers or len(search_centers) != self.expected_marker_count:
            return self._extract_detected_points_global(main_frame)

        radius = self._local_search_radius
        result = []

        for cx, cy in search_centers:
            # Extract local ROI around expected marker position
            x1 = max(0, cx - radius)
            x2 = min(frame_w, cx + radius)
            y1 = max(0, cy - radius)
            y2 = min(frame_h, cy + radius)

            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                result.append((cx, cy))  # keep previous position
                continue

            # Find the brightest blob in this local region
            peak_val = float(roi_gray.max())
            if peak_val < 120:
                result.append((cx, cy))  # no bright blob found, hold position
                continue

            # Threshold at 70% of peak to find the bright region
            thresh = int(peak_val * 0.7)
            _, binary = cv2.threshold(roi_gray, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                result.append((cx, cy))
                continue

            # Find the contour closest to the center of the search region
            best_contour = None
            best_dist = float('inf')
            roi_cx, roi_cy = radius, radius  # center of ROI

            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] < 3:
                    continue
                cnt_cx = M["m10"] / M["m00"]
                cnt_cy = M["m01"] / M["m00"]
                d = ((cnt_cx - roi_cx) ** 2 + (cnt_cy - roi_cy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_contour = cnt

            if best_contour is None:
                result.append((cx, cy))
                continue

            # Brightness-weighted centroid for sub-pixel accuracy
            M = cv2.moments(best_contour)
            bcx = M["m10"] / M["m00"] + x1
            bcy = M["m01"] / M["m00"] + y1

            wcx, wcy, peak = self._brightness_weighted_centroid(
                gray, bcx, bcy, max(6, int((cv2.contourArea(best_contour)) ** 0.5))
            )

            # Check saturation — reject colored projector reflections
            sr = 6
            sy1 = max(0, int(wcy) - sr)
            sy2 = min(frame_h, int(wcy) + sr)
            sx1 = max(0, int(wcx) - sr)
            sx2 = min(frame_w, int(wcx) + sr)
            sat_roi = sat_channel[sy1:sy2, sx1:sx2]
            avg_sat = float(sat_roi.mean()) if sat_roi.size else 128

            if avg_sat > 80:
                # Likely a projector reflection, not IR marker — hold position
                result.append((cx, cy))
                continue

            result.append((int(round(wcx)), int(round(wcy))))

        self._auto_thresh_info = f"local_search r={radius} found={len(result)}"
        return result

    def _extract_detected_points(self, main_frame):
        """Main detection entry point. Uses local search after calibration,
        global search before."""
        if self._calibrated and self._calibrated_positions:
            return self._extract_detected_points_local(main_frame)
        return self._extract_detected_points_global(main_frame)

    def _extract_detected_points_global(self, main_frame):
        gray_full = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(main_frame, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]
        frame_h, frame_w = gray_full.shape[:2]

        # Light blur to reduce sensor noise but preserve blob peaks
        blurred = cv2.GaussianBlur(gray_full, (5, 5), 0)

        edge_margin = max(3, int(min(frame_w, frame_h) * 0.005))

        # --- SimpleBlobDetector on inverted image ---
        inverted = cv2.bitwise_not(blurred)
        blob_keypoints = self._blob_detector.detect(inverted)

        # Score each blob using brightness, size, saturation, and differential response
        frame_blobs = []
        diag_all = []  # diagnostic: all blobs before filtering
        has_dark_ref = (self._calib_dark_ref is not None and
                        self._calib_dark_ref.shape == gray_full.shape)

        for kp in blob_keypoints:
            bcx, bcy = kp.pt[0], kp.pt[1]
            r = max(4, int(kp.size / 2))

            # Brightness-weighted centroid
            wcx, wcy, peak = self._brightness_weighted_centroid(
                gray_full, bcx, bcy, r
            )

            if wcx < edge_margin or wcx > frame_w - edge_margin:
                continue
            if wcy < edge_margin or wcy > frame_h - edge_margin:
                continue

            # Check saturation for all blobs (needed for scoring and filtering)
            sr = max(4, int(kp.size / 2) + 2)
            sy1 = max(0, int(wcy) - sr)
            sy2 = min(frame_h, int(wcy) + sr)
            sx1 = max(0, int(wcx) - sr)
            sx2 = min(frame_w, int(wcx) + sr)
            sat_roi = sat_channel[sy1:sy2, sx1:sx2]
            avg_sat = float(sat_roi.mean()) if sat_roi.size else 128

            # Compute differential brightness: how much brighter is this spot
            # compared to when the projector was OFF?
            # High diff = responds strongly to projector (retroreflective marker)
            # Low diff = static light source (monitor, lamp) — NOT a marker
            diff_brightness = 0.0
            if has_dark_ref:
                dark_val = float(self._calib_dark_ref[
                    max(0, min(int(wcy), frame_h - 1)),
                    max(0, min(int(wcx), frame_w - 1))
                ])
                diff_brightness = max(0, peak - dark_val)

            # Diagnostic: log all blobs (before filtering)
            if not self._calibrated and self._warmup_frames < 5:
                diag_all.append((
                    int(wcx), int(wcy), int(peak), int(avg_sat),
                    f"{kp.size:.1f}", f"d={diff_brightness:.0f}"
                ))

            if peak < _MIN_BLOB_BRIGHTNESS:
                continue

            if has_dark_ref:
                # DIFFERENTIAL MODE: score based on how much blob responds to projector
                # Static sources (monitors, lamps) have diff≈0, markers have high diff
                if diff_brightness < _DIFF_BRIGHTNESS_FLOOR:
                    continue  # static source, not a projector-responsive marker
                # Score: differential response is primary, size and whiteness are secondary
                sat_penalty = max(0, avg_sat - _SAT_PENALTY_THRESHOLD) * _SAT_PENALTY_SCALE
                score = diff_brightness * _SCORE_DIFF_WEIGHT + kp.size * _SCORE_SIZE_WEIGHT - sat_penalty
            else:
                # NO REFERENCE: fall back to absolute scoring
                sat_penalty = max(0, avg_sat - 40) * 0.3 + max(0, avg_sat - 120) * 1.0
                score = kp.size * 10.0 + peak * 2.0 - sat_penalty

            frame_blobs.append((wcx, wcy, score))

        # Dump diagnostic for first few frames
        if diag_all and not self._calibrated and self._warmup_frames < 5:
            diag_all.sort(key=lambda x: -x[2])  # sort by peak brightness
            self.logger.info("DIAG blobs (x,y,peak,sat,size,diff): %s", diag_all[:15])

        # Update temporal history and get stable blobs
        stable = self._update_blob_history(frame_blobs)

        # Take top N stable blobs by score (brightest + largest + whitest)
        # IR markers are the biggest, brightest, whitest blobs in the scene
        target_n = self.expected_marker_count
        scored = [(s, (int(round(cx)), int(round(cy)))) for s, cx, cy in stable]
        all_candidates = self._nms_points(scored, min_distance=15, limit=target_n * 3)
        result = all_candidates[:target_n]

        # Log top candidate sizes for debugging
        top_sizes = ""
        if scored:
            top_scored = sorted(scored, key=lambda x: x[0], reverse=True)[:6]
            top_sizes = " top=[" + ",".join(f"{s:.0f}@{p}" for s, p in top_scored) + "]"
        self._auto_thresh_info = (
            f"blob_det={len(blob_keypoints)} frame={len(frame_blobs)} "
            f"stable={len(stable)} final={len(result)}{top_sizes}"
        )
        return result

    def _detect_markers_from_diff(self):
        """Guitar silhouette detection using differential imaging.

        The guitar appears as a DARK shape against the bright wall when
        the projector illuminates the scene. We detect this silhouette
        and extract 4 key points from it (body corners + neck tip).

        Strategy:
        1. Diff image → find projector coverage area (bright wall)
        2. Within projector area, find DARK regions (objects blocking projector light)
        3. The tallest dark region = the guitar
        4. Extract 4 feature points from the guitar contour
        """
        if self._calib_dark_ref is None or self._calib_illum_ref is None:
            self.logger.error("Cannot detect markers: missing dark/illum references")
            return None

        dark = self._calib_dark_ref
        illum = self._calib_illum_ref
        frame_h, frame_w = dark.shape[:2]

        import os as _os
        _base = _os.path.dirname(_os.path.abspath(__file__))

        # --- Step 1: Compute diff to find projector coverage area ---
        diff = np.clip(illum.astype(np.float32) - dark.astype(np.float32), 0, 255).astype(np.uint8)
        diff_blurred = cv2.GaussianBlur(diff, (15, 15), 0)
        if self._debug_mode:
            cv2.imwrite(_os.path.join(_base, "diff_image.png"), diff)

        # Projector mask: areas significantly brighter with projector on
        diff_mean = float(diff_blurred.mean())
        diff_std = float(diff_blurred.std())
        proj_thresh = max(5, diff_mean + diff_std * 0.5)
        _, proj_mask = cv2.threshold(diff_blurred, int(proj_thresh), 255, cv2.THRESH_BINARY)

        kernel = np.ones((15, 15), np.uint8)
        proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_OPEN, kernel)
        proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_CLOSE, kernel)
        if self._debug_mode:
            cv2.imwrite(_os.path.join(_base, "projector_mask.png"), proj_mask)

        proj_area = int(np.count_nonzero(proj_mask))
        self.logger.info("Projector coverage: %.0f%% of frame, thresh=%d",
                         100.0 * proj_area / (frame_w * frame_h), int(proj_thresh))

        if proj_area < frame_w * frame_h * 0.05:
            self.logger.warning("Projector coverage too small")
            return None

        # Compute bounding rect of projector coverage for camera→projector mapping
        proj_contours_rect, _ = cv2.findContours(proj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if proj_contours_rect:
            # Use the largest contour (the main projector area)
            largest_proj = max(proj_contours_rect, key=cv2.contourArea)
            px, py, pw, ph = cv2.boundingRect(largest_proj)
            self._proj_camera_rect = (px, py, pw, ph)
            self.logger.info(
                "Projector rect in camera space: x=%d y=%d w=%d h=%d (%.1f%% of frame)",
                px, py, pw, ph, 100.0 * pw * ph / (frame_w * frame_h)
            )

        # --- Step 2: Find guitar silhouette using ratio-based detection ---
        # Instead of absolute brightness thresholding (which fails when the guitar
        # is front-lit by reflected projector light), use the RATIO of illuminate
        # to dark brightness. The wall gets much brighter with the projector
        # (high ratio), while objects blocking the light stay similar (low ratio).
        illum_blurred = cv2.GaussianBlur(illum, (9, 9), 0)
        dark_blurred = cv2.GaussianBlur(dark, (9, 9), 0)

        # Compute brightness ratio: how much brighter did each pixel get?
        dark_safe = dark_blurred.astype(np.float32) + 1.0  # avoid division by zero
        illum_f = illum_blurred.astype(np.float32)
        ratio = illum_f / dark_safe

        # Within projector area, compute ratio stats
        proj_pixels_ratio = ratio[proj_mask > 0]
        ratio_median = float(np.median(proj_pixels_ratio))
        ratio_p25 = float(np.percentile(proj_pixels_ratio, 25))

        # Also compute brightness stats for logging
        proj_pixels_illum = illum_blurred[proj_mask > 0]
        illum_median = float(np.median(proj_pixels_illum))

        self.logger.info("Ratio stats: median=%.2f, p25=%.2f, illum_median=%.0f",
                         ratio_median, ratio_p25, illum_median)

        # Projector rect center: prefer contours near the center of projector coverage
        if self._proj_camera_rect:
            prx, pry, prw, prh = self._proj_camera_rect
            proj_cx = prx + prw / 2
            proj_cy = pry + prh / 2
            inner_x1 = prx + prw * 0.10
            inner_x2 = prx + prw * 0.90
            inner_y1 = pry + prh * 0.10
            inner_y2 = pry + prh * 0.90
        else:
            proj_cx, proj_cy = frame_w / 2, frame_h / 2
            inner_x1, inner_x2 = frame_w * 0.10, frame_w * 0.90
            inner_y1, inner_y2 = frame_h * 0.10, frame_h * 0.90

        # --- Edge-based guitar detection ---
        # Use Canny edge detection on the illuminated frame within the projector
        # area. The guitar creates strong edges against the bright wall. Find
        # contours from edges and pick the tallest vertical one as the guitar.
        #
        # Also try adaptive thresholding: detects local brightness changes,
        # finding objects that are darker than their immediate surroundings.

        best_candidate = None
        used_method = "edge"

        # Method 1: Adaptive threshold within projector area
        # This finds objects darker than their local neighborhood
        illum_proj = illum_blurred.copy()
        illum_proj[proj_mask == 0] = 255  # mask out non-projector area
        adapt = cv2.adaptiveThreshold(
            illum_proj, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 15  # block=51, C=15: detect dark features
        )
        adapt[proj_mask == 0] = 0  # only within projector area
        adapt_kernel = np.ones((5, 5), np.uint8)
        adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, adapt_kernel)
        adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if self._debug_mode:
            cv2.imwrite(_os.path.join(_base, "adaptive_thresh.png"), adapt)

        # Method 2: Canny edge detection on illuminated frame
        edges = cv2.Canny(illum_blurred, 30, 80)
        edges[proj_mask == 0] = 0
        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edge_closed = cv2.morphologyEx(edge_dilated, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        if self._debug_mode:
            cv2.imwrite(_os.path.join(_base, "edges.png"), edges)

        # Also use ratio-based detection at a moderate threshold
        ratio_thresh = ratio_median * 0.75  # moderate: things below 75% of median ratio
        ratio_sil = np.zeros((frame_h, frame_w), dtype=np.uint8)
        ratio_sil[(proj_mask > 0) & (ratio < ratio_thresh)] = 255
        ratio_sil = cv2.morphologyEx(ratio_sil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        ratio_sil = cv2.morphologyEx(ratio_sil, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Combine: use adaptive threshold AND ratio to find objects
        # Areas that are locally dark AND have low brightness ratio = strong guitar candidates
        combined = cv2.bitwise_and(adapt, ratio_sil)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if self._debug_mode:
            cv2.imwrite(_os.path.join(_base, "combined_detection.png"), combined)

        # Try each detection method: combined first, then adaptive, then ratio alone
        detection_methods = [
            ("combined", combined),
            ("adaptive", adapt),
            ("ratio", ratio_sil),
        ]

        for method_name, sil_image in detection_methods:
            contours, _ = cv2.findContours(sil_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self.logger.info("  %s: no contours", method_name)
                continue

            for cnt_i in contours:
                area = cv2.contourArea(cnt_i)
                if area < 200 or area > frame_w * frame_h * 0.08:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt_i)
                if bh < 40:
                    continue
                aspect = bh / max(1, bw)
                if aspect < 1.3:
                    continue
                cnt_cx = x + bw / 2
                cnt_cy = y + bh / 2
                is_inner = (inner_x1 <= cnt_cx <= inner_x2 and inner_y1 <= cnt_cy <= inner_y2)

                dx = (cnt_cx - proj_cx) / max(1, prw / 2 if self._proj_camera_rect else frame_w / 2)
                dy = (cnt_cy - proj_cy) / max(1, prh / 2 if self._proj_camera_rect else frame_h / 2)
                centrality = 1.0 / (1.0 + dx ** 2 + dy ** 2)
                inner_mult = 3.0 if is_inner else 0.3
                aspect_bonus = aspect ** 1.2
                area_factor = (area / 500.0) ** 0.5
                score = bh * aspect_bonus * centrality * inner_mult * area_factor

                self.logger.info(
                    "  %s: bbox=(%d,%d,%d,%d) area=%.0f aspect=%.1f inner=%s score=%.1f",
                    method_name, x, y, bw, bh, area, aspect,
                    "Y" if is_inner else "N", score
                )

                if best_candidate is None or score > best_candidate[1]:
                    best_candidate = ((cnt_i, area, x, y, bw, bh, aspect), score, ratio_thresh)
                    used_method = method_name

            # If we found a good inner candidate, stop trying other methods
            if best_candidate:
                cand_info = best_candidate[0]
                cand_cx = cand_info[2] + cand_info[4] / 2
                if (inner_x1 <= cand_cx <= inner_x2) and cand_info[5] >= 60 and cand_info[1] >= 500:
                    self.logger.info("  Found guitar via %s, stopping", method_name)
                    break

        if best_candidate is None:
            self.logger.warning("No guitar-shaped silhouettes found at any threshold")
            return None

        (cnt, area, gx, gy, gw, gh, aspect), score, used_thresh = best_candidate

        self.logger.info("Guitar detection: method=%s thresh=%.1f", used_method, used_thresh)
        self.logger.info("Guitar neck detected: bbox=(%d,%d,%d,%d) area=%.0f aspect=%.1f score=%.1f",
                         gx, gy, gw, gh, area, aspect, score)

        # Regenerate the silhouette at the best threshold for saving
        if used_method == "ratio":
            silhouette = np.zeros((frame_h, frame_w), dtype=np.uint8)
            silhouette[(proj_mask > 0) & (ratio < used_thresh)] = 255
        else:
            silhouette = np.zeros((frame_h, frame_w), dtype=np.uint8)
            silhouette[(proj_mask > 0) & (illum_blurred < used_thresh)] = 255
        sil_kernel = np.ones((7, 7), np.uint8)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, sil_kernel)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if self._debug_mode:
            cv2.imwrite(_os.path.join(_base, "silhouette.png"), silhouette)
            ratio_vis = np.clip(ratio * 50, 0, 255).astype(np.uint8)  # scale for visibility
            cv2.imwrite(_os.path.join(_base, "ratio_image.png"), ratio_vis)

        # Collect ALL contours at this threshold for body-merging later
        contours_final, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = []
        for c in contours_final:
            a = cv2.contourArea(c)
            bx, by, bbw, bbh = cv2.boundingRect(c)
            ba = bbh / max(1, bbw)
            all_contours.append((c, a, bx, by, bbw, bbh, ba))
        all_contours.sort(key=lambda c: -c[1])
        for c in all_contours[:10]:
            self.logger.info("Silhouette contour: bbox=(%d,%d,%d,%d) area=%.0f aspect=%.1f",
                             c[2], c[3], c[4], c[5], c[1], c[6])

        # --- Step 3: Build guitar polygon from neck contour + nearby body contours ---
        # The detected contour is typically the guitar NECK (the body reflects
        # projector light instead of blocking it). We find nearby contours that
        # could be the body and merge them, then build a tight polygon.
        neck_pts = cnt.reshape(-1, 2)
        neck_top_y = int(np.min(neck_pts[:, 1]))
        neck_bot_y = int(np.max(neck_pts[:, 1]))
        neck_cx = int(np.mean(neck_pts[:, 0]))
        neck_left_x = int(np.min(neck_pts[:, 0]))
        neck_right_x = int(np.max(neck_pts[:, 0]))
        neck_w = max(neck_right_x - neck_left_x, 10)
        neck_h = neck_bot_y - neck_top_y

        # Search for body contours: any contour near the bottom of the neck
        # that could be the guitar body (wider, below neck)
        body_search_x_range = neck_w * 5  # body is wider than neck
        body_search_y_below = neck_h * 2  # body extends below neck
        merged_contour_pts = list(neck_pts)  # start with neck points

        for other_cnt, other_area, ox, oy, ow, oh, o_aspect in all_contours:
            if other_cnt is cnt:
                continue
            if other_area < 100:
                continue
            # Check if this contour is near/below the neck
            other_cx = ox + ow // 2
            other_cy = oy + oh // 2
            # Must be horizontally close to neck center
            if abs(other_cx - neck_cx) > body_search_x_range:
                continue
            # Must be below or overlapping with the neck bottom
            if other_cy < neck_top_y - 20:
                continue
            if oy > neck_bot_y + body_search_y_below:
                continue
            # This contour is a candidate body part
            self.logger.info("  Merging body contour: bbox=(%d,%d,%d,%d) area=%.0f",
                             ox, oy, ow, oh, other_area)
            merged_contour_pts.extend(other_cnt.reshape(-1, 2).tolist())

        # Build convex hull from merged points
        merged_arr = np.array(merged_contour_pts, dtype=np.int32)
        hull = cv2.convexHull(merged_arr)
        hull_pts = hull.reshape(-1, 2)

        # Simplify the hull to a reasonable polygon (8-16 points)
        perimeter = cv2.arcLength(hull, True)
        epsilon = perimeter * 0.02  # 2% approximation
        approx = cv2.approxPolyDP(hull, epsilon, True)
        guitar_poly = [(int(p[0][0]), int(p[0][1])) for p in approx]

        # If merged contour is too small (only neck, no body found),
        # extrapolate the body below the neck. Adjust multiplier based on
        # how wide the detected contour already is.
        hull_bbox = cv2.boundingRect(hull)
        hull_w, hull_h = hull_bbox[2], hull_bbox[3]
        if hull_h < neck_h * 1.3 or hull_w < neck_w * 1.5:
            self.logger.info("No body contours found near neck, extrapolating body (neck_w=%d)", neck_w)
            # If the contour is narrow (pure neck, <30px), use larger body multiplier.
            # If wider (already includes some body), use smaller multiplier.
            if neck_w < 30:
                body_mult = 3.0  # narrow neck → bigger body extrapolation
            elif neck_w < 50:
                body_mult = 2.0  # medium width → moderate body
            else:
                body_mult = 1.5  # wide contour already includes body

            body_w = max(int(neck_w * body_mult), 60)
            body_h = max(int(neck_h * 0.8), 60)  # body height ~80% of neck
            body_top_y = neck_bot_y
            body_bot_y = min(body_top_y + body_h, frame_h - 5)
            body_left = max(neck_cx - body_w // 2, 5)
            body_right = min(neck_cx + body_w // 2, frame_w - 5)
            # T-shape polygon: neck on top, body below
            neck_pad = max(3, neck_w // 6)
            guitar_poly = [
                (neck_left_x - neck_pad, neck_top_y),        # top-left of neck
                (neck_right_x + neck_pad, neck_top_y),       # top-right of neck
                (neck_right_x + neck_pad, body_top_y),       # neck meets body (right)
                (body_right, body_top_y),                     # body top-right
                (body_right, body_bot_y),                     # body bottom-right
                (body_left, body_bot_y),                      # body bottom-left
                (body_left, body_top_y),                      # body top-left
                (neck_left_x - neck_pad, body_top_y),         # neck meets body (left)
            ]
            self.logger.info("Body extrapolation: mult=%.1f, body_w=%d, body_h=%d, total=%dx%d",
                             body_mult, body_w, body_h,
                             body_right - body_left, body_bot_y - neck_top_y)

        # Store the polygon for direct use as a mask
        self._guitar_polygon = guitar_poly

        # Also compute 4 key markers for tracking (extremes of the polygon)
        poly_arr = np.array(guitar_poly)
        min_x, min_y = poly_arr.min(axis=0)
        max_x, max_y = poly_arr.max(axis=0)
        cx_poly = int((min_x + max_x) / 2)
        markers = [
            (cx_poly, int(min_y)),                    # top center
            (int(min_x), int((min_y + max_y) / 2)),   # left center
            (int(max_x), int((min_y + max_y) / 2)),   # right center
            (cx_poly, int(max_y)),                     # bottom center
        ]

        self.logger.info(
            "Guitar polygon (%d points): %s",
            len(guitar_poly), guitar_poly
        )
        self.logger.info(
            "Guitar estimated: neck=(%d,%d-%d) w=%d, bbox=(%d,%d to %d,%d), markers=%s",
            neck_cx, neck_top_y, neck_bot_y, neck_w,
            min_x, min_y, max_x, max_y, markers
        )

        # Save debug visualization
        debug = cv2.cvtColor(illum, cv2.COLOR_GRAY2BGR)
        proj_contours_dbg, _ = cv2.findContours(proj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, proj_contours_dbg, -1, (0, 255, 255), 1)
        cv2.drawContours(debug, [cnt], -1, (0, 255, 0), 2)
        # Draw the guitar polygon in bright red
        poly_np = np.array(guitar_poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug, [poly_np], True, (0, 0, 255), 2)
        # Draw merged hull in cyan
        cv2.drawContours(debug, [hull], -1, (255, 255, 0), 1)
        labels = ["TOP", "LEFT", "RIGHT", "BOT"]
        for i, (mx, my) in enumerate(markers):
            cv2.circle(debug, (mx, my), 8, (0, 0, 255), -1)
            cv2.putText(debug, f"{labels[i]} ({mx},{my})",
                        (mx + 12, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if self._debug_mode:
            cv2.imwrite(_os.path.join(_base, "diff_debug.png"), debug)

        return markers

    def _validate_marker_distances(self, points):
        """After calibration, reject detection if inter-marker distances
        don't match the learned pattern (markers are always same distance apart)."""
        if not self._marker_distances or len(points) != self.expected_marker_count:
            return points

        current_dists = []
        for p1, p2 in combinations(points, 2):
            current_dists.append(
                float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
            )
        current_dists.sort()

        # Use adaptive tolerance: for small distances, use absolute tolerance;
        # for large distances, use relative tolerance
        abs_tol = 10.0  # pixels
        rel_tol = self._distance_tolerance
        mismatches = 0
        for cd, ld in zip(current_dists, self._marker_distances):
            if ld <= 0:
                continue
            # Allow whichever is more lenient: absolute or relative
            if abs(cd - ld) > max(abs_tol, ld * rel_tol):
                mismatches += 1

        # If more than half the distances are off, reject
        if mismatches > len(self._marker_distances) // 2:
            return []  # reject — stabilizer will hold previous positions

        return points

    def _match_marker_configuration(self, detected_points):
        with self._marker_config_lock:
            marker_config = self.marker_config
            marker_fingerprint = list(self.marker_fingerprint)

        if not (
            marker_config
            and len(marker_config) > 1
            and len(detected_points) >= len(marker_config)
        ):
            return detected_points

        points_to_check = detected_points[: self.max_points_to_check]
        num_markers = len(marker_config)
        src_pts = np.float32(marker_config).reshape(-1, 1, 2)

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

            if len(current_fingerprint) != len(marker_fingerprint):
                continue

            if not all(
                np.isclose(value, marker_fingerprint[i], rtol=0.15)
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

    def _match_points_to_previous(self, new_points):
        """Match new detections to previous smoothed points via nearest-neighbor
        to prevent point identity swapping between frames."""
        if not self.smoothed_points or len(self.smoothed_points) != len(new_points):
            return new_points

        n = len(new_points)
        # Build cost matrix: distance from each previous point to each new point
        used_new = [False] * n
        ordered = [None] * n

        # Greedy nearest-neighbor assignment
        prev_arr = np.array(self.smoothed_points, dtype=np.float32)
        new_arr = np.array(new_points, dtype=np.float32)

        for _ in range(n):
            best_dist = float('inf')
            best_prev = -1
            best_new = -1
            for pi in range(n):
                if ordered[pi] is not None:
                    continue
                for ni in range(n):
                    if used_new[ni]:
                        continue
                    d = float(np.sum((prev_arr[pi] - new_arr[ni]) ** 2))
                    if d < best_dist:
                        best_dist = d
                        best_prev = pi
                        best_new = ni
            if best_prev >= 0:
                ordered[best_prev] = new_points[best_new]
                used_new[best_new] = True

        # Fill any remaining (shouldn't happen, but safety)
        for i in range(n):
            if ordered[i] is None:
                ordered[i] = new_points[i]

        return ordered

    def _init_kalman_filter(self, x, y):
        """Create a Kalman filter for one marker point.
        State: [x, y, vx, vy] — position + velocity.
        Measurement: [x, y] — position only."""
        kf = cv2.KalmanFilter(4, 2)  # 4 state dims, 2 measurement dims

        # Transition matrix: position updates by velocity each frame
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix: we only observe position
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise: very low = trust the model, resist jitter
        # Markers are mostly stationary, so keep this very small
        pn = 0.005  # process noise magnitude
        kf.processNoiseCov = np.array([
            [pn, 0,  0,  0],
            [0,  pn, 0,  0],
            [0,  0,  pn * 0.5, 0],
            [0,  0,  0,  pn * 0.5],
        ], dtype=np.float32)

        # Measurement noise: moderate = measurements are somewhat noisy
        mn = 8.0  # measurement noise — higher = more smoothing
        kf.measurementNoiseCov = np.array([
            [mn, 0],
            [0,  mn],
        ], dtype=np.float32)

        # Initial state
        kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

        # Error covariance: start with moderate uncertainty
        kf.errorCovPre = np.eye(4, dtype=np.float32) * 10
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10

        return kf

    def _stabilize_tracked_points(self, tracked_points):
        if not tracked_points:
            self.tracking_lost_frames += 1
            if self.smoothed_points and self.tracking_lost_frames <= self.max_lost_tracking_frames:
                # Predict without measurement to maintain positions
                if self._kalman_initialized:
                    for kf in self._kalman_filters:
                        kf.predict()
                return self.smoothed_points
            self.smoothed_points = []
            self._kalman_filters = []
            self._kalman_initialized = False
            return []

        self.tracking_lost_frames = 0

        # If number of points changed, reinitialize Kalman filters
        if len(self.smoothed_points) != len(tracked_points):
            self.smoothed_points = [tuple(p) for p in tracked_points]
            self._kalman_filters = [
                self._init_kalman_filter(float(p[0]), float(p[1]))
                for p in tracked_points
            ]
            self._kalman_initialized = True
            return self.smoothed_points

        # Match new points to previous positions to prevent identity swapping
        matched = self._match_points_to_previous(tracked_points)

        # Update Kalman filters with new measurements
        stabilized = []
        for i, (prev, curr) in enumerate(zip(self.smoothed_points, matched)):
            kf = self._kalman_filters[i]

            # Check for large jumps — if marker teleported, reset that filter
            dist = ((prev[0] - curr[0]) ** 2 + (prev[1] - curr[1]) ** 2) ** 0.5
            if dist > 60:
                self._kalman_filters[i] = self._init_kalman_filter(
                    float(curr[0]), float(curr[1])
                )
                stabilized.append((int(curr[0]), int(curr[1])))
                continue

            # Predict next state
            kf.predict()

            # Correct with measurement
            measurement = np.array([[float(curr[0])], [float(curr[1])]], dtype=np.float32)
            corrected = kf.correct(measurement)

            sx = int(round(float(corrected[0, 0])))
            sy = int(round(float(corrected[1, 0])))
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
            self.logger.info("Trying camera %s with backend %s", self.video_source, backend)
            cap = self._open_capture_with_backend(self.video_source, backend)
            if not cap.isOpened():
                self.logger.info("Backend %s failed to open camera", backend)
                cap.release()
                continue

            self.logger.info("Camera opened with backend %s", backend)
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
        try:
            self._process_video_inner()
        except Exception:
            self.logger.exception("FATAL: process_video crashed")

    def _process_video_inner(self):
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
            self.frame_width = w
            self.frame_height = h
            self._ensure_buffers(h, w)
            projector_output = self._projector_output_buffer
            projector_output.fill(0)

            # ---- Calibration state machine ----
            # If idle (not calibrating and not calibrated), just show camera feed
            if not self._calibrated and self._calib_phase == "idle":
                # Show status overlay prompting user to calibrate
                cv2.putText(main_frame, "Press 'Calibrate' to start",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                qt_main = QImage(
                    cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.frame_ready.emit(qt_main)
                # Send black to projector when not calibrated
                qt_proj = QImage(
                    cv2.cvtColor(projector_output, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.projector_frame_ready.emit(qt_proj)
                self.trackers_detected.emit(0)
                QThread.msleep(int(1000.0 / self._target_fps))
                continue

            # Phase 1: DARK - projector OFF, collect dark reference
            if not self._calibrated and self._calib_phase == "dark":
                self._calib_frame_count += 1
                # No exposure lock — let auto-exposure handle each phase naturally
                gray_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                if self._calib_frame_count > 10:  # skip first 10 (camera settling)
                    self._calib_dark_frames.append(gray_frame.astype(np.float32))
                if self._calib_frame_count >= 25:
                    if self._calib_dark_frames:
                        self._calib_dark_ref = np.mean(
                            self._calib_dark_frames, axis=0
                        ).astype(np.uint8)
                        if self._debug_mode:
                            import os as _os
                            _base = _os.path.dirname(_os.path.abspath(__file__))
                            cv2.imwrite(_os.path.join(_base, "dark_ref.png"), self._calib_dark_ref)
                    self._calib_dark_frames = []  # free memory
                    self._calib_phase = "illuminate"
                    self._calib_frame_count = 0
                    self.logger.info("Calibration: dark reference captured, switching to illuminate phase")
                # Show status, emit frames, skip detection
                cv2.putText(main_frame, f"CALIBRATING - Dark ref {self._calib_frame_count}/25",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                qt_main = QImage(
                    cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.frame_ready.emit(qt_main)
                qt_proj = QImage(
                    cv2.cvtColor(projector_output, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.projector_frame_ready.emit(qt_proj)
                self.trackers_detected.emit(0)
                continue

            # Phase 2: ILLUMINATE - projector FULL WHITE, collect illuminated reference
            if not self._calibrated and self._calib_phase == "illuminate":
                projector_output[:] = 200  # bright white for retroreflective response (not 255 to avoid camera auto-exposure saturation)
                self._calib_frame_count += 1
                gray_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                if self._calib_frame_count > 30:  # skip first 30 (projector on + auto-exposure settle)
                    self._calib_illum_frames.append(gray_frame.astype(np.float32))
                if self._calib_frame_count >= 45:
                    if self._calib_illum_frames:
                        self._calib_illum_ref = np.mean(
                            self._calib_illum_frames, axis=0
                        ).astype(np.uint8)
                        if self._debug_mode:
                            import os as _os
                            _base = _os.path.dirname(_os.path.abspath(__file__))
                            cv2.imwrite(_os.path.join(_base, "illum_ref.png"), self._calib_illum_ref)
                    self._calib_illum_frames = []  # free memory
                    self._calib_phase = "detect"
                    self._calib_frame_count = 0
                    self.logger.info("Calibration: illuminate reference captured, switching to detect phase")
                # Show status, emit frames
                cv2.putText(main_frame, f"CALIBRATING - Illuminate {self._calib_frame_count}/45",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                qt_main = QImage(
                    cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.frame_ready.emit(qt_main)
                qt_proj = QImage(
                    cv2.cvtColor(projector_output, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.projector_frame_ready.emit(qt_proj)
                self.trackers_detected.emit(0)
                continue

            # Phase 3: DETECT - one-shot difference-based marker detection
            if not self._calibrated and self._calib_phase == "detect":
                self._calib_phase = "done"  # only run once
                self.logger.info("Calibration: running difference-based marker detection")
                markers = self._detect_markers_from_diff()
                if markers and len(markers) == self.expected_marker_count:
                    # Store markers — don't finalize yet, proceed to proj_scan
                    self._pending_markers = markers
                    self._calibrated_positions = list(markers)
                    # Learn inter-marker distances
                    dists = []
                    for p1, p2 in combinations(markers, 2):
                        dists.append(float(np.linalg.norm(
                            np.array(p1, dtype=np.float64) - np.array(p2, dtype=np.float64)
                        )))
                    self._marker_distances = sorted(dists)
                    self.logger.info(
                        "Markers found: %s, distances: %s — proceeding to projector scan",
                        markers, [f"{d:.1f}" for d in self._marker_distances]
                    )
                    # Save debug image
                    debug_frame = main_frame.copy()
                    for mi, (mx, my) in enumerate(markers):
                        cv2.circle(debug_frame, (mx, my), 15, (0, 0, 255), 3)
                        cv2.putText(debug_frame, f"M{mi} ({mx},{my})",
                                    (mx + 18, my - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(debug_frame,
                                f"Calibrated {len(markers)} markers from diff image",
                                (10, debug_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if self._debug_mode:
                        _base_d = os.path.dirname(os.path.abspath(__file__))
                        cv2.imwrite(os.path.join(_base_d, "calibration_debug.png"), debug_frame)
                    # Next phase: project 4 corner dots to compute camera→projector homography
                    self._calib_phase = "proj_scan"
                    self._calib_frame_count = 0
                else:
                    self.logger.warning(
                        "Calibration failed: found %d markers (need %d). Restarting...",
                        len(markers) if markers else 0, self.expected_marker_count
                    )
                    self._calib_phase = "dark"
                    self._calib_frame_count = 0
                    self._calib_dark_ref = None
                    self._calib_illum_ref = None
                    self._blob_history = []
                continue

            # Phase 4: PROJ_SCAN - project 4 corner dots, detect in camera, compute homography
            if not self._calibrated and self._calib_phase == "proj_scan":
                self._calib_frame_count += 1
                proj_w, proj_h = self._proj_resolution
                inset = 0.05
                # Known projector corner positions (10% inset from edges)
                proj_corners = [
                    (int(proj_w * inset), int(proj_h * inset)),              # TL
                    (int(proj_w * (1 - inset)), int(proj_h * inset)),        # TR
                    (int(proj_w * (1 - inset)), int(proj_h * (1 - inset))),  # BR
                    (int(proj_w * inset), int(proj_h * (1 - inset))),        # BL
                ]
                # Draw 4 large bright dots on projector output (scaled to camera buffer size)
                projector_output.fill(0)
                for (px, py) in proj_corners:
                    bx = int(px * w / proj_w)
                    by = int(py * h / proj_h)
                    cv2.circle(projector_output, (bx, by), 30, (255, 255, 255), -1)

                # At frame 30, detect the dots in camera and compute homography
                if self._calib_frame_count == 30:
                    gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                    if self._calib_dark_ref is not None:
                        diff = np.clip(
                            gray.astype(np.float32) - self._calib_dark_ref.astype(np.float32),
                            0, 255
                        ).astype(np.uint8)
                    else:
                        diff = gray
                    diff_blur = cv2.GaussianBlur(diff, (7, 7), 0)
                    diff_max = float(diff_blur.max())
                    thresh_val = max(25, diff_max * 0.3)
                    _, binary = cv2.threshold(diff_blur, int(thresh_val), 255, cv2.THRESH_BINARY)
                    kk = np.ones((5, 5), np.uint8)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kk)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kk)

                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    blobs = []
                    for cnt in contours:
                        M = cv2.moments(cnt)
                        if M["m00"] < 15:
                            continue
                        cx_b = M["m10"] / M["m00"]
                        cy_b = M["m01"] / M["m00"]
                        area = cv2.contourArea(cnt)
                        blobs.append((area, cx_b, cy_b))
                    blobs.sort(key=lambda x: -x[0])
                    self.logger.info(
                        "Proj scan: found %d blobs (need 4): %s",
                        len(blobs),
                        [(int(b[1]), int(b[2]), int(b[0])) for b in blobs[:8]]
                    )

                    if len(blobs) >= 4:
                        cam_pts = np.float32([(b[1], b[2]) for b in blobs[:4]])
                        # Order as TL, TR, BR, BL using sum/diff heuristic
                        s = cam_pts.sum(axis=1)
                        d = cam_pts[:, 0] - cam_pts[:, 1]
                        cam_ordered = np.float32([
                            cam_pts[np.argmin(s)],  # TL: smallest x+y
                            cam_pts[np.argmax(d)],  # TR: largest x-y (large x, small y)
                            cam_pts[np.argmax(s)],  # BR: largest x+y
                            cam_pts[np.argmin(d)],  # BL: smallest x-y (small x, large y)
                        ])
                        proj_pts = np.float32(proj_corners)
                        H, _ = cv2.findHomography(cam_ordered, proj_pts)
                        if H is not None:
                            self._cam_to_proj_H = H
                            self.logger.info(
                                "Camera→Projector homography computed. Cam corners: %s → Proj corners: %s",
                                cam_ordered.tolist(), proj_pts.tolist()
                            )
                            # Save debug image
                            _base_ps = os.path.dirname(os.path.abspath(__file__))
                            dbg = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
                            labels = ["TL", "TR", "BR", "BL"]
                            for i_d, (cx_d, cy_d) in enumerate(cam_ordered):
                                cv2.circle(dbg, (int(cx_d), int(cy_d)), 10, (0, 0, 255), -1)
                                cv2.putText(dbg, labels[i_d], (int(cx_d) + 12, int(cy_d)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            if self._debug_mode:
                                cv2.imwrite(os.path.join(_base_ps, "proj_scan_debug.png"), dbg)
                        else:
                            self.logger.warning("findHomography returned None")
                    else:
                        self.logger.warning(
                            "Proj scan: only %d blobs, using projector rect fallback",
                            len(blobs)
                        )

                    # Fallback: if dot-based homography failed, use _proj_camera_rect
                    if self._cam_to_proj_H is None and self._proj_camera_rect is not None:
                        rx, ry, rw, rh = self._proj_camera_rect
                        # Camera rect maps to full projector output
                        cam_rect = np.float32([
                            [rx, ry], [rx + rw, ry],
                            [rx + rw, ry + rh], [rx, ry + rh]
                        ])
                        proj_full = np.float32([
                            [0, 0], [proj_w, 0],
                            [proj_w, proj_h], [0, proj_h]
                        ])
                        H_fallback, _ = cv2.findHomography(cam_rect, proj_full)
                        if H_fallback is not None:
                            self._cam_to_proj_H = H_fallback
                            self.logger.info(
                                "Fallback homography from proj rect: cam(%d,%d %dx%d) → proj(%dx%d)",
                                rx, ry, rw, rh, proj_w, proj_h
                            )

                    # Always save debug image so we can see what the camera saw
                    _base_ps2 = os.path.dirname(os.path.abspath(__file__))
                    dbg2 = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
                    cv2.putText(dbg2, f"Proj scan: {len(blobs)} blobs found (need 4), thresh={int(thresh_val)}, max={int(diff_max)}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    for bi, (ba, bx_d, by_d) in enumerate(blobs[:8]):
                        cv2.circle(dbg2, (int(bx_d), int(by_d)), 8, (0, 255, 0), 2)
                        cv2.putText(dbg2, f"#{bi} a={int(ba)}", (int(bx_d) + 10, int(by_d)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                    if self._debug_mode:
                        cv2.imwrite(os.path.join(_base_ps2, "proj_scan_all_blobs.png"), dbg2)
                        cv2.imwrite(os.path.join(_base_ps2, "proj_scan_binary.png"), binary)

                    # Finalize calibration (with or without homography)
                    markers = self._pending_markers
                    self._calibrated = True
                    self.logger.info(
                        "Calibration complete. %d markers, homography=%s",
                        len(markers), "YES" if self._cam_to_proj_H is not None else "NO"
                    )
                    self.markers_calibrated.emit(markers)

                # Status overlay and frame emission
                cv2.putText(main_frame, f"CALIBRATING - Projector scan {self._calib_frame_count}/30",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                qt_main = QImage(
                    cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.frame_ready.emit(qt_main)
                qt_proj = QImage(
                    cv2.cvtColor(projector_output, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.projector_frame_ready.emit(qt_proj)
                self.trackers_detected.emit(0)
                continue

            # Post-calibration: projector outputs composited masks (handled below)

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

            # Draw all raw detections as small green circles for debug
            for point in all_detected_points:
                cv2.circle(main_frame, point, 4, (0, 255, 0), 1)

            # Draw matched/tracked points as larger filled red circles
            for point in tracked_points:
                cv2.circle(main_frame, point, 8, (0, 0, 255), -1)

            # Debug overlay: detection count
            cv2.putText(main_frame, f"Det:{len(all_detected_points)} Trk:{len(tracked_points)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if self.show_mask_overlays:
                for mask in self.masks:
                    if not mask.source_points:
                        continue
                    pts = np.array(mask.source_points, dtype=np.int32).reshape((-1, 1, 2))
                    if len(pts) >= 3:
                        # Color-code: green for guitar, yellow for background
                        color = (0, 255, 0) if "guitar" in mask.name.lower() else (0, 255, 255)
                        thickness = 3 if "guitar" in mask.name.lower() else 1
                        cv2.polylines(main_frame, [pts], True, color, thickness)
                        # Label the mask
                        cx = int(np.mean([p[0] for p in mask.source_points]))
                        cy = int(np.mean([p[1] for p in mask.source_points]))
                        cv2.putText(main_frame, mask.name, (cx - 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            t0 = time.perf_counter()
            for i, mask in enumerate(self.masks):
                if not self._calibrated:
                    break  # skip video compositing during calibration
                if self.active_cue_index >= 0 and i != self.active_cue_index:
                    continue
                if not mask.source_points:
                    continue

                dst_pts = np.float32(mask.source_points)
                if len(dst_pts) < 3:
                    continue

                # --- Debug solid-color mode: fill masks with distinct colors ---
                if self.debug_solid_colors:
                    if "guitar" in mask.name.lower():
                        fill_color = (0, 0, 255)  # bright red (BGR)
                    elif "background" in mask.name.lower():
                        fill_color = (255, 100, 0)  # blue-ish (BGR)
                    else:
                        fill_color = (0, 200, 0)  # green
                    mask_image = self._mask_buffer
                    mask_image.fill(0)
                    cv2.fillPoly(mask_image, [np.int32(dst_pts)], (255, 255, 255))
                    # Create solid color frame
                    solid = np.full_like(projector_output, fill_color, dtype=np.uint8)
                    projector_output[:] = cv2.bitwise_and(
                        projector_output, cv2.bitwise_not(mask_image)
                    )
                    projector_output[:] = cv2.add(
                        projector_output, cv2.bitwise_and(solid, mask_image)
                    )
                    # Draw outline and label
                    cv2.polylines(projector_output, [np.int32(dst_pts)], True, (255, 255, 255), 2)
                    cx_label = int(np.mean([p[0] for p in mask.source_points]))
                    cy_label = int(np.mean([p[1] for p in mask.source_points]))
                    cv2.putText(projector_output, mask.name, (cx_label - 30, cy_label),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    continue

                cue_path = mask.get_active_video_path() if hasattr(mask, "get_active_video_path") else mask.video_path

                if not cue_path:
                    if not self.show_mask_overlays:
                        continue
                    overlay_color = (30, 30, 30) if mask.name.lower() == "background" else (0, 120, 255)
                    cv2.fillPoly(projector_output, [np.int32(dst_pts)], overlay_color)
                    cv2.polylines(projector_output, [np.int32(dst_pts)], True, (255, 255, 255), 2)
                    continue

                if mask.type == "dynamic":
                    if mask.linked_marker_count != len(tracked_points):
                        continue

                    src_pts = self._get_cached_source_points(mask)
                    dst_markers = self._calculate_destination_points(tracked_points)
                    anchor_pts = np.float32(getattr(mask, "marker_anchor_points", []) or [])

                    if len(anchor_pts) == len(dst_markers) and len(anchor_pts) >= 4:
                        marker_matrix = cv2.getPerspectiveTransform(anchor_pts[:4], np.float32(dst_markers[:4]))
                        if marker_matrix is None:
                            continue
                        dst_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), marker_matrix).reshape(-1, 2)
                    else:
                        if len(tracked_points) != len(mask.source_points):
                            continue
                        dst_pts = dst_markers
                else:
                    # Use bounding rect of polygon for perspective transform
                    all_x = [p[0] for p in mask.source_points]
                    all_y = [p[1] for p in mask.source_points]
                    bx1, by1 = min(all_x), min(all_y)
                    bx2, by2 = max(all_x), max(all_y)
                    dst_rect = np.float32([
                        [bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]
                    ])

                if cue_path not in self.video_captures:
                    cap_new = cv2.VideoCapture(cue_path)
                    if not cap_new.isOpened():
                        self.logger.error("Could not open cue video: %s", cue_path)
                        cap_new.release()
                        continue
                    self.video_captures[cue_path] = cap_new

                cap_cue = self.video_captures[cue_path]
                ret_cue, frame_cue = cap_cue.read()

                if not ret_cue:
                    cap_cue.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_cue, frame_cue = cap_cue.read()

                if not ret_cue:
                    continue

                if mask.type != "dynamic":
                    fh, fw = frame_cue.shape[:2]
                    src_pts = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_rect)
                else:
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
                qt_image_still = QImage(
                    cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
                self.still_frame_ready.emit(qt_image_still)
                self.logger.info("still_frame_ready emitted (%dx%d)", w, h)
                self._capture_still_frame_flag = False

            qt_image_main = QImage(
                cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB).tobytes(),
                w, h, w * 3, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image_main)

            t0 = time.perf_counter()
            # Apply camera→projector homography if available.
            # This warps the composited output (in camera pixel space) into
            # projector pixel space so content aligns with the physical scene.
            if self._cam_to_proj_H is not None:
                proj_w, proj_h = self._proj_resolution
                warped = cv2.warpPerspective(projector_output, self._cam_to_proj_H, (proj_w, proj_h))
                qt_image_proj = QImage(
                    cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).tobytes(),
                    proj_w, proj_h, proj_w * 3, QImage.Format_RGB888)
            else:
                qt_image_proj = QImage(
                    cv2.cvtColor(projector_output, cv2.COLOR_BGR2RGB).tobytes(),
                    w, h, w * 3, QImage.Format_RGB888)
            self.projector_frame_ready.emit(qt_image_proj)
            projector_ms = (time.perf_counter() - t0) * 1000.0

            frame_time_ms = (time.perf_counter() - frame_start) * 1000.0
            self._recent_frame_times.append(frame_time_ms)

            now = time.perf_counter()
            if now - self._last_debug_emit > 2.0:
                thresh_info = getattr(self, '_auto_thresh_info', '')
                self.logger.info(
                    "frame %.1fms | detected=%d tracked=%d masks=%d threshold=%s:%d | %s",
                    frame_time_ms,
                    len(all_detected_points),
                    len(tracked_points),
                    len(self.masks),
                    self.threshold_mode,
                    self.ir_threshold,
                    thresh_info,
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
