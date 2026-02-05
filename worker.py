
import cv2
import numpy as np
import time
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker
from PyQt5.QtGui import QImage
from itertools import combinations, permutations
from scipy.interpolate import interp1d, RBFInterpolator
from utils import resource_path
from mask import Mask
from sls_utils import generate_gray_code_patterns, decode_gray_code

class VideoPlayer(QThread):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._running = True
        self.latest_frame = None
        self.frame_id = 0
        self.frame_buffer = []
        self.max_buffer_size = 128 # Approx 4s buffer at 30fps for maximum smoothness
        self.mutex = QMutex()
        self.is_image = video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

        if self.is_image:
            self.cap = None
            self.fps = 1.0
            img = cv2.imread(video_path)
            if img is not None:
                self.latest_frame = img
                self.frame_id = 1
        else:
            # Try to use FFmpeg backend for better hardware support
            self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(video_path)

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps > 240: self.fps = 30.0

        self.playback_speed = 1.0

    def run(self):
        try:
            if self.is_image:
                while self._running:
                    time.sleep(1.0) # Just keep the thread alive
                return

            while self._running:
                # Use RAM to buffer frames for maximum smoothness
                buffer_count = 0
                with QMutexLocker(self.mutex):
                    buffer_count = len(self.frame_buffer)

                if buffer_count < self.max_buffer_size:
                    if self.cap is None or not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.video_path)
                        if not self.cap.isOpened():
                            time.sleep(1.0)
                            continue

                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            self.cap.release()
                            self.cap = cv2.VideoCapture(self.video_path)
                            ret, frame = self.cap.read()

                    if ret:
                        with QMutexLocker(self.mutex):
                            self.frame_buffer.append(frame)

                # Dynamic sleep to keep buffer full but not peg CPU
                sleep_time = 0.005 if buffer_count < self.max_buffer_size // 2 else 0.02
                time.sleep(sleep_time)

        except Exception as e:
            print(f"VideoPlayer Critical Error ({self.video_path}): {e}")

    def get_frame(self):
        with QMutexLocker(self.mutex):
            if self.frame_buffer:
                self.latest_frame = self.frame_buffer.pop(0)
                self.frame_id += 1

            if self.latest_frame is None:
                return None, 0
            return self.latest_frame, self.frame_id # Return reference to avoid copy overhead (will be copied/UMatted in Worker)

    def restart(self):
        if not self.is_image and self.cap is not None:
            with QMutexLocker(self.mutex):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def stop(self):
        self._running = False
        self.wait()
        if self.cap:
            self.cap.release()

class TrackingThread(QThread):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        main_cap = None
        target_fps = 20
        frame_time = 1.0 / target_fps

        while self._running:
            try:
                start_time = time.time()

                if self.worker._camera_changed or self.worker.requested_camera_res != self.worker._current_camera_res:
                    if self.worker._camera_changed:
                        if main_cap: main_cap.release()
                        # Optimization: Use DSHOW and specify resolution IMMEDIATELY to avoid slow opening
                        main_cap = cv2.VideoCapture(self.worker.video_source, cv2.CAP_DSHOW)
                        if not main_cap.isOpened():
                            main_cap = cv2.VideoCapture(self.worker.video_source)

                        if main_cap.isOpened():
                            # Optimization: set MJPG to improve frame rates at high resolution
                            main_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                            w_req, h_req = self.worker.requested_camera_res
                            main_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_req)
                            main_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_req)

                            # Buffering: Minimize latency
                            main_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                        self.worker._camera_changed = False

                    if main_cap and main_cap.isOpened():

                        # Disable Auto Exposure and Auto Gain to prevent brightness swings from projector light
                        # Note: behavior is backend dependent. 1/0.25 usually means manual.
                        main_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                        main_cap.set(cv2.CAP_PROP_GAIN, 0)

                        # Read back actual resolution
                        act_w = int(main_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        act_h = int(main_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.worker._current_camera_res = (act_w, act_h)

                        # Clear frame buffer to avoid resolution mismatch
                        with QMutexLocker(self.worker.latest_main_frame_mutex):
                             self.worker.latest_main_frame = None

                        # Update requested res to match reality to prevent infinite re-initialization
                        # if the camera doesn't support the requested resolution.
                        self.worker.requested_camera_res = (act_w, act_h)

                        self.worker.camera_matrix = None # Reset estimation
                        print(f"Camera Resolution Switched to: {act_w}x{act_h}")
                    else:
                        if self.worker._camera_changed:
                            self.worker.camera_error.emit(self.worker.video_source)
                        main_cap = None

                if main_cap is None:
                    QThread.msleep(1000)
                    continue

                ret, main_frame = main_cap.read()
                if not ret:
                    self.worker.camera_error.emit(self.worker.video_source)
                    main_cap.release()
                    main_cap = cv2.VideoCapture(self.worker.video_source)
                    QThread.msleep(1000)
                    continue

                h_cam, w_cam = main_frame.shape[:2]

                # IR Tracking
                # Optimization: Skip tracking during setup/calibration to save CPU
                if not (self.worker._run_sls_flag or self.worker._run_calibration_flag or self.worker._run_boundary_detection_flag):
                    tracked_points = self.worker.get_tracked_points(main_frame)
                else:
                    tracked_points = []

                with QMutexLocker(self.worker.tracking_mutex):
                    self.worker.last_tracked_points_internal = tracked_points
                    self.worker.last_homography_internal = self.worker.last_homography
                    self.worker.confidence_internal = self.worker.confidence
                    self.worker.tracking_frame_count += 1

                if tracked_points is None: tracked_points = []
                self.worker.trackers_detected.emit(len(tracked_points))
                self.worker.trackers_ready.emit(tracked_points)

                if tracked_points and len(tracked_points) >= 2:
                    current_dist = np.linalg.norm(np.array(tracked_points[0]) - np.array(tracked_points[1]))
                    if self.worker._calibrate_depth_flag and current_dist > 0.001:
                        self.worker.baseline_distance = current_dist
                        self.worker._calibrate_depth_flag = False

                    if self.worker.baseline_distance > 0.001:
                        self.worker.proximity_val = current_dist / self.worker.baseline_distance
                    else:
                        self.worker.proximity_val = 1.0

                # HUD Data preparation (camera side)
                with QMutexLocker(self.worker.latest_main_frame_mutex):
                    self.worker.latest_main_frame = main_frame.copy()
                    self.worker.latest_main_frame_id += 1
                    self.worker.latest_tracked_points_for_ui = tracked_points

                # Handle Calibration Flags (some need to run in camera thread)
                if self.worker._capture_still_frame_flag:
                    # Capture immediately to avoid "Waiting for Camera" hang
                    rgb = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                    # Perform specialized high-res detection for the still frame
                    still_dots, rejected_dots = self.worker.get_tracked_points(main_frame, force_full=True, return_rejected=True)
                    self.worker.still_frame_ready.emit(QImage(rgb.data, w_cam, h_cam, w_cam * 3, QImage.Format_RGB888).copy(), still_dots, rejected_dots)
                    self.worker._capture_still_frame_flag = False

                elapsed = time.time() - start_time
                sleep_time = max(1, int((frame_time - elapsed) * 1000))
                QThread.msleep(sleep_time)
            except Exception as e:
                print(f"Critical Error in Tracking Thread: {e}")
                QThread.msleep(500)

        if main_cap:
            main_cap.release()

class Worker(QObject):
    frame_ready = pyqtSignal(QImage)
    projector_frame_ready = pyqtSignal(QImage)
    still_frame_ready = pyqtSignal(QImage, list, list) # rgb, detected, rejected
    trackers_detected = pyqtSignal(int)
    trackers_ready = pyqtSignal(list)
    camera_error = pyqtSignal(int)
    system_warning = pyqtSignal(str)
    status_update = pyqtSignal(str)
    calibration_complete = pyqtSignal(bool)
    boundary_detected = pyqtSignal(list) # List of points

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.video_source = 0
        self.projector_width = 1280
        self.projector_height = 720
        self.warp_grid_res = 3
        self.warp_points = []
        for y in [0.0, 0.5, 1.0]:
            for x in [0.0, 0.5, 1.0]:
                self.warp_points.append([x, y])
        self.map_x = None
        self.map_y = None
        self._warp_map_dirty = True
        self._warp_is_identity = True
        self.masks = []
        self.mask_mutex = QMutex()
        self.video_players = {}
        self.player_mutex = QMutex()
        self.ir_threshold = 200
        self.auto_threshold = False
        self._camera_changed = True
        self.requested_camera_res = (9999, 9999) # Default to max FOV for stability
        self._current_camera_res = (0, 0)
        self.calibration_camera_res = None
        self.baseline_distance = 0
        self.depth_sensitivity = 1.0
        self._calibrate_depth_flag = False
        self._capture_still_frame_flag = False
        self.marker_config = None
        self.marker_config_base = None
        self.marker_capture_res = None
        self.bpm = 120.0
        self.last_tracked_points = None
        self.last_tracked_points_internal = []
        self.last_homography = None
        self.last_homography_internal = None
        self.confidence_internal = 0.0
        self.marker_fingerprint = []
        self.roi_padding = 100 # Increased for high-res FOV
        self.tracking_mutex = QMutex()

        # Smoothing and Confidence
        self.kalman_filters = []
        self.smoothed_points = None
        self.smoothing_factor = 0.8 # Increased for better stability/smoothing as requested
        self.history_points = []
        self.history_len = 20 # Increased for maximum stability as requested
        self.confidence = 0.0
        self.confidence_gain = 0.25
        self.confidence_decay = 0.05

        # Crossfade management
        self.fades = {}
        self.fade_duration = 1.0

        # FX Buffers
        self.trail_buffers = {} # mask_id -> last_frame
        self.noise_offset = 0

        # Particle System
        self.particles = []
        self.particle_preset = 'none' # 'none', 'dust', 'rain', 'trail'
        self.particle_max_count = 100

        # Proximity Modulation
        self.proximity_mode = 'none' # 'none', 'kaleidoscope', 'glitch', 'rgb_shift'
        self.proximity_val = 1.0

        # Audio Reactivity
        self.audio_bands = [0, 0, 0] # bass, mid, high
        self.audio_reactive_target = 'none' # legacy trigger target
        self.audio_gain = 1.0
        self.audio_param_mappings = {} # fx_name -> band_index (0,1,2)

        # Beat Automation
        self.auto_pilot = False
        self.beat_counter = 0
        self.auto_pilot_interval = 8 # beats

        # Stats for HUD
        self.fps = 0
        self.tracking_fps = 0
        self.frame_count = 0
        self.tracking_frame_count = 0
        self.last_stats_time = time.time()
        self.show_hud = True
        self.blackout_active = False
        self.tracking_freeze_enabled = False

        # Safety Mode
        self.safety_mode_enabled = True
        self.fallback_generator = 'radial'
        self.pnp_enabled = False
        self.occlusion_enabled = False
        self.back_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.occlusion_mask = None

        # Calibration/Alignment Mode
        self.show_camera_on_projector = False
        self.show_calibration_pattern = False
        self.show_calibration_verify = False
        self.h_c2p = None # Camera to Projector homography
        self.rbf_x = None # RBF for X mapping
        self.rbf_y = None # RBF for Y mapping
        self.sls_lut_x = None # Dense LUT from SLS
        self.sls_lut_y = None # Dense LUT from SLS
        self.sls_valid_mask = None
        self._run_calibration_flag = False
        self._calib_frames_captured = 0
        self._calib_corners_sum = None
        self._calib_total_frames = 15

        # Structured Light Scanning (SLS)
        self._run_sls_flag = False
        self._sls_step = 0
        self._sls_patterns_x = []
        self._sls_patterns_y = []
        self._sls_captures_x = []
        self._sls_captures_y = []
        self._sls_wait_frames = 12
        self._sls_curr_wait = 0

        # Projector Boundary Detection
        self._run_boundary_detection_flag = False
        self._boundary_step = 0
        self._boundary_captures = []
        self.projector_boundary = None # Mask points (camera space)

        # Splash Mode
        self.show_splash = False
        self.splash_player = None

        # Master FX
        self.master_active_fx = []
        self.master_tint_color = (255, 255, 255)
        self.master_brightness = 0 # -100 to 100
        self.master_contrast = 0   # -100 to 100
        self.master_saturation = 100 # 0 to 200
        self.master_grain = 0 # 0 to 100
        self.master_bloom = 0 # 0 to 100
        self.master_fader = 1.0 # 0.0 to 1.0

        # Crossfade states for masks
        self.mask_fade_levels = {} # mask_id -> current_fade (0.0 to 1.0)

        # Performance Tuning
        self.render_width = 1280
        self.render_height = 720
        self.render_scale = 0.7 # Default scale factor
        self.throttle_level = 0.0 # 0.0 to 1.0 (degrade quality)

        # Caching
        self.static_warp_cache = {} # mask_id -> (video_frame_id, warped_frame, mask_img)
        self.video_umat_cache = {}  # (path, frame_id) -> UMat

        # Reusable Buffers
        self.projector_buffer = None
        self.mask_buffer = None
        self.u_projector_output = None
        self.u_mask_buffer = None
        self.latest_main_frame = None
        self.latest_main_frame_mutex = QMutex()
        self.latest_main_frame_id = 0
        self.last_projected_frame_for_masking = None
        self.last_projected_frame_mutex = QMutex()
        self._last_captured_frame_id = -1
        self.latest_tracked_points_for_ui = []
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        self.last_raw_detections = []
        self.cached_plasma_grid = None
        self.cached_nebula_grid = None
        self.generator_buffer = None
        self.blend_temp1 = None
        self.blend_temp2 = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.tracking_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

        # Global OpenCV Performance Optimizations & GPU Acceleration
        try:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(4)
            # Enable OpenCL (GPU acceleration)
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.haveOpenCL():
                self.status_update.emit(f"GPU Acceleration Enabled: {cv2.ocl.useOpenCL()}")
                print(f"[DEBUG] OpenCL Available. Using GPU: {cv2.ocl.useOpenCL()}")
            else:
                print("[DEBUG] OpenCL NOT Available in this environment.")
        except Exception as e:
            print(f"[DEBUG] Failed to init GPU acceleration: {e}")

        # Start Tracking Thread
        self.tracking_thread = TrackingThread(self)
        self.tracking_thread.start()

    def boost_contrast(self, frame):
        if frame is None: return None
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return self.clahe.apply(gray)
        else:
            return self.clahe.apply(frame)

    def normalize_intensity(self, gray_frame):
        """Simple Min-Max normalization to improve mapper robustness without CLAHE artifacts."""
        if gray_frame is None: return None
        # Using a slight clipping to ignore dead/hot pixels
        p5, p95 = np.percentile(gray_frame, (5, 95))
        if p95 > p5:
            res = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return res
        return gray_frame

    def init_kalman(self, count):
        self.kalman_filters = []
        for _ in range(count):
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            # Maximum stability: trust the model heavily to avoid jitter
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.00005
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2
            kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
            self.kalman_filters.append(kf)

    def set_marker_points(self, points):
        """Sets the reference marker configuration. Expects a list of normalized (0.0-1.0) coordinates."""
        if points is None:
            self.clear_marker_config()
            return

        with QMutexLocker(self.tracking_mutex):
            # Reset tracking history when markers change
            self.history_points = []
            self.smoothed_points = None
            self.last_homography = None
            self.confidence = 0.0

            new_config = []
            for p in points:
                if hasattr(p, 'x'): # QPoint or QPointF (normalized)
                    new_config.append((p.x(), p.y()))
                else: # Tuple or list (normalized)
                    new_config.append((p[0], p[1]))

            self.marker_config = new_config
            if self.marker_config and len(self.marker_config) > 1:
                self.init_kalman(len(self.marker_config))
                distances = []
                for p1, p2 in combinations(self.marker_config, 2):
                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    distances.append(dist)
                self.marker_fingerprint = sorted(distances)
            else:
                self.marker_fingerprint = []
        self.last_tracked_points = None
        self.smoothed_points = None

    def clear_marker_config(self):
        with QMutexLocker(self.tracking_mutex):
            self.marker_config = None
            self.marker_config_base = None
            self.marker_capture_res = None
            self.marker_fingerprint = []
            self.last_tracked_points = None
            self.smoothed_points = None
            self.last_homography = None
            self.confidence = 0.0

    def set_auto_threshold(self, enabled):
        self.auto_threshold = enabled

    def set_smoothing(self, value):
        self.smoothing_factor = value

    def capture_still_frame(self):
        # Set to max FOV for capture
        self.requested_camera_res = (9999, 9999)
        self._capture_still_frame_flag = True

    def calibrate_depth(self):
        self._calibrate_depth_flag = True

    def set_h_c2p(self, matrix_list, cam_res=None):
        if matrix_list is None:
            self.h_c2p = None
            self.rbf_x = None
            self.rbf_y = None
        else:
            if cam_res:
                self.calibration_camera_res = cam_res
            else:
                self.calibration_camera_res = self._current_camera_res

            # Handle list of points for RBF or matrix for Homography
            if isinstance(matrix_list, list) and len(matrix_list) > 4 and isinstance(matrix_list[0], list):
                # Probably a point list [(cam_x, cam_y, proj_x, proj_y), ...]
                self.init_rbf_from_points(matrix_list)
            else:
                self.h_c2p = np.array(matrix_list, dtype=np.float32)

    def init_rbf_from_points(self, points):
        pts = np.array(points, dtype=np.float32)
        cam_pts = pts[:, :2]
        proj_x = pts[:, 2]
        proj_y = pts[:, 3]

        # Using Thin Plate Spline kernel ('thin_plate_spline' or 'linear')
        self.rbf_x = RBFInterpolator(cam_pts, proj_x, kernel='thin_plate_spline', smoothing=0.1)
        self.rbf_y = RBFInterpolator(cam_pts, proj_y, kernel='thin_plate_spline', smoothing=0.1)
        # Store points for persistence
        self.calib_points = points

    def set_depth_sensitivity(self, value):
        self.depth_sensitivity = value

    def set_ir_threshold(self, value):
        self.ir_threshold = value

    def set_video_source(self, source):
        self.video_source = source
        self._camera_changed = True

    def set_bpm(self, bpm):
        self.bpm = bpm
        self.update_video_speeds()

    def restart_mask_video(self, video_path):
        if not video_path: return
        with QMutexLocker(self.player_mutex):
            if video_path in self.video_players:
                self.video_players[video_path].restart()

    def update_video_speeds(self):
        with QMutexLocker(self.player_mutex):
            for mask in self.masks:
                if mask.video_path in self.video_players:
                    if mask.video_bpm > 0:
                        speed = self.bpm / mask.video_bpm
                        self.video_players[mask.video_path].playback_speed = speed
                    else:
                        self.video_players[mask.video_path].playback_speed = 1.0

    def trigger_beat(self):
        self.beat_counter += 1
        if self.auto_pilot and (self.beat_counter % self.auto_pilot_interval == 0):
            self.run_auto_pilot()

    def run_auto_pilot(self):
        styles = ['none', 'acid', 'noir', 'retro']
        self.set_style(np.random.choice(styles))

    def set_particle_preset(self, preset):
        self.particle_preset = preset
        self.particles = []

    def set_proximity_mode(self, mode):
        self.proximity_mode = mode

    def set_audio_bands(self, bass, mid, high):
        self.audio_bands = [bass * self.audio_gain, mid * self.audio_gain, high * self.audio_gain]

    def set_audio_target(self, target):
        self.audio_reactive_target = target

    def set_audio_gain(self, gain):
        self.audio_gain = gain

    def set_audio_param_mapping(self, fx_name, band_index):
        if band_index == -1:
            if fx_name in self.audio_param_mappings:
                del self.audio_param_mappings[fx_name]
        else:
            self.audio_param_mappings[fx_name] = band_index

    def set_style(self, style_name):
        # Clears all FX and sets a specific preset style
        for mask in self.masks:
            mask.active_fx = []
            if style_name == 'acid':
                mask.active_fx = ['kaleidoscope', 'hue_cycle', 'glitch']
            elif style_name == 'noir':
                mask.active_fx = ['edges', 'trails', 'invert']
            elif style_name == 'retro':
                mask.active_fx = ['rgb_shift', 'trails', 'tint']
                mask.tint_color = (255, 0, 255) # Magenta

    def switch_video(self, tag, video_path):
        for mask in self.masks:
            if mask.tag == tag:
                if mask.video_path and mask.video_path != video_path:
                    self.fades[tag] = {
                        'prev_path': mask.video_path,
                        'start_time': time.time()
                    }
                mask.video_path = video_path
        self.update_video_speeds()
        self.cleanup_resources()

    def switch_cue(self, tag, index):
        for mask in self.masks:
            if mask.tag == tag:
                if hasattr(mask, 'playlist') and 0 <= index < len(mask.playlist):
                    mask.playlist_index = index
                    self.switch_video(tag, mask.playlist[index])

    def toggle_mask(self, tag, visible):
        for mask in self.masks:
            if mask.tag == tag:
                mask.visible = visible

    def set_fx(self, tag, fx_name, enabled):
        if tag == 'master':
            if enabled and fx_name not in self.master_active_fx:
                self.master_active_fx.append(fx_name)
            elif not enabled and fx_name in self.master_active_fx:
                self.master_active_fx.remove(fx_name)
            return

        for mask in self.masks:
            if mask.tag == tag:
                if enabled and fx_name not in mask.active_fx:
                    mask.active_fx.append(fx_name)
                elif not enabled and fx_name in mask.active_fx:
                    mask.active_fx.remove(fx_name)

    def get_lfo_value(self, mask):
        speed = mask.fx_params.get('lfo_speed', 1.0)
        t = time.time() * (self.bpm / 60.0) * speed
        shape = mask.fx_params.get('lfo_shape', 'sine')

        if shape == 'square':
            return 1.0 if (np.sin(2 * np.pi * t) >= 0) else 0.0
        elif shape == 'triangle':
            return 2 * np.abs(2 * (t - np.floor(t + 0.5)))
        elif shape == 'sawtooth':
            return 2 * (t - np.floor(t))
        else: # sine
            return (np.sin(2 * np.pi * t) + 1) / 2

    def apply_fx(self, frame, mask, live_only=False, h=None, w=None):
        mask_id = id(mask)
        lfo_val = 1.0
        lfo_enabled = mask.fx_params.get('lfo_enabled')
        if lfo_enabled:
            lfo_val = self.get_lfo_value(mask)

        if not mask.active_fx and mask.design_overlay == 'none':
            return frame

        # Performance: skip FX if heavily throttled and not essential
        if self.throttle_level > 0.9 and not live_only:
             return frame

        # Performance: Use GPU for FX (Maintain UMat if already on GPU)
        is_umat = isinstance(frame, cv2.UMat)
        u_frame = frame if is_umat else cv2.UMat(frame)

        # Get dimensions (works for both UMat and numpy)
        if h is None or w is None:
            if is_umat:
                h, w = u_frame.get().shape[:2]
            else:
                h, w = frame.shape[:2]
        res_scale = np.sqrt((w * h) / (1280 * 720))

        # 1. Transform/Mirror FX (GPU Accelerated)
        if not live_only:
            if 'mirror_h' in mask.active_fx:
                u_left = cv2.UMat(u_frame, (0, 0, w//2, h))
                u_right = cv2.flip(u_left, 1)
                u_right.copyTo(cv2.UMat(u_frame, (w//2, 0, w//2, h)))
            if 'mirror_v' in mask.active_fx:
                u_top = cv2.UMat(u_frame, (0, 0, w, h//2))
                u_bot = cv2.flip(u_top, 0)
                u_bot.copyTo(cv2.UMat(u_frame, (0, h//2, w, h//2)))
            if 'kaleidoscope' in mask.active_fx:
                u_quad = cv2.UMat(u_frame, (0, 0, w//2, h//2))
                cv2.flip(u_quad, 1).copyTo(cv2.UMat(u_frame, (w//2, 0, w//2, h//2)))
                cv2.flip(u_quad, 0).copyTo(cv2.UMat(u_frame, (0, h//2, w//2, h//2)))
                cv2.flip(u_quad, -1).copyTo(cv2.UMat(u_frame, (w//2, h//2, w//2, h//2)))

        # 2. Timing/Glitch FX
        if 'strobe' in mask.active_fx:
            trigger = False
            if self.audio_reactive_target == 'strobe':
                if self.audio_bands[0] > 0.6: trigger = True
            else:
                period = 60.0 / self.bpm
                if (time.time() % period) < (period / 2.0): trigger = True
            if trigger: u_frame = cv2.UMat(np.zeros_like(frame))

        if 'rgb_shift' in mask.active_fx:
            mod = 1.0
            if self.proximity_mode == 'rgb_shift': mod = self.proximity_val
            if 'rgb_shift' in self.audio_param_mappings:
                mod *= (self.audio_bands[self.audio_param_mappings['rgb_shift']] * 5)
            shift = int(10 * res_scale * mod * (lfo_val if mask.fx_params.get('lfo_target') == 'rgb_shift' else 1.0))
            if shift != 0:
                # RGB shift remains easier in CPU due to roll() but we can optimize by splitting
                b, g, r = cv2.split(u_frame)
                # Note: OpenCL doesn't have a direct 'roll', but we can use warpAffine for small shifts
                M_r = np.float32([[1, 0, -shift], [0, 1, 0]])
                M_b = np.float32([[1, 0, shift], [0, 1, 0]])
                r = cv2.warpAffine(r, M_r, (w, h))
                b = cv2.warpAffine(b, M_b, (w, h))
                u_frame = cv2.merge([b, g, r])

        if 'glitch' in mask.active_fx:
            mod = 1.0
            if self.proximity_mode == 'glitch': mod = self.proximity_val
            if self.audio_reactive_target == 'glitch': mod *= (self.audio_bands[0] * 2)

            for _ in range(int(3 * mod)):
                g_h = max(1, int(10 * res_scale))
                gy = np.random.randint(0, max(1, h - g_h))
                gsh = int(np.random.randint(-20, 20) * res_scale)
                if gsh == 0: continue

                # Performance: Use GPU ROI and warpAffine with BORDER_WRAP to simulate roll
                u_row = cv2.UMat(u_frame, (gy, gy + g_h), (0, w))
                M = np.float32([[1, 0, gsh], [0, 1, 0]])
                u_row_shifted = cv2.warpAffine(u_row, M, (w, g_h), borderMode=cv2.BORDER_WRAP)
                u_row_shifted.copyTo(u_row)

        if 'trails' in mask.active_fx:
            if mask_id in self.trail_buffers:
                u_trail, th, tw = self.trail_buffers[mask_id]
                # Check if dimensions match (in case of resize)
                if th == h and tw == w:
                    u_frame = cv2.addWeighted(u_frame, 0.4, u_trail, 0.6, 0)

            # Store copy on GPU to avoid download/upload
            u_copy = cv2.UMat(h, w, cv2.CV_8UC3)
            u_frame.copyTo(u_copy)
            self.trail_buffers[mask_id] = (u_copy, h, w)

        if 'feedback' in mask.active_fx:
            if mask_id in self.trail_buffers:
                # Use same buffer as trails if both active, otherwise it might be tuple or UMat
                data = self.trail_buffers[mask_id]
                if isinstance(data, tuple):
                    u_prev, ph, pw = data
                else: # Legacy/Fallback
                    u_prev, ph, pw = data, 0, 0

                if ph == h and pw == w:
                    M = cv2.getRotationMatrix2D((w//2, h//2), 1, 1.02)
                    u_prev_w = cv2.warpAffine(u_prev, M, (w, h))
                    u_frame = cv2.addWeighted(u_frame, 0.7, u_prev_w, 0.3, 0)

            u_copy = cv2.UMat(h, w, cv2.CV_8UC3)
            u_frame.copyTo(u_copy)
            self.trail_buffers[mask_id] = (u_copy, h, w)

        if 'hue_cycle' in mask.active_fx:
            u_hsv = cv2.cvtColor(u_frame, cv2.COLOR_BGR2HSV)
            shift = (time.time() * (self.bpm / 60.0) * 30) % 180
            if mask.fx_params.get('lfo_target') == 'hue': shift *= lfo_val
            h, s, v = cv2.split(u_hsv)
            h = cv2.add(h, shift) # Note: Wrap handling is internal in cv2.add for uint8/180
            u_hsv = cv2.merge([h, s, v])
            u_frame = cv2.cvtColor(u_hsv, cv2.COLOR_HSV2BGR)

        # 3. Static/Heavy FX (GPU Accelerated)
        if not live_only:
            if 'blur' in mask.active_fx:
                mod = 1.0
                if 'blur' in self.audio_param_mappings:
                    mod *= (self.audio_bands[self.audio_param_mappings['blur']] * 3)
                base_size = 15 * res_scale * (1.0 - self.throttle_level * 0.8)
                ksize = int(base_size * mod * (lfo_val if mask.fx_params.get('lfo_target') == 'blur' else 1.0))
                if ksize % 2 == 0: ksize += 1
                if ksize > 1:
                    if self.throttle_level > 0.5:
                        u_frame = cv2.blur(u_frame, (ksize, ksize))
                    else:
                        u_frame = cv2.GaussianBlur(u_frame, (ksize, ksize), 0)

            if 'invert' in mask.active_fx:
                u_frame = cv2.bitwise_not(u_frame)

            if 'edges' in mask.active_fx:
                u_gray = cv2.cvtColor(u_frame, cv2.COLOR_BGR2GRAY)
                u_edges = cv2.Canny(u_gray, 100, 200)
                u_frame = cv2.cvtColor(u_edges, cv2.COLOR_GRAY2BGR)

            if 'tint' in mask.active_fx:
                mod = 1.0
                if 'tint' in self.audio_param_mappings:
                    mod *= (self.audio_bands[self.audio_param_mappings['tint']] * 3)
                alpha = 0.3 * mod * (lfo_val if mask.fx_params.get('lfo_target') == 'tint' else 1.0)
                # Performance: cv2.rectangle on UMat is used as fill
                u_tint = cv2.UMat(h, w, cv2.CV_8UC3)
                cv2.rectangle(u_tint, (0, 0), (w, h), mask.tint_color, -1)
                u_frame = cv2.addWeighted(u_frame, 1.0 - alpha, u_tint, alpha, 0)

            if 'duotone' in mask.active_fx:
                u_gray = cv2.cvtColor(u_frame, cv2.COLOR_BGR2GRAY)
                comp = (255 - mask.tint_color[0], 255 - mask.tint_color[1], 255 - mask.tint_color[2])
                lut = np.zeros((256, 1, 3), dtype=np.uint8)
                for i in range(256):
                    a = i / 255.0
                    lut[i, 0, 0] = int(comp[0] * (1 - a) + mask.tint_color[0] * a)
                    lut[i, 0, 1] = int(comp[1] * (1 - a) + mask.tint_color[1] * a)
                    lut[i, 0, 2] = int(comp[2] * (1 - a) + mask.tint_color[2] * a)
                u_frame = cv2.LUT(cv2.merge([u_gray, u_gray, u_gray]), lut)

            if 'pixelate' in mask.active_fx:
                div = max(2, int(16 * res_scale))
                small_w, small_h = max(1, w // div), max(1, h // div)
                u_small = cv2.resize(u_frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                u_frame = cv2.resize(u_small, (w, h), interpolation=cv2.INTER_NEAREST)

            if 'chroma_aberration' in mask.active_fx:
                b, g, r = cv2.split(u_frame)
                M_b = np.float32([[1, 0, 0], [0, 1, 5]])
                M_r = np.float32([[1, 0, -5], [0, 1, 0]])
                b = cv2.warpAffine(b, M_b, (w, h))
                r = cv2.warpAffine(r, M_r, (w, h))
                u_frame = cv2.merge([b, g, r])

            if 'ooze' in mask.active_fx and self.throttle_level < 0.7:
                t = time.time()
                for x in range(0, w, 20):
                    length = int((np.sin(t + x * 0.1) * 0.5 + 0.5) * h)
                    cv2.line(u_frame, (x, 0), (x, length), (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.circle(u_frame, (x, length), 5, (0, 255, 100), -1)

            if 'matrix' in mask.active_fx:
                t = time.time()
                m_step = int(15 * res_scale)
                m_font = 0.4 * res_scale
                for x in range(0, w, m_step):
                    speed = 1.0 + (np.sin(x) * 0.5 + 0.5)
                    y = int((t * 200 * res_scale * speed) % (h + 100 * res_scale)) - int(100 * res_scale)
                    for i in range(10):
                        alpha = (10 - i) / 10.0
                        color = (0, int(255 * alpha), 0)
                        cv2.putText(u_frame, chr(np.random.randint(33, 126)), (x, int(y - i*15*res_scale)),
                                    cv2.FONT_HERSHEY_SIMPLEX, m_font, color, 1)

            if 'vhs' in mask.active_fx:
                jitter = int(np.random.randint(-5, 5) * res_scale)
                M_jitter = np.float32([[1, 0, jitter], [0, 1, 0]])
                u_frame = cv2.warpAffine(u_frame, M_jitter, (w, h))

                y_vhs = np.random.randint(0, h)
                v_h = max(1, int(2 * res_scale))
                actual_vh = min(v_h, h - y_vhs)
                if actual_vh > 0:
                    # ROI on GPU
                    u_vhs_line = cv2.UMat(u_frame, (0, y_vhs, w, actual_vh))
                    u_vhs_line_mod = cv2.add(u_vhs_line, (50, 50, 50, 0))
                    u_vhs_line_mod.copyTo(u_vhs_line)

                b, g, r = cv2.split(u_frame)
                v_shift = max(1, int(3 * res_scale))
                M_v_shift = np.float32([[1, 0, v_shift], [0, 1, 0]])
                r = cv2.warpAffine(r, M_v_shift, (w, h))
                u_frame = cv2.merge([b, g, r])

            if 'scanline' in mask.active_fx:
                t = time.time()
                y_pos = int((t * 100) % h)
                cv2.line(u_frame, (0, y_pos), (w, y_pos), (255, 255, 255), 1)
                # Note: scanline rows are better on CPU or with specialized kernel
                # But we can approximate with a scaling
                u_frame = cv2.convertScaleAbs(u_frame, alpha=0.9)

        return u_frame.get()

    def get_design_mask(self, design_name, h, w):
        if not hasattr(self, '_design_cache'): self._design_cache = {}
        cache_key = (design_name, h, w)
        if cache_key in self._design_cache:
            return self._design_cache[cache_key]

        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        if design_name == 'spiral':
            for i in range(0, 360 * 5, 2):
                r = i // 10
                x = int(center[0] + r * np.cos(np.radians(i)))
                y = int(center[1] + r * np.sin(np.radians(i)))
                if -100 < x < w + 100 and -100 < y < h + 100:
                    cv2.circle(mask, (x, y), 10 + i // 100, 255, -1)
        elif design_name == 'moon':
            cv2.circle(mask, center, h // 3, 255, -1)
            offset = h // 10
            cv2.circle(mask, (center[0] + offset, center[1] - offset), h // 3, 0, -1)
        elif design_name == 'mushroom':
            # Cap
            cv2.ellipse(mask, center, (w // 3, h // 4), 0, 180, 360, 255, -1)
            # Stem
            cv2.rectangle(mask, (center[0] - w // 10, center[1]), (center[0] + w // 10, center[1] + h // 3), 255, -1)
            # Spots
            cv2.circle(mask, (center[0], center[1] - h // 8), h // 20, 0, -1)
            cv2.circle(mask, (center[0] - w // 6, center[1] - h // 12), h // 25, 0, -1)
            cv2.circle(mask, (center[0] + w // 6, center[1] - h // 12), h // 25, 0, -1)
        elif design_name == 'star':
            pts = []
            for i in range(10):
                r = (h // 3) if i % 2 == 0 else (h // 6)
                theta = np.radians(i * 36)
                pts.append([center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)])
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        elif design_name == 'hexagon':
            pts = []
            for i in range(6):
                theta = np.radians(i * 60)
                pts.append([center[0] + (h // 3) * np.cos(theta), center[1] + (h // 3) * np.sin(theta)])
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        elif design_name == 'heart':
            for i in range(0, 360):
                t = np.radians(i)
                x = 16 * np.sin(t)**3
                y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
                px = int(center[0] + x * (h//50))
                py = int(center[1] + y * (h//50))
                if -50 < px < w + 50 and -50 < py < h + 50:
                    cv2.circle(mask, (px, py), h // 40, 255, -1)
        else:
            mask.fill(255)

        self._design_cache[cache_key] = mask
        return mask

    def get_tracked_points(self, frame, force_full=False, return_rejected=False):
        # Already called from TrackingThread, but let's ensure we use tracking_mutex for shared state
        if frame is None: return ([], []) if return_rejected else []

        # Defensive catch-all to prevent thread crashes
        try:
            res = self._get_tracked_points_internal(frame, force_full, return_rejected)
            if res is None:
                 return ([], []) if return_rejected else []
            return res
        except Exception as e:
            print(f"Error in tracking logic: {e}")
            return ([], []) if return_rejected else []

    def _get_tracked_points_internal(self, frame, force_full=False, return_rejected=False):
        # Already called from TrackingThread, but let's ensure we use tracking_mutex for shared state
        if frame is None: return ([], []) if return_rejected else []
        h, w = frame.shape[:2]

        # Performance: Use GPU (UMat) for image processing
        u_frame = cv2.UMat(frame)
        with QMutexLocker(self.tracking_mutex):
            marker_cfg = self.marker_config
            marker_fp = self.marker_fingerprint if self.marker_fingerprint is not None else []

        roi_x, roi_y, roi_w, roi_h = 0, 0, w, h
        # Scale padding with resolution for wide setups
        # Increased to 6% for better high-velocity neck tracking
        dynamic_padding = max(self.roi_padding, int(w * 0.06))

        if not force_full and self.last_tracked_points is not None:
            # pts are normalized
            pts = np.array(self.last_tracked_points)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            # Convert to pixels for ROI calculation
            roi_x = max(0, int(min_x * w - dynamic_padding))
            roi_y = max(0, int(min_y * h - dynamic_padding))
            roi_w = min(w - roi_x, int((max_x - min_x) * w + 2 * dynamic_padding))
            roi_h = min(h - roi_y, int((max_y - min_y) * h + 2 * dynamic_padding))

        u_roi = cv2.UMat(u_frame, (roi_x, roi_y, roi_w, roi_h))
        u_gray = cv2.cvtColor(u_roi, cv2.COLOR_BGR2GRAY)

        # 1. Projector Interference Masking (Feed-Forward)
        # Identify areas likely hit by bright projector light and suppress them.
        if self.h_c2p is not None:
            with QMutexLocker(self.last_projected_frame_mutex):
                if self.last_projected_frame_for_masking is not None:
                    ph, pw = self.last_projected_frame_for_masking.shape[:2]
                    S = np.array([[float(pw), 0, 0], [0, float(ph), 0], [0, 0, 1]], dtype=np.float32)
                    M_cam_to_proj = S @ self.h_c2p
                    T_roi = np.array([[1, 0, roi_x], [0, 1, roi_y], [0, 0, 1]], dtype=np.float32)
                    M_roi_to_proj = M_cam_to_proj @ T_roi
                    try:
                        # Performance: warp at lower resolution for interference masking
                        mask_w, mask_h = max(32, roi_w // 4), max(32, roi_h // 4)
                        T_scale = np.array([[mask_w/roi_w, 0, 0], [0, mask_h/roi_h, 0], [0, 0, 1]], dtype=np.float32)

                        # Use UMat for warped masking
                        u_proj_last = cv2.UMat(self.last_projected_frame_for_masking)
                        u_proj_small = cv2.warpPerspective(u_proj_last, T_scale @ M_roi_to_proj, (mask_w, mask_h))
                        u_proj_mask = cv2.resize(u_proj_small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

                        # Aggressive suppression (0.8) - help cancel out bright projector spots
                        u_suppression = cv2.multiply(u_proj_mask, 0.8)
                        u_gray = cv2.subtract(u_gray, u_suppression)
                    except: pass

        # 2. Advanced Pre-processing (GPU Accelerated)
        th_size = max(150, int(w / 30))
        th_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_size, th_size))
        u_tophat = cv2.morphologyEx(u_gray, cv2.MORPH_TOPHAT, th_kernel)

        u_gray_boosted = cv2.addWeighted(u_gray, 0.4, u_tophat, 0.6, 0)
        u_gray_boosted = self.tracking_clahe.apply(u_gray_boosted)

        if self.auto_threshold:
            _, u_t_global = cv2.threshold(u_gray_boosted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            u_t_local = cv2.adaptiveThreshold(u_gray_boosted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
            u_thresh = cv2.bitwise_and(u_t_global, u_t_local)
        else:
            effective_threshold = min(self.ir_threshold, 248)
            _, u_thresh = cv2.threshold(u_gray_boosted, effective_threshold, 255, cv2.THRESH_BINARY)

        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        u_thresh = cv2.morphologyEx(u_thresh, cv2.MORPH_OPEN, kernel_clean)
        u_thresh = cv2.dilate(u_thresh, kernel_clean, iterations=3)

        # Download from GPU to CPU for contour finding (OpenCV findContours needs CPU array)
        thresh = u_thresh.get()
        gray_boosted = u_gray_boosted.get()

        # Using findContours for efficient blob finding
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_points_raw = []
        rejected_points_raw = []

        # Min circularity: more lenient for still capture, strict for live tracking
        # Lowered to 0.60 to handle perspective distortion on large markers
        min_circ = 0.40 if force_full else 0.60

        for contour in contours:
            area = cv2.contourArea(contour)
            # Increased max_area to 15000 for high-res 4K+ camera feeds
            if 1 <= area < 15000:
                # Calculate circularity
                peri = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

                # Use intensity-weighted moments for sub-pixel centroid stability
                x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
                pad = 3
                x1 = max(0, x_b - pad)
                y1 = max(0, y_b - pad)
                x2 = min(roi_w, x_b + w_b + pad)
                y2 = min(roi_h, y_b + h_b + pad)

                dot_roi = gray_boosted[y1:y2, x1:x2]
                mask_roi = thresh[y1:y2, x1:x2]
                # Mask the grayscale ROI with the binary blob to ignore nearby noise/projector light
                # while preserving intensity weighting for sub-pixel precision.
                weighted_roi = cv2.bitwise_and(dot_roi, mask_roi)

                # REFINEMENT: Localized contrast boost for the blob area
                if np.max(weighted_roi) > 0:
                    weighted_roi = cv2.normalize(weighted_roi, None, 0, 255, cv2.NORM_MINMAX)

                M = cv2.moments(weighted_roi)

                if M["m00"] != 0:
                    cx_px = (M["m10"] / M["m00"]) + x1
                    cy_px = (M["m01"] / M["m00"]) + y1

                    # Sample peak intensity in the dot area
                    intensity = np.max(dot_roi)

                    cX = (cx_px + roi_x) / w
                    cY = (cy_px + roi_y) / h
                    candidate = {'pos': (cX, cY), 'intensity': intensity, 'area': area, 'circ': circularity}

                    if circularity >= min_circ and intensity >= (self.ir_threshold * 0.9):
                        detected_points_raw.append(candidate)
                    else:
                        rejected_points_raw.append(candidate)

        # Sort by intensity (brightest first) then area/circularity
        detected_points_raw.sort(key=lambda x: (x['intensity'], x['area'], x['circ']), reverse=True)
        detected_points_all = [x['pos'] for x in detected_points_raw]

        # Dot Persistence: only keep dots that were present in the last frame to filter out transient flashes
        detected_points = []
        # Radius for temporal persistence (normalized units)
        persistence_radius = 0.015

        for p in detected_points_all:
            # Check against ALL detections from the previous frame
            is_persistent = any(np.linalg.norm(np.array(p) - np.array(prev_p)) < persistence_radius for prev_p in self.last_raw_detections)

            # Check against last known tracked positions (with larger radius for movement)
            is_near_tracked = any(np.linalg.norm(np.array(p) - np.array(lp)) < 0.05 for lp in (self.last_tracked_points or []))

            if is_persistent or is_near_tracked:
                detected_points.append(p)
            elif force_full:
                # Still frames for calibration should include everything
                detected_points.append(p)

        self.last_raw_detections = detected_points_all # Store all for next frame's comparison
        rejected_points = [x['pos'] for x in rejected_points_raw]

        if return_rejected:
            # We skip the tracking logic and return raw dots for calibration dialog
            return detected_points, rejected_points

        if marker_cfg and len(marker_cfg) >= 3 and len(detected_points) >= (len(marker_cfg) - 1):
            # OC-TOLERANCE: If we see at least 3 markers (and 1 is missing), we can estimate the guitar pose
            num_markers = len(marker_cfg)

            # Pre-calculate Kalman predictions for stickiness cost (helps handle "swallowed" markers)
            predictions = []
            if self.confidence > 0.3 and len(self.kalman_filters) == num_markers:
                for kf in self.kalman_filters:
                    pred = kf.predict()
                    predictions.append((pred[0, 0], pred[1, 0]))

            # If we were tracking, prioritize points near last known location
            if self.confidence > 0.2 and self.last_tracked_points and len(self.last_tracked_points) > 0:
                center = np.mean(self.last_tracked_points, axis=0)
                detected_points.sort(key=lambda p: np.linalg.norm(np.array(p) - center))

            num_markers = len(marker_cfg)
            # Limit search to top candidates to keep CPU usage low
            limit = 15
            points_to_check = detected_points[:limit]

            best_match = None
            best_matrix = None
            best_overall_error = 1000.0 # High initial cost

            # Try to find a match with N markers first, then N-1
            # We use list to avoid search_count being a generator or None
            # Defensive checks to avoid search crashes
            if marker_fp is None: marker_fp = []
            if points_to_check is None: points_to_check = []

            for search_count in [num_markers, num_markers - 1]:
                if search_count < 3: continue
                if len(points_to_check) < search_count: continue

                for indices in combinations(range(len(points_to_check)), search_count):
                    point_combo = [points_to_check[i] for i in indices]
                    if not point_combo or len(point_combo) < search_count: continue

                    # EARLY GEOMETRIC PRUNING (RIGID BODY CHECK)
                    # We check if the distances between these points roughly match the marker fingerprint
                    # before doing permutations and homography. Fingerprint is distance-based and order-independent.
                    combo_distances = []
                    for i_d, j_d in combinations(range(len(point_combo)), 2):
                        combo_distances.append(np.linalg.norm(np.array(point_combo[i_d]) - np.array(point_combo[j_d])))
                    combo_fp = sorted(combo_distances)

                    # Get relevant reference fingerprint for this search count
                    if search_count == num_markers:
                        ref_fp = marker_fp
                    else:
                        # For N-1, we don't have a single ref_fp, but we can check if it's a subset
                        # Simplified check: just proceed for N-1 for now, or use a more complex subset check
                        ref_fp = None

                    if ref_fp and len(combo_fp) == len(ref_fp):
                        m1 = np.mean(ref_fp)
                        m2 = np.mean(combo_fp)
                        if m1 > 0.0001 and m2 > 0.0001:
                            fp_err = np.mean([abs((a/m2) - (b/m1)) for a, b in zip(combo_fp, ref_fp)])
                            if fp_err > 0.12: continue # Reject combinations with wrong geometry early
                        else:
                            continue

                    # If we found all N markers
                    if search_count == num_markers:
                        src_pts = np.float32(marker_cfg).reshape(-1, 1, 2)

                        # Case 1: Already tracking - Use distance-based ordering from last known transform
                        if self.last_homography is not None and self.confidence > 0.3:
                            transformed_src = cv2.perspectiveTransform(src_pts, self.last_homography)
                            ordered_dst = []
                            temp_dst = list(point_combo)
                            for i in range(num_markers):
                                if i >= len(transformed_src) or not temp_dst: break
                                pred = transformed_src[i][0]
                                closest_idx = min(range(len(temp_dst)), key=lambda j: np.linalg.norm(np.array(temp_dst[j]) - pred))
                                ordered_dst.append(temp_dst.pop(closest_idx))

                            if len(ordered_dst) == num_markers:
                                ordered_dst_arr = np.float32(ordered_dst).reshape(-1, 1, 2)
                                if len(src_pts) >= 4:
                                    m, _ = cv2.findHomography(src_pts, ordered_dst_arr, 0)
                                else:
                                    m, _ = cv2.estimateAffinePartial2D(src_pts, ordered_dst_arr)
                                    if m is not None:
                                        h = np.eye(3, dtype=np.float32); h[:2, :] = m; m = h

                                if m is not None:
                                    proj = cv2.perspectiveTransform(src_pts, m)
                                    err = np.mean(np.linalg.norm(proj - ordered_dst_arr, axis=2))
                                    if err < 0.05: # Reasonable lock
                                        best_match = ordered_dst
                                        best_matrix = m
                                        best_overall_error = err
                                        break # Good enough

                        # Case 2: Searching - Try all permutations and find the most physically plausible one
                        if best_match is None and len(marker_cfg) <= 4:
                            for p in permutations(point_combo):
                                p_arr = np.float32(p).reshape(-1, 1, 2)
                                if len(src_pts) >= 4:
                                    m, _ = cv2.findHomography(src_pts, p_arr, 0)
                                else:
                                    m, _ = cv2.estimateAffinePartial2D(src_pts, p_arr)
                                    if m is not None:
                                        h = np.eye(3, dtype=np.float32); h[:2, :] = m; m = h

                                if m is not None:
                                    # Calculate reprojection error
                                    proj = cv2.perspectiveTransform(src_pts, m)
                                    err = np.mean(np.linalg.norm(proj - p_arr, axis=2))

                                    # Geometric sanity check: Check determinant to avoid mirrors/flips
                                    det = np.linalg.det(m[:2, :2])
                                    if det <= 0.1: continue # Quick prune

                                    # Fingerprint already calculated in early pruning
                                    # We just reuse fp_err or recalculate it for the cost
                                    # Normalized cost: err + fingerprint_mismatch
                                    if marker_fp and len(combo_fp) == len(marker_fp):
                                        m1 = np.mean(marker_fp)
                                        m2 = np.mean(combo_fp)
                                        fp_err = np.mean([abs((a/m2) - (b/m1)) for a, b in zip(combo_fp, marker_fp)])
                                    else:
                                        fp_err = 0.5 # Penalty for mismatch in length/missing FP

                                    # Marker Stickiness: favor points closer to their predicted positions
                                    stickiness_err = 0
                                    if predictions and len(predictions) == num_markers:
                                        for i in range(num_markers):
                                            stickiness_err += np.linalg.norm(np.array(p[i]) - np.array(predictions[i]))
                                    elif self.last_tracked_points and len(self.last_tracked_points) == num_markers:
                                        for i in range(num_markers):
                                            stickiness_err += np.linalg.norm(np.array(p[i]) - np.array(self.last_tracked_points[i]))

                                    # Increased stickiness weight to prioritize temporal continuity over visual exactness
                                    total_cost = err + (fp_err * 6.0) + (stickiness_err * 10.0)

                                    if total_cost < best_overall_error:
                                        best_overall_error = total_cost
                                        best_match = list(p)
                                        best_matrix = m
                            if best_match and best_overall_error < 0.04: break

                    # If we found N-1 markers
                    elif search_count == num_markers - 1 and (self.last_homography is not None or predictions):
                        for missing_idx in range(num_markers):
                            subset_cfg = [marker_cfg[i] for i in range(num_markers) if i != missing_idx]
                            src_pts = np.float32(subset_cfg).reshape(-1, 1, 2)

                            # Use Kalman predictions if available for better alignment of the subset
                            if predictions and len(predictions) == num_markers:
                                subset_preds = [predictions[i] for i in range(num_markers) if i != missing_idx]
                                transformed_src = np.float32(subset_preds).reshape(-1, 1, 2)
                            else:
                                transformed_src = cv2.perspectiveTransform(src_pts, self.last_homography)
                            ordered_dst = []
                            temp_dst = list(point_combo)
                            for i in range(search_count):
                                if i >= len(transformed_src) or not temp_dst: break
                                pred = transformed_src[i][0]
                                closest_idx = min(range(len(temp_dst)), key=lambda j: np.linalg.norm(np.array(temp_dst[j]) - pred))
                                ordered_dst.append(temp_dst.pop(closest_idx))

                            if len(ordered_dst) == search_count:
                                ordered_dst_arr = np.float32(ordered_dst).reshape(-1, 1, 2)
                                if len(src_pts) >= 4:
                                    m, _ = cv2.findHomography(src_pts, ordered_dst_arr, 0)
                                else:
                                    m, _ = cv2.estimateAffinePartial2D(src_pts, ordered_dst_arr)
                                    if m is not None:
                                        h = np.eye(3, dtype=np.float32); h[:2, :] = m; m = h

                                if m is not None:
                                    proj = cv2.perspectiveTransform(src_pts, m)
                                    err = np.mean(np.linalg.norm(proj - ordered_dst_arr, axis=2))
                                    if err < 0.05:
                                        full_src = np.float32(marker_cfg).reshape(-1, 1, 2)
                                        predicted_full = cv2.perspectiveTransform(full_src, m).reshape(-1, 2)

                                        # RECOVERY: Look for the missing marker in the rejected list
                                        # This is crucial for the "tricky" neck marker
                                        final_points = list(predicted_full)
                                        missing_pred = predicted_full[missing_idx]

                                        all_cands = detected_points_raw + rejected_points_raw
                                        best_cand = None
                                        min_d = 0.03 # Search radius

                                        for cand in all_cands:
                                            cdist = np.linalg.norm(np.array(cand['pos']) - missing_pred)
                                            if cdist < min_d:
                                                min_d = cdist
                                                best_cand = cand['pos']

                                        if best_cand:
                                            final_points[missing_idx] = best_cand

                                        best_match = [tuple(p) for p in final_points]
                                        best_matrix = m
                                        best_overall_error = err
                                        break
                        if best_match: break
                if best_match: break

            if best_match and best_overall_error < 0.1: # More relaxed to ensure capture in noisy environments
                # Sanity Check: Filter out sudden jumps (reflections)
                if self.last_tracked_points is not None and self.confidence > 0.3:
                    prev_center = np.mean(self.last_tracked_points, axis=0)
                    curr_center = np.mean(best_match, axis=0)
                    # For a guitar, moving > 8% of screen width in one frame is unlikely
                    if np.linalg.norm(curr_center - prev_center) > 0.08:
                        self.confidence = max(0.0, self.confidence - 0.2)
                        pts = self.last_tracked_points if self.last_tracked_points is not None else []
                        return (pts, []) if return_rejected else pts

                # Kalman Correction
                kalman_pts = []
                for i, pt in enumerate(best_match):
                    if i >= len(self.kalman_filters): break
                    # Correct state with new observation
                    self.kalman_filters[i].correct(np.array([[np.float32(pt[0])], [np.float32(pt[1])]]))
                    # statePost is the state after correction
                    state = self.kalman_filters[i].statePost
                    kalman_pts.append((state[0, 0], state[1, 0]))

                if len(kalman_pts) < len(marker_cfg):
                    pts = self.last_tracked_points if self.last_tracked_points else detected_points
                    return (pts, []) if return_rejected else pts

                new_points_arr = np.array(kalman_pts, dtype=np.float32)

                # Decisive Reset: If current points are too far from the history
                if self.smoothed_points is not None and self.confidence > 0.5:
                    dist = np.linalg.norm(np.mean(new_points_arr, axis=0) - np.mean(self.smoothed_points, axis=0))
                    if dist > 0.06: # Significant jump detected (lowered for better stability)
                        self.confidence = max(0.0, self.confidence - 0.15)
                        pts = self.last_tracked_points if self.last_tracked_points is not None else []
                        return (pts, []) if return_rejected else pts

                # High-Distance Temporal Smoothing with Adaptive Alpha & Deadband
                if self.smoothed_points is None:
                    self.smoothed_points = new_points_arr
                    alpha = 1.0
                else:
                    # Calculate movement magnitude
                    move_dist = np.linalg.norm(np.mean(new_points_arr, axis=0) - np.mean(self.smoothed_points, axis=0))

                    # Deadband: if movement is microscopic, ignore it to prevent static jitter
                    # Increased for high-res stability
                    if move_dist < 0.0035:
                        pts = self.last_tracked_points if self.last_tracked_points is not None else []
                        return (pts, []) if return_rejected else pts

                    # Adaptive Alpha: lower when still, higher when moving
                    # base_alpha for stillness
                    base_alpha = (1.0 - self.smoothing_factor) * 0.08
                    # sensitivity to motion
                    motion_scale = 2.0
                    alpha = np.clip(base_alpha + move_dist * motion_scale, base_alpha, 0.9)

                    self.smoothed_points = self.smoothed_points * (1.0 - alpha) + new_points_arr * alpha

                # Moving Average over history
                self.history_points.append(self.smoothed_points.copy())
                if len(self.history_points) > self.history_len:
                    self.history_points.pop(0)

                avg_points = np.mean(self.history_points, axis=0)

                # Decisive snap: if we just regained tracking, don't average with old data
                if self.confidence < 0.3:
                    avg_points = new_points_arr
                    self.history_points = [new_points_arr.copy()]

                # RE-ESTIMATE HOMOGRAPHY from smoothed points to ensure stability of the mask
                src_pts = np.float32(marker_cfg).reshape(-1, 1, 2)
                dst_pts = np.float32(avg_points).reshape(-1, 1, 2)
                if len(src_pts) >= 4:
                    m, _ = cv2.findHomography(src_pts, dst_pts, 0)
                else:
                    m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if m is not None:
                        h = np.eye(3, dtype=np.float32); h[:2, :] = m; m = h

                if m is not None:
                    self.last_homography = m

                if self.confidence < 0.5 and (self.confidence + self.confidence_gain) >= 0.5:
                    self.status_update.emit("Tracking Status: LOCKED")

                self.confidence = min(1.0, self.confidence + self.confidence_gain)
                self.last_tracked_points = [tuple(p) for p in avg_points]
                return (self.last_tracked_points, []) if return_rejected else self.last_tracked_points

        if self.last_tracked_points is not None:
            if roi_x != 0 or roi_y != 0 or roi_w != w or roi_h != h:
                self.last_tracked_points = None
                res = self.get_tracked_points(frame, return_rejected=return_rejected)
                return res if res is not None else (([], []) if return_rejected else [])

        if self.confidence > 0.1 and (self.confidence - self.confidence_decay) <= 0.1:
            self.status_update.emit("Tracking Status: LOST")

        # Drop confidence faster if we don't see anything
        self.confidence = max(0.0, self.confidence - self.confidence_decay * 1.5)

        # If tracking is completely lost, clear history to avoid "jumps" when regained
        if self.confidence <= 0:
            self.history_points = []
            self.smoothed_points = None
            self.last_homography = None
            # Decisive Reset of Kalman filters to prevent old state from polluting new lock
            if self.marker_config:
                self.init_kalman(len(self.marker_config))

        if (self.confidence > 0.1 or self.tracking_freeze_enabled) and self.last_tracked_points is not None:
            return (self.last_tracked_points, []) if return_rejected else self.last_tracked_points

        return (detected_points, rejected_points) if return_rejected else detected_points

    def _update_warp_maps(self, w_render, h_render, w_proj, h_proj):
        if w_render <= 0 or h_render <= 0 or w_proj <= 0 or h_proj <= 0: return
        self.map_x = np.zeros((h_proj, w_proj), dtype=np.float32)
        self.map_y = np.zeros((h_proj, w_proj), dtype=np.float32)

        wp = np.array(self.warp_points)
        wp[:, 0] *= w_proj
        wp[:, 1] *= h_proj

        res = self.warp_grid_res

        for r in range(res - 1):
            for c in range(res - 1):
                # Quad corners in the point list
                p1 = r * res + c
                p2 = r * res + (c + 1)
                p3 = (r + 1) * res + (c + 1)
                p4 = (r + 1) * res + c

                dst_q = np.float32([wp[p1], wp[p2], wp[p3], wp[p4]])

                # Source quad in linear grid (mapped back to render resolution)
                src_x_start = c * (w_render // (res - 1))
                src_y_start = r * (h_render // (res - 1))
                src_x_end = (c + 1) * (w_render // (res - 1))
                src_y_end = (r + 1) * (h_render // (res - 1))

                if c == res - 2: src_x_end = w_render
                if r == res - 2: src_y_end = h_render

                src_q = np.float32([
                    [src_x_start, src_y_start], [src_x_end, src_y_start],
                    [src_x_end, src_y_end], [src_x_start, src_y_end]
                ])

                # For remap, we need mapping from output pixel back to input pixel
                M = cv2.getPerspectiveTransform(dst_q, src_q)

                # Find destination quad bounding box
                min_x = int(np.min(dst_q[:, 0]))
                max_x = int(np.max(dst_q[:, 0]))
                min_y = int(np.min(dst_q[:, 1]))
                max_y = int(np.max(dst_q[:, 1]))

                # Clip to image boundaries (at projector resolution)
                min_x, max_x = max(0, min_x), min(w_proj, max_x)
                min_y, max_y = max(0, min_y), min(h_proj, max_y)

                # Create a localized coordinate grid
                yy, xx = np.mgrid[min_y:max_y, min_x:max_x].astype(np.float32)
                points = np.stack([xx.ravel(), yy.ravel()], axis=1).reshape(-1, 1, 2)

                if points.size > 0:
                    transformed = cv2.perspectiveTransform(points, M).reshape(-1, 2)
                    # Map only the pixels inside the warped quad
                    mask = np.zeros((max_y-min_y, max_x-min_x), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(dst_q - [min_x, min_y]), 255)

                    # Apply transformation
                    self.map_x[min_y:max_y, min_x:max_x][mask > 0] = transformed[:, 0].reshape(max_y-min_y, max_x-min_x)[mask > 0]
                    self.map_y[min_y:max_y, min_x:max_x][mask > 0] = transformed[:, 1].reshape(max_y-min_y, max_x-min_x)[mask > 0]

        self._warp_map_dirty = False

    def process_video(self):
        """Main Rendering Loop (Target 30 FPS)"""
        target_fps = 30
        frame_time = 1.0 / target_fps

        while self._running:
            try:
                start_time = time.time()

                if self.blackout_active:
                    h_proj, w_proj = self.projector_height, self.projector_width
                    black = np.zeros((h_proj, w_proj, 3), dtype=np.uint8)
                    self.projector_frame_ready.emit(QImage(black.data, w_proj, h_proj, w_proj * 3, QImage.Format_RGB888).copy())
                    elapsed = time.time() - start_time
                    QThread.msleep(max(1, int((frame_time - elapsed) * 1000)))
                    continue

                # 1. Update Tracking Info (from shared data)
                with QMutexLocker(self.tracking_mutex):
                    tracked_points = self.last_tracked_points_internal
                    curr_homography = self.last_homography_internal
                    curr_confidence = self.confidence_internal
                    t_fps = self.tracking_frame_count / max(1, (time.time() - self.last_stats_time))
                    self.tracking_fps = t_fps

                # 2. Get Main Frame (for UI/Outlines/Calibration)
                with QMutexLocker(self.latest_main_frame_mutex):
                    main_frame = self.latest_main_frame.copy() if self.latest_main_frame is not None else None
                    ui_tracked_points = self.latest_tracked_points_for_ui
                    curr_frame_id = self.latest_main_frame_id

                if main_frame is None:
                    main_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(main_frame, "WAITING FOR CAMERA...", (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

                h_cam, w_cam = main_frame.shape[:2]

                # Projector and Render Resolution
                h_proj, w_proj = self.projector_height, self.projector_width

                # Dynamically calculate render resolution based on projector dimensions
                # For ultra-wide setups (like 4480px), we want higher internal res
                if self._run_sls_flag or self._run_boundary_detection_flag or self._run_calibration_flag:
                    # Bypassing render scale for calibration steps to ensure 1:1 pixel mapping
                    w, h = w_proj, h_proj
                else:
                    target_w = int(w_proj * self.render_scale)
                    target_h = int(h_proj * self.render_scale)

                    # Clamp to reasonable limits for performance
                    self.render_width = max(640, min(target_w, 2560))
                    self.render_height = max(360, min(target_h, 1440))
                    w, h = self.render_width, self.render_height

                if self.camera_matrix is None:
                    # Estimate camera matrix based on FOV (approx 60 deg)
                    f = max(w_cam, h_cam)
                    self.camera_matrix = np.array([
                        [f, 0, w_cam / 2],
                        [0, f, h_cam / 2],
                        [0, 0, 1]
                    ], dtype=np.float32)

                if self.projector_buffer is None or self.projector_buffer.shape[:2] != (h, w):
                    self.projector_buffer = np.zeros((h, w, 3), dtype=np.uint8)
                    self.mask_buffer = np.zeros((h, w, 3), dtype=np.uint8)
                    self.u_projector_output = cv2.UMat(np.zeros((h, w, 3), dtype=np.uint8))
                    self.u_mask_buffer = cv2.UMat(np.zeros((h, w, 3), dtype=np.uint8))
                    self.static_warp_cache.clear()
                    self.video_umat_cache.clear()
                    self._warp_map_dirty = True

                if self._run_boundary_detection_flag:
                    # Ensure resolution is stable before capturing
                    if self.requested_camera_res[0] > 5000 or \
                       main_frame.shape[1] != self.requested_camera_res[0] or \
                       main_frame.shape[0] != self.requested_camera_res[1]:
                        # Still waiting for camera thread to catch up
                        self._sls_curr_wait = 0
                        continue

                    if self._boundary_step == 0:
                        # Capture Black Frame
                        self.projector_buffer.fill(0)
                        if self._sls_curr_wait >= self._sls_wait_frames:
                            if curr_frame_id > self._last_captured_frame_id:
                                # Use normalized grayscale for boundary detection
                                gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                                self._boundary_captures.append(self.normalize_intensity(gray))
                                self._boundary_step = 1
                                self._sls_curr_wait = 0
                                self._last_captured_frame_id = curr_frame_id
                        else:
                            self._sls_curr_wait += 1
                    elif self._boundary_step == 1:
                        # Capture White Frame
                        self.projector_buffer.fill(255)
                        if self._sls_curr_wait >= self._sls_wait_frames:
                            if curr_frame_id > self._last_captured_frame_id:
                                # Use normalized grayscale for boundary detection
                                gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                                self._boundary_captures.append(self.normalize_intensity(gray))
                                self._boundary_step = 2
                                self._sls_curr_wait = 0
                                self._last_captured_frame_id = curr_frame_id
                        else:
                            self._sls_curr_wait += 1
                    elif self._boundary_step == 2:
                        # Process Detection
                        print("Processing Projector Boundary...")
                        black = self._boundary_captures[0]
                        white = self._boundary_captures[1]

                        # Use a more robust diff: max(0, white - black) to ignore things that get DARKER when white is projected
                        diff = cv2.subtract(white, black)

                        # Mask out persistent bright spots (like IR markers)
                        # Markers are usually near-saturated in both frames
                        _, marker_mask = cv2.threshold(black, 230, 255, cv2.THRESH_BINARY)
                        kernel_small = np.ones((3,3), np.uint8)
                        marker_mask = cv2.dilate(marker_mask, kernel_small, iterations=5)
                        diff[marker_mask > 0] = 0

                        # Try adaptive thresholding first for better detail in uneven lighting
                        thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 21, -5)

                        # Combined with Otsu for global structure
                        _, otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        thresh = cv2.bitwise_or(thresh, otsu)

                        # Clean up
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                        thresh = cv2.dilate(thresh, kernel, iterations=3)

                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # Filter contours by size - significantly lowered threshold (0.1% of FOV) to capture small projectors
                        valid_contours = [c for c in contours if cv2.contourArea(c) > (w_cam * h_cam * 0.001)]

                        if valid_contours:
                            # Find largest contour
                            main_contour = max(valid_contours, key=cv2.contourArea)
                            area_ratio = cv2.contourArea(main_contour) / (w_cam * h_cam)

                            # Use Convex Hull to ignore internal details and just get the footprint
                            hull = cv2.convexHull(main_contour)

                            # Simplification (use a smaller epsilon to avoid losing detail)
                            peri = cv2.arcLength(hull, True)
                            approx = cv2.approxPolyDP(hull, 0.01 * peri, True)

                            # Ensure at least a quad if it's too simple
                            if len(approx) < 4:
                                rect = cv2.minAreaRect(hull)
                                approx = cv2.boxPoints(rect).reshape(-1, 1, 2)

                            # If the detected boundary is still very complex, use minAreaRect
                            elif len(approx) > 12:
                                rect = cv2.minAreaRect(hull)
                                approx = cv2.boxPoints(rect).reshape(-1, 1, 2)

                            # Store as normalized points
                            pts = [ (float(p[0][0]) / w_cam, float(p[0][1]) / h_cam) for p in approx ]
                            self.projector_boundary = pts
                            self.boundary_detected.emit(pts)
                        else:
                            print("Error: No projector light detected!")
                            self.boundary_detected.emit([])

                        self._run_boundary_detection_flag = False
                        self._boundary_step = 0
                        self._boundary_captures = []
                elif self._run_sls_flag:
                    # Ensure resolution is stable before capturing
                    if self.requested_camera_res[0] > 5000 or \
                       main_frame.shape[1] != self.requested_camera_res[0] or \
                       main_frame.shape[0] != self.requested_camera_res[1]:
                        # Still waiting for camera thread to catch up
                        self._sls_curr_wait = 0
                        continue

                    # Structured Light Scanning takes priority
                    total_x = len(self._sls_patterns_x)
                    total_y = len(self._sls_patterns_y)

                    if self._sls_step < total_x:
                        self.status_update.emit(f"Room Scan: Capturing X pattern {self._sls_step+1}/{total_x}...")
                        pattern = self._sls_patterns_x[self._sls_step]
                        # Patterns are already generated at projector resolution
                        # We must ensure they are shown 1:1 to avoid aliasing
                        if pattern.shape[1] != w or pattern.shape[0] != h:
                            pattern = cv2.resize(pattern, (w, h), interpolation=cv2.INTER_NEAREST)
                        self.projector_buffer = cv2.merge([pattern, pattern, pattern])
                        if self._sls_curr_wait >= self._sls_wait_frames:
                            if curr_frame_id > self._last_captured_frame_id:
                                # Use normalized grayscale for SLS to ensure Gray code decoding is reliable
                                gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                                self._sls_captures_x.append(self.normalize_intensity(gray))
                                self._sls_step += 1
                                self._sls_curr_wait = 0
                                self._last_captured_frame_id = curr_frame_id
                        else:
                            self._sls_curr_wait += 1
                    elif self._sls_step < total_x + total_y:
                        idx = self._sls_step - total_x
                        self.status_update.emit(f"Room Scan: Capturing Y pattern {idx+1}/{total_y}...")
                        pattern = self._sls_patterns_y[idx]
                        if pattern.shape[1] != w or pattern.shape[0] != h:
                            pattern = cv2.resize(pattern, (w, h), interpolation=cv2.INTER_NEAREST)
                        self.projector_buffer = cv2.merge([pattern, pattern, pattern])
                        if self._sls_curr_wait >= self._sls_wait_frames:
                            if curr_frame_id > self._last_captured_frame_id:
                                # Use normalized grayscale for SLS to ensure Gray code decoding is reliable
                                gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                                self._sls_captures_y.append(self.normalize_intensity(gray))
                                self._sls_step += 1
                                self._sls_curr_wait = 0
                                self._last_captured_frame_id = curr_frame_id
                        else:
                            self._sls_curr_wait += 1
                    else:
                        # Decoding
                        print("Decoding Room Scan...")
                        proj_x, valid_x = decode_gray_code(self._sls_captures_x, self.projector_width)
                        proj_y, valid_y = decode_gray_code(self._sls_captures_y, self.projector_height)
                        valid = valid_x & valid_y

                        # Refine valid mask with morphological cleanup
                        kernel_valid = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                        valid_uint8 = (valid.astype(np.uint8) * 255)
                        valid_uint8 = cv2.morphologyEx(valid_uint8, cv2.MORPH_CLOSE, kernel_valid)
                        valid = valid_uint8 > 128

                        # Store dense LUT as normalized [0-1] and apply median filter to remove noise-induced "spikes"
                        self.sls_lut_x = cv2.medianBlur(proj_x.astype(np.float32), 5) / self.projector_width
                        self.sls_lut_y = cv2.medianBlur(proj_y.astype(np.float32), 5) / self.projector_height
                        self.sls_valid_mask = valid.astype(np.uint8)

                        # Automatically derive Projector Boundary from SLS valid mask
                        # Clean up the mask aggressively to fill holes and smooth edges
                        kernel = np.ones((21, 21), np.uint8)
                        mask_solid = cv2.morphologyEx(self.sls_valid_mask, cv2.MORPH_CLOSE, kernel)
                        mask_solid = cv2.morphologyEx(mask_solid, cv2.MORPH_OPEN, kernel)
                        mask_solid = cv2.dilate(mask_solid, kernel, iterations=2)

                        contours, _ = cv2.findContours(mask_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # Filter contours by size
                        valid_contours = [c for c in contours if cv2.contourArea(c) > (w_cam * h_cam * 0.1)]

                        if valid_contours:
                            main_contour = max(valid_contours, key=cv2.contourArea)
                            # Use convex hull and force a simple polygon or rectangle
                            hull = cv2.convexHull(main_contour)
                            peri = cv2.arcLength(hull, True)
                            approx = cv2.approxPolyDP(hull, 0.01 * peri, True)

                            if len(approx) < 4 or len(approx) > 12:
                                rect = cv2.minAreaRect(hull)
                                approx = cv2.boxPoints(rect).reshape(-1, 1, 2)

                            # Store as normalized points
                            pts = [ (float(p[0][0]) / w_cam, float(p[0][1]) / h_cam) for p in approx ]
                            self.projector_boundary = pts
                            self.boundary_detected.emit(pts)
                            print(f"One-Click Sync: Detected boundary with {len(pts)} points.")

                        # Collect mapping points for RBF (sub-sampled)
                        calib_data = []
                        step = 10 # Sample every 10 pixels
                        # Ensure indices are within bounds
                        h_idx, w_idx = valid.shape
                        for r in range(0, h_idx, step):
                            for c in range(0, w_idx, step):
                                if valid[r, c]:
                                    # Normalize proj_x/y to [0, 1] based on full native projector res
                                    nx = float(proj_x[r, c]) / self.projector_width
                                    ny = float(proj_y[r, c]) / self.projector_height
                                    calib_data.append([float(c), float(r), nx, ny])

                        if len(calib_data) > 10:
                            self.init_rbf_from_points(calib_data)
                            # Estimate homography from sample points for fallback (maps to norm space)
                            pts_arr = np.array(calib_data)
                            self.h_c2p, _ = cv2.findHomography(pts_arr[:100, :2], pts_arr[:100, 2:])
                            self.calibration_camera_res = (w_cam, h_cam)
                            self.calibration_complete.emit(True)
                        else:
                            self.calibration_complete.emit(False)

                        self._run_sls_flag = False
                        self.show_calibration_verify = True # Automatically show verification
                elif self.show_camera_on_projector:
                    cv2.resize(main_frame, (w, h), dst=self.projector_buffer)
                elif self.show_calibration_pattern:
                    # Draw a centered checkerboard pattern for robust calibration
                    # Inset by 15% to ensure it stays on camera even if alignment is rough
                    self.projector_buffer.fill(255) # White background
                    margin_x = int(w * 0.15)
                    margin_y = int(h * 0.15)
                    grid_w = w - 2 * margin_x
                    grid_h = h - 2 * margin_y

                    cols, rows = 10, 7 # 9x6 internal corners
                    sq_w = grid_w // cols
                    sq_h = grid_h // rows

                    # Center the actual grid
                    start_x = margin_x + (grid_w % cols) // 2
                    start_y = margin_y + (grid_h % rows) // 2

                    for r in range(rows):
                        for c in range(cols):
                            if (r + c) % 2 == 1:
                                cv2.rectangle(self.projector_buffer,
                                              (start_x + c * sq_w, start_y + r * sq_h),
                                              (start_x + (c + 1) * sq_w, start_y + (r + 1) * sq_h),
                                              (0, 0, 0), -1)

                    # Overlay status (moved to bottom to avoid grid interference)
                    status_text = f"ANALYZING... ({self._calib_frames_captured}/{self._calib_total_frames})"
                    if hasattr(self, '_last_calib_attempt_time') and time.time() - self._last_calib_attempt_time > 3.0:
                        status_text += " (Check Exposure/Alignment)"
                    cv2.putText(self.projector_buffer, status_text, (w//2-200, h - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Visual capture confirmation
                    if hasattr(self, '_last_calib_success_time') and time.time() - self._last_calib_success_time < 0.2:
                        # Flash green border on successful frame capture
                        cv2.rectangle(self.projector_buffer, (0, 0), (w, h), (0, 255, 0), 20)
                elif self.show_calibration_verify:
                    self.projector_buffer.fill(0)
                    # Draw high-visibility grid and circles
                    margin_x = int(w * 0.15)
                    margin_y = int(h * 0.15)
                    grid_w = w - 2 * margin_x
                    grid_h = h - 2 * margin_y
                    sq_w = grid_w // 10
                    sq_h = grid_h // 7
                    start_x = margin_x + (grid_w % 10) // 2
                    start_y = margin_y + (grid_h % 7) // 2

                    # Draw lines
                    for r in range(rows := 7 + 1):
                        y_line = start_y + r * sq_h
                        if y_line < h:
                            cv2.line(self.projector_buffer, (0, y_line), (w, y_line), (50, 50, 50), 1)
                    for c in range(cols := 10 + 1):
                        x_line = start_x + c * sq_w
                        if x_line < w:
                            cv2.line(self.projector_buffer, (x_line, 0), (x_line, h), (50, 50, 50), 1)

                    for r in range(1, 7):
                        for c in range(1, 10):
                            color = (0, 255, 0) if (r+c)%2==0 else (0, 255, 255)
                            target_x = start_x + c * sq_w
                            target_y = start_y + r * sq_h
                            # Circle with crosshair
                            cv2.circle(self.projector_buffer, (target_x, target_y), 15, color, 2, cv2.LINE_AA)
                            cv2.circle(self.projector_buffer, (target_x, target_y), 2, (255, 255, 255), -1)
                            cv2.line(self.projector_buffer, (target_x-20, target_y), (target_x+20, target_y), color, 1)
                            cv2.line(self.projector_buffer, (target_x, target_y-20), (target_x, target_y+20), color, 1)

                            cv2.putText(self.projector_buffer, f"{c},{r}", (target_x + 18, target_y + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.putText(self.projector_buffer, "VERIFICATION MODE - CHECK CAMERA ALIGNMENT", (50, h-50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Clear composition buffers
                    self.projector_buffer.fill(0)
                    if self.u_projector_output is None:
                        self.u_projector_output = cv2.UMat(np.zeros((h, w, 3), dtype=np.uint8))
                    else:
                        cv2.rectangle(self.u_projector_output, (0, 0), (w, h), (0, 0, 0), -1)

                projector_output = self.projector_buffer
                # Performance: Composition on GPU
                u_projector_output = self.u_projector_output

                # If we updated the CPU buffer (SLS/Calibration), upload it to GPU
                if self._run_sls_flag or self._run_calibration_flag or self._run_boundary_detection_flag or self.show_camera_on_projector or self.show_calibration_pattern or self.show_calibration_verify or self.show_splash:
                    u_projector_output = cv2.UMat(projector_output)
                    self.u_projector_output = u_projector_output # Keep reference

                # Handle Auto-Calibration logic (Multi-frame averaging)
                if self._run_calibration_flag:
                    # Ensure resolution is stable before capturing
                    if self.requested_camera_res[0] > 5000 or \
                       main_frame.shape[1] != self.requested_camera_res[0] or \
                       main_frame.shape[0] != self.requested_camera_res[1]:
                        continue

                    if curr_frame_id > self._last_captured_frame_id:
                        gray = self.boost_contrast(main_frame)
                        board_size = (9, 6)

                        # Try high-accuracy detection first
                        ret, corners = cv2.findChessboardCornersSB(gray, board_size, cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_EXHAUSTIVE)

                        if not ret:
                            if not hasattr(self, '_last_calib_attempt_time'):
                                self._last_calib_attempt_time = time.time()

                            if time.time() - self._last_calib_attempt_time > 5.0:
                                # Fallback to standard detection after 5 seconds
                                ret, corners = cv2.findChessboardCorners(gray, board_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

                        if ret:
                            if self._calib_corners_sum is None:
                                self._calib_corners_sum = corners.astype(np.float64)
                            else:
                                self._calib_corners_sum += corners
                            self._calib_frames_captured += 1
                            self._last_calib_success_time = time.time()
                            self._last_calib_attempt_time = time.time()

                        self._last_captured_frame_id = curr_frame_id

                    # Check if we have enough samples or reached timeout
                    if self._calib_frames_captured >= self._calib_total_frames:
                        avg_corners = (self._calib_corners_sum / self._calib_frames_captured).astype(np.float32)

                        # Calculate reference points matching the centered pattern
                        margin_x = int(w * 0.15)
                        margin_y = int(h * 0.15)
                        grid_w = w - 2 * margin_x
                        grid_h = h - 2 * margin_y
                        sq_w = grid_w // 10
                        sq_h = grid_h // 7
                        start_x = margin_x + (grid_w % 10) // 2
                        start_y = margin_y + (grid_h % 7) // 2

                        proj_pts_norm = []
                        for r in range(1, 7):
                            for c in range(1, 10):
                                proj_pts_norm.append([(start_x + c * sq_w) / w, (start_y + r * sq_h) / h])

                        proj_pts_norm = np.array(proj_pts_norm, dtype=np.float32)
                        cam_pts = avg_corners.reshape(-1, 2)

                        # Store as point list for RBF (cam_pixels to norm_projector)
                        calib_data = []
                        for i in range(len(cam_pts)):
                            calib_data.append([float(cam_pts[i, 0]), float(cam_pts[i, 1]),
                                               float(proj_pts_norm[i, 0]), float(proj_pts_norm[i, 1])])

                        self.init_rbf_from_points(calib_data)
                        # Also keep homography as a robust fallback (maps cam_pixels to norm_projector)
                        self.h_c2p, _ = cv2.findHomography(cam_pts, proj_pts_norm)
                        self.calibration_camera_res = (w_cam, h_cam)

                        self._run_calibration_flag = False
                        self._calib_frames_captured = 0
                        self._calib_corners_sum = None
                        self.calibration_complete.emit(True)
                    elif self._run_calibration_flag and self.frame_count % 100 == 0: # Timeout safety
                         # If we've been trying too long (frame_count is not perfect for time but works)
                         pass

                if self.show_splash:
                    if self.splash_player is None:
                        self.splash_player = VideoPlayer(resource_path('logo.mkv'))
                        self.splash_player.start()

                    splash_frame, _ = self.splash_player.get_frame()
                    if splash_frame is not None:
                        projector_output = cv2.resize(splash_frame, (w, h))
                    else:
                        cv2.putText(projector_output, "PRE-SHOW", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                else:
                    if self.splash_player is not None:
                        self.splash_player.stop()
                        self.splash_player = None

                # Tracking is now handled by TrackingThread.
                # We use ui_tracked_points for visualization on the main_frame.
                if ui_tracked_points:
                    for pt in ui_tracked_points:
                        try:
                            # pt is normalized, convert to pixels for drawing
                            px = int(pt[0] * w_cam)
                            py = int(pt[1] * h_cam)
                            cv2.circle(main_frame, (px, py), 5, (0, 0, 255), -1)
                        except (TypeError, ValueError, IndexError):
                            pass

                # Draw calibration overlay on camera feed
                if self._run_calibration_flag:
                    gray = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                    if ret:
                        cv2.drawChessboardCorners(main_frame, (9, 6), corners, ret)
                        cv2.putText(main_frame, f"CALIBRATING: {self._calib_frames_captured}/{self._calib_total_frames}",
                                    (10, h_cam-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(main_frame, "CHECKERBOARD NOT FOUND", (10, h_cam-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Skip particles and occlusion during setup to save CPU and avoid corrupting calibration patterns
                if not (self._run_sls_flag or self._run_calibration_flag or self._run_boundary_detection_flag):
                    self.update_particles(tracked_points, h, w, w_cam, h_cam)
                    # Draw particles on CPU and upload once (more efficient than individual GPU draws for many points)
                    particle_layer = np.zeros((h, w, 3), dtype=np.uint8)
                    self.draw_particles(particle_layer)
                    u_projector_output = cv2.add(u_projector_output, cv2.UMat(particle_layer))

                # Performer Occlusion Logic
                if self.occlusion_enabled and not (self._run_sls_flag or self._run_calibration_flag or self._run_boundary_detection_flag):
                    fg_mask = self.back_subtractor.apply(main_frame)
                    # Cleanup mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

                    # Exclude the guitar area from occlusion if tracking
                    if curr_confidence > 0.5 and tracked_points:
                        pts = np.array(tracked_points, dtype=np.int32)
                        cv2.fillPoly(fg_mask, [cv2.convexHull(pts)], 0)

                    # Transform occlusion mask to internal render space
                    self.occlusion_mask = self.warp_full_frame_to_projector(fg_mask, w_cam, h_cam, w, h)

                with QMutexLocker(self.mask_mutex):
                    iterable_masks = [m for m in self.masks] # Snapshot for this frame

                # Frame cache for this rendering cycle
                cycle_generator_cache = {}
                cycle_video_cache = {} # path -> (frame_resized, id)

                for mask in iterable_masks:
                    mask_id = id(mask)
                    target_fade = 1.0 if mask.visible else 0.0
                    curr_fade = self.mask_fade_levels.get(mask_id, 0.0)

                    # Smoothly transition fade level (approx 0.5s at 30fps)
                    if curr_fade < target_fade:
                        curr_fade = min(target_fade, curr_fade + 0.1)
                    elif curr_fade > target_fade:
                        curr_fade = max(target_fade, curr_fade - 0.1)
                    self.mask_fade_levels[mask_id] = curr_fade

                    if curr_fade <= 0 and not mask.visible: continue
                    if not mask.video_path: continue

                    frame_cue = None
                    frame_id = 0

                    if mask.video_path == "generative":
                        if "generative" not in cycle_generator_cache:
                            # Keep generative frames as UMat if possible (though most gen is CPU for now)
                            gen_frame = self.get_generative_frame(h, w)
                            cycle_generator_cache["generative"] = (cv2.UMat(gen_frame), self.frame_count, h, w)
                        u_frame_cue, frame_id, fh, fw = cycle_generator_cache["generative"]
                    elif mask.video_path.startswith("generator:"):
                        pattern = mask.video_path.split(":")[-1]
                        if mask.video_path not in cycle_generator_cache:
                            vj_frame = self.get_vj_generator(pattern, h, w)
                            cycle_generator_cache[mask.video_path] = (cv2.UMat(vj_frame), self.frame_count, h, w)
                        u_frame_cue, frame_id, fh, fw = cycle_generator_cache[mask.video_path]
                    else:
                        if mask.video_path in cycle_video_cache:
                            u_frame_cue, frame_id, fh, fw = cycle_video_cache[mask.video_path]
                        else:
                            with QMutexLocker(self.player_mutex):
                                if mask.video_path not in self.video_players:
                                    player = VideoPlayer(mask.video_path)
                                    player.start()
                                    self.video_players[mask.video_path] = player
                                player = self.video_players[mask.video_path]
                                frame_cue_raw, frame_id = player.get_frame()

                            if frame_cue_raw is not None:
                                # Check RAM cache for UMat to save upload time
                                cache_key_umat = (mask.video_path, frame_id)
                                u_frame_cue = self.video_umat_cache.get(cache_key_umat)
                                if u_frame_cue is None:
                                    # Downscale once per video path per cycle if needed
                                    fh_raw, fw_raw = frame_cue_raw.shape[:2]
                                    if fw_raw > w or fh_raw > h:
                                        frame_cue_raw = cv2.resize(frame_cue_raw, (w, h), interpolation=cv2.INTER_LINEAR)

                                    u_frame_cue = cv2.UMat(frame_cue_raw)
                                    if len(self.video_umat_cache) > 200:
                                        self.video_umat_cache.pop(next(iter(self.video_umat_cache)))
                                    self.video_umat_cache[cache_key_umat] = u_frame_cue

                                fh, fw = frame_cue_raw.shape[:2]
                                cycle_video_cache[mask.video_path] = (u_frame_cue, frame_id, fh, fw)
                            else:
                                u_frame_cue = None
                                fh, fw = 0, 0

                    if mask.tag in self.fades:
                        fade_info = self.fades[mask.tag]
                        elapsed = time.time() - fade_info['start_time']
                        if elapsed < self.fade_duration:
                            if fade_info['prev_path'] in self.video_players:
                                with QMutexLocker(self.player_mutex):
                                    prev_player = self.video_players[fade_info['prev_path']]
                                    prev_frame, _ = prev_player.get_frame()
                                if prev_frame is not None and u_frame_cue is not None:
                                    u_prev = cv2.UMat(prev_frame)
                                    alpha = elapsed / self.fade_duration
                                    u_frame_cue = cv2.addWeighted(u_prev, 1.0 - alpha, u_frame_cue, alpha, 0)
                        else:
                            del self.fades[mask.tag]

                    if u_frame_cue is not None:
                        # Performance: Already on GPU (UMat)

                        # Performance: Only apply FX if not fully throttled or essential
                        u_frame_cue = self.apply_fx(u_frame_cue, mask, h=h, w=w)

                        if mask.design_overlay != 'none':
                            # design masks are generated on CPU but bitwise_and on GPU is faster
                            design_m = self.get_design_mask(mask.design_overlay, h, w)
                            u_design_m = cv2.UMat(design_m)
                            u_design_m_3ch = cv2.merge([u_design_m, u_design_m, u_design_m])
                            u_frame_cue = cv2.bitwise_and(u_frame_cue, u_design_m_3ch)

                        # Safety Fallback
                        is_safe = True
                        if mask.type == 'dynamic' and curr_confidence < 0.2 and self.safety_mode_enabled:
                            is_safe = False
                            # Override frame with fallback generator
                            u_frame_cue = cv2.UMat(self.get_vj_generator(self.fallback_generator, h, w))

                        u_effective_frame_cue = u_frame_cue
                        if mask.type == 'dynamic' and mask.is_linked:
                            # Only fade out tracked masks when tracking is shaky
                            if curr_confidence < 1.0:
                                # Ensure it doesn't go totally black if safety fallback is active
                                min_vis = 0.4 if not is_safe else 0.0
                                alpha = max(min_vis, curr_confidence)
                                u_effective_frame_cue = cv2.convertScaleAbs(u_frame_cue, alpha=alpha)

                        effective_opacity = mask.opacity * curr_fade
                        if mask.fx_params.get('lfo_enabled') and mask.fx_params.get('lfo_target') == 'opacity':
                            effective_opacity *= lfo_val

                        if effective_opacity < 1.0:
                            u_effective_frame_cue = cv2.convertScaleAbs(u_effective_frame_cue, alpha=effective_opacity)

                        # Static Caching Check
                        cache_key = (mask_id, tuple(map(tuple, mask.source_points)))
                        u_warped_cue = None
                        u_mask_img = None

                        if mask.type == 'static' and cache_key in self.static_warp_cache:
                            cached_fid, u_warped_cached, u_mask_cached = self.static_warp_cache[cache_key]
                            if cached_fid == frame_id:
                                u_warped_cue = u_warped_cached
                                u_mask_img = u_mask_cached

                        if u_warped_cue is None:
                            if mask.type == 'dynamic' and ((mask.is_linked and curr_homography is not None) or not is_safe):
                                src_pts = np.float32(mask.source_points)

                                if is_safe:
                                    cal_w, cal_h = self.calibration_camera_res if self.calibration_camera_res else (w_cam, h_cam)
                                    # 3D PnP Perspective Logic
                                    if self.pnp_enabled and self.marker_config and len(self.marker_config) == 4 and len(tracked_points) == 4:
                                        # Model points (markers in their reference plane, Z=0)
                                        model_pts = np.array([ [p[0]*cal_w, p[1]*cal_h, 0] for p in self.marker_config ], dtype=np.float32)
                                        image_pts = (np.array(tracked_points, dtype=np.float32) * [w_cam, h_cam]).reshape(-1, 2)

                                        # Use a camera matrix matching current w_cam, h_cam
                                        f = max(w_cam, h_cam)
                                        cam_mtx = np.array([[f, 0, w_cam/2], [0, f, h_cam/2], [0, 0, 1]], dtype=np.float32)
                                        success, rvec, tvec = cv2.solvePnP(model_pts, image_pts, cam_mtx, self.dist_coeffs)

                                        if success:
                                            # Project the mask points based on 3D pose
                                            mask_model_pts = np.array([ [p[0]*cal_w, p[1]*cal_h, 0] for p in mask.source_points ], dtype=np.float32)
                                            dst_pts_raw, _ = cv2.projectPoints(mask_model_pts, rvec, tvec, cam_mtx, self.dist_coeffs)
                                            dst_pts_px = dst_pts_raw.reshape(-1, 2)
                                            # Normalize back for transform_to_projector
                                            dst_pts_norm = dst_pts_px / [w_cam, h_cam]
                                        else:
                                            # Fallback to standard Homography (maps Normalized to Normalized)
                                            dst_pts_norm = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), curr_homography).reshape(-1, 2)
                                    else:
                                        # Fallback to standard Homography (maps Normalized to Normalized)
                                        dst_pts_norm = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), curr_homography).reshape(-1, 2)

                                    # Audio Scaling
                                    if self.audio_reactive_target == 'scale':
                                        center = np.mean(dst_pts_norm, axis=0)
                                        scale_factor = 1.0 + self.audio_bands[0] * 0.5
                                        dst_pts_norm = (dst_pts_norm - center) * scale_factor + center
                                else:
                                    # Tracking lost or not linked: stay at source points (already normalized)
                                    dst_pts_norm = src_pts

                                # Transform camera coordinates to internal render coordinates
                                dst_pts = self.transform_to_projector(dst_pts_norm, target_w=w, target_h=h)

                                # Warp video to the dynamic polygon
                                # Dimensions already known from cache
                                video_corners = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])

                                if len(dst_pts) == 4:
                                    dst_pts_warp = np.float32(dst_pts).reshape(4, 2)
                                elif len(dst_pts) >= 3:
                                    dst_pts_arr = np.array(dst_pts, dtype=np.float32)
                                    if not np.all(np.isfinite(dst_pts_arr)):
                                        continue
                                    min_x, min_y = np.min(dst_pts_arr, axis=0)
                                    max_x, max_y = np.max(dst_pts_arr, axis=0)
                                    dst_pts_warp = np.float32([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
                                else:
                                    continue

                                matrix, _ = cv2.findHomography(video_corners, dst_pts_warp)
                                if matrix is not None:
                                    # Use UMat for perspective warp - STAY ON GPU
                                    u_warped_cue = cv2.warpPerspective(u_effective_frame_cue, matrix, (w, h))

                                    # Draw mask
                                    curr_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
                                    draw_pts = np.int32(dst_pts)
                                    if mask.bezier_enabled and len(dst_pts) >= 3:
                                        pts = np.array(dst_pts)
                                        pts = np.vstack([pts, pts[0]])
                                        t_interp = np.linspace(0, 1, len(pts))
                                        t_new = np.linspace(0, 1, 100)
                                        f_x = interp1d(t_interp, pts[:, 0], kind='quadratic')
                                        f_y = interp1d(t_interp, pts[:, 1], kind='quadratic')
                                        draw_pts = np.stack([f_x(t_new), f_y(t_new)], axis=1).astype(np.int32)

                                    cv2.fillPoly(curr_mask_img, [draw_pts], (255, 255, 255))
                                    u_mask_img = cv2.UMat(curr_mask_img)
                                    if mask.feather > 0:
                                        k = int(mask.feather) | 1
                                        u_mask_img = cv2.boxFilter(u_mask_img, -1, (k, k))

                            elif mask.type == 'static' or (mask.type == 'dynamic' and not mask.is_linked):
                                if not mask.source_points:
                                    # Use UMat for resize
                                    u_warped_cue = cv2.resize(u_effective_frame_cue, (w, h))
                                    # Use cv2.rectangle on GPU for fill
                                    u_mask_img = cv2.UMat(h, w, cv2.CV_8UC3)
                                    cv2.rectangle(u_mask_img, (0, 0), (w, h), (255, 255, 255), -1)
                                else:
                                    src_pts_static = np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]])
                                    # mask.source_points is already normalized
                                    dst_pts = self.transform_to_projector(mask.source_points, target_w=w, target_h=h)

                                    if len(dst_pts) == 4:
                                        matrix = cv2.getPerspectiveTransform(src_pts_static, np.float32(dst_pts).reshape(4, 2))
                                    elif len(dst_pts) >= 3:
                                        dst_pts_arr = np.array(dst_pts, dtype=np.float32)
                                        min_x, min_y = np.min(dst_pts_arr, axis=0)
                                        max_x, max_y = np.max(dst_pts_arr, axis=0)
                                        bbox_pts = np.float32([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
                                        matrix = cv2.getPerspectiveTransform(src_pts_static, bbox_pts)
                                    else:
                                        matrix = None

                                    if matrix is not None:
                                        # Use UMat for perspective warp
                                        u_warped_cue = cv2.warpPerspective(u_effective_frame_cue, matrix, (w, h))

                                        curr_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
                                        cv2.fillPoly(curr_mask_img, [np.int32(dst_pts)], (255, 255, 255))
                                        u_mask_img = cv2.UMat(curr_mask_img)
                                        if mask.feather > 0:
                                            k = int(mask.feather) | 1
                                            u_mask_img = cv2.boxFilter(u_mask_img, -1, (k, k))

                            # Store in cache if static
                            if mask.type == 'static' and u_warped_cue is not None:
                                self.static_warp_cache[cache_key] = (frame_id, u_warped_cue, u_mask_img)

                        if u_warped_cue is not None and u_mask_img is not None:
                            # Performance: Keep blending on GPU
                            u_projector_output = self.blend_frames(u_projector_output, u_warped_cue, u_mask_img, mask.blend_mode)

                    # Draw outlines on projector during calibration/alignment
                    if self.show_camera_on_projector:
                        # Determine current points for outline
                        if mask.type == 'dynamic' and mask.is_linked and curr_homography is not None:
                            # curr_homography maps Normalized to Normalized
                            src_pts = np.float32(mask.source_points).reshape(-1, 1, 2)
                            draw_pts_norm = cv2.perspectiveTransform(src_pts, curr_homography).reshape(-1, 2)
                        else:
                            draw_pts_norm = np.array(mask.source_points)

                        draw_pts = self.transform_to_projector(draw_pts_norm, target_w=w, target_h=h).astype(np.int32)

                        if len(draw_pts) >= 3:
                            # Drawing on UMat is supported and efficient for small number of calls
                            cv2.polylines(u_projector_output, [draw_pts], True, (255, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(u_projector_output, f"{mask.tag or mask.name}", (int(draw_pts[0][0]), int(draw_pts[0][1] - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


                self.frame_count += 1
                now = time.time()
                if now - self.last_stats_time >= 1.0:
                    self.fps = self.frame_count / (now - self.last_stats_time)
                    self.frame_count = 0
                    self.last_stats_time = now

                if self.show_hud:
                    hud_color = (255, 0, 255) # Purple
                    cv2.putText(main_frame, f"FPS: {self.fps:.1f} (Track: {self.tracking_fps:.1f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
                    cv2.putText(main_frame, f"Conf: {curr_confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
                    cv2.putText(main_frame, f"BPM: {self.bpm:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
                    if self.throttle_level > 0.1:
                        cv2.putText(main_frame, f"THROTTLE: {self.throttle_level:.1%}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if self.auto_pilot:
                        cv2.putText(main_frame, "AUTO-PILOT ON", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw outlines on main_frame for UI feedback
                with QMutexLocker(self.mask_mutex):
                    for mask in self.masks:
                        if mask.visible:
                            # Determine current points for outline
                            if mask.is_linked and curr_homography is not None:
                                src_pts = np.float32(mask.source_points).reshape(-1, 1, 2)
                                try:
                                    # curr_homography maps Normalized to Normalized
                                    draw_pts_norm = cv2.perspectiveTransform(src_pts, curr_homography).reshape(-1, 2)
                                    draw_pts = (draw_pts_norm * [w_cam, h_cam]).astype(np.int32)
                                except:
                                    draw_pts = (np.array(mask.source_points) * [w_cam, h_cam]).astype(np.int32)
                            else:
                                draw_pts = (np.array(mask.source_points) * [w_cam, h_cam]).astype(np.int32)

                            if len(draw_pts) >= 2:
                                color = (0, 255, 0) if mask.is_linked else (0, 255, 255)
                                cv2.polylines(main_frame, [draw_pts], True, color, 2, cv2.LINE_AA)
                                if mask.is_linked:
                                    cv2.putText(main_frame, f"LINKED: {mask.name}", tuple(draw_pts[0]),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                rgb_main = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(QImage(rgb_main.data, w_cam, h_cam, w_cam * 3, QImage.Format_RGB888).copy())

                # Apply Master FX to the composition before warping
                # Apply Master FX to the composition before warping
                if self.master_active_fx or self.master_brightness != 0 or self.master_contrast != 0 or self.master_saturation != 100 or self.master_grain > 0 or self.master_bloom > 0 or self.master_fader < 1.0:

                    if self.master_active_fx:
                        master_proxy = Mask("Master", [], None)
                        master_proxy.active_fx = self.master_active_fx
                        master_proxy.tint_color = self.master_tint_color
                        # apply_fx already returns UMat if input is UMat
                        u_projector_output = self.apply_fx(u_projector_output, master_proxy, h=h, w=w)

                    # Apply Brightness/Contrast/Saturation on GPU
                    if self.master_brightness != 0 or self.master_contrast != 0:
                        alpha = (self.master_contrast + 100.0) / 100.0
                        beta = self.master_brightness
                        u_projector_output = cv2.convertScaleAbs(u_projector_output, alpha=alpha, beta=beta)

                    if self.master_saturation != 100:
                        u_hsv = cv2.cvtColor(u_projector_output, cv2.COLOR_BGR2HSV)
                        h_c, s_c, v_c = cv2.split(u_hsv)
                        s_c = cv2.multiply(s_c, self.master_saturation / 100.0)
                        u_hsv = cv2.merge([h_c, s_c, v_c])
                        u_projector_output = cv2.cvtColor(u_hsv, cv2.COLOR_HSV2BGR)

                    # Apply Grain (GPU accelerated)
                    if self.master_grain > 0:
                        # We still generate noise on CPU but can add on GPU
                        noise = np.random.randint(0, self.master_grain, (h, w, 3), dtype=np.uint8)
                        u_noise = cv2.UMat(noise)
                        u_projector_output = cv2.add(u_projector_output, u_noise)

                    # Apply Bloom (GPU accelerated)
                    if self.master_bloom > 0:
                        u_gray = cv2.cvtColor(u_projector_output, cv2.COLOR_BGR2GRAY)
                        _, u_mask = cv2.threshold(u_gray, 200, 255, cv2.THRESH_BINARY)
                        u_mask_3ch = cv2.merge([u_mask, u_mask, u_mask])
                        u_highlights = cv2.bitwise_and(u_projector_output, u_mask_3ch)
                        k = int(self.master_bloom / 2) | 1
                        u_bloom = cv2.GaussianBlur(u_highlights, (k, k), 0)
                        u_projector_output = cv2.addWeighted(u_projector_output, 1.0, u_bloom, 0.5, 0)

                    # Apply Master Fader on GPU
                    if self.master_fader < 1.0:
                        u_projector_output = cv2.convertScaleAbs(u_projector_output, alpha=self.master_fader)

                # Apply Projector Boundary Global Clip (in internal resolution)
                # Disable clipping during Wizard steps 0 and 1 to prevent interference with calibration
                is_wizard = (self.projector_width == 1280 and self.projector_height == 720) # Simple heuristic or pass flag

                # Global clipping removed as it was causing invisible masks and is redundant with per-mask shapes

                # Apply Performer Occlusion
                if self.occlusion_enabled and self.occlusion_mask is not None:
                    # Black out the performer on GPU
                    u_occl = cv2.UMat(self.occlusion_mask)
                    u_inv_mask = cv2.bitwise_not(u_occl)
                    u_inv_mask_3ch = cv2.merge([u_inv_mask, u_inv_mask, u_inv_mask])
                    u_projector_output = cv2.bitwise_and(u_projector_output, u_inv_mask_3ch)

                # 9-point grid warping (piecewise perspective optimization)
                # Optimized: Already using GPU (u_projector_output)

                if self._warp_is_identity:
                    if (w, h) != (w_proj, h_proj):
                        u_warped_output = cv2.resize(u_projector_output, (w_proj, h_proj), interpolation=cv2.INTER_LINEAR)
                    else:
                        u_warped_output = u_projector_output
                else:
                    if self.map_x is None or self.map_x.shape[:2] != (h_proj, w_proj) or self._warp_map_dirty:
                        self._update_warp_maps(w, h, w_proj, h_proj)

                    # Note: map_x and map_y remain on CPU for setup but remap happens on GPU if input is UMat
                    u_warped_output = cv2.remap(u_projector_output, self.map_x, self.map_y, cv2.INTER_LINEAR)

                # Convert to RGB on GPU before downloading
                u_rgb_proj = cv2.cvtColor(u_warped_output, cv2.COLOR_BGR2RGB)
                rgb_proj = u_rgb_proj.get()
                self.projector_frame_ready.emit(QImage(rgb_proj.data, w_proj, h_proj, w_proj * 3, QImage.Format_RGB888).copy())

                # 10. Update feedback mask for tracking (Projector Feed-Forward)
                # We save the grayscale version of what was just projected to help the tracker
                # ignore areas that are bright due to the projector.
                # stay on GPU for grayscale conversion
                with QMutexLocker(self.last_projected_frame_mutex):
                    u_gray_mask = cv2.cvtColor(u_warped_output, cv2.COLOR_BGR2GRAY)
                    self.last_projected_frame_for_masking = u_gray_mask.get()

                elapsed = time.time() - start_time

                # Auto-Throttle Logic
                if elapsed > frame_time:
                    self.throttle_level = min(1.0, self.throttle_level + 0.05)
                else:
                    self.throttle_level = max(0.0, self.throttle_level - 0.01)

                sleep_time = max(1, int((frame_time - elapsed) * 1000))
                QThread.msleep(sleep_time)
            except Exception as e:
                print(f"Critical Error in Rendering Loop: {e}")
                QThread.msleep(500)

        self.tracking_thread.stop()
        self.tracking_thread.wait()
        for player in self.video_players.values(): player.stop()

    def update_particles(self, tracked_points, h, w, w_cam, h_cam):
        if self.particle_preset == 'none':
            self.particles = []
            return

        # Performance: Throttle particle count
        effective_max = int(self.particle_max_count * (1.0 - self.throttle_level * 0.9))

        # Emit new particles
        if tracked_points and len(self.particles) < effective_max:
            # Transform tracked points to projector space once for emission
            # Tracked points are already normalized [0-1]
            norm_pts = np.array(tracked_points, dtype=np.float32)
            proj_pts = self.transform_to_projector(norm_pts, target_w=w, target_h=h)
            for pt in proj_pts:
                if np.random.random() > 0.7:
                    p = {
                        'x': float(pt[0]),
                        'y': float(pt[1]),
                        'vx': np.random.uniform(-2, 2),
                        'vy': np.random.uniform(-2, 2),
                        'life': 1.0,
                        'decay': np.random.uniform(0.01, 0.05)
                    }
                    if self.particle_preset == 'rain':
                        p['vy'] = np.random.uniform(5, 10)
                        p['vx'] = 0
                    elif self.particle_preset == 'dust':
                        p['vy'] = np.random.uniform(-1, 0.5)
                    self.particles.append(p)

        # Update existing
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= p['decay']
            if p['life'] <= 0 or p['x'] < 0 or p['x'] >= w or p['y'] < 0 or p['y'] >= h:
                self.particles.remove(p)

    def draw_particles(self, frame):
        if self.particle_preset == 'none': return
        h, w = frame.shape[:2]
        res_scale = np.sqrt((w * h) / (1280 * 720))
        for p in self.particles:
            color = (0, 255, 0) if self.particle_preset == 'rain' else (255, 255, 200)
            alpha = int(p['life'] * 255)
            p_size = max(1, int(2 * res_scale))
            # Use anti-aliasing for smoother particles
            cv2.circle(frame, (int(p['x']), int(p['y'])), p_size, color, -1, cv2.LINE_AA)

    def get_vj_generator(self, pattern, h, w):
        # Performance: Render generators at lower resolution if throttled
        orig_w, orig_h = w, h
        if self.throttle_level > 0.8:
            w, h = w // 4, h // 4
        elif self.throttle_level > 0.4:
            w, h = w // 2, h // 2

        if self.generator_buffer is None or self.generator_buffer.shape[:2] != (h, w):
            self.generator_buffer = np.zeros((h, w, 3), dtype=np.uint8)

        self.generator_buffer.fill(0)
        frame = self.generator_buffer

        t = time.time() * (self.bpm / 60.0)

        if pattern == 'grid':
            # Vectorized Grid using NumPy
            spacing = 40
            offset_x = int((t * 20) % spacing)
            offset_y = int((t * 15) % spacing)

            # Draw vertical lines
            x_indices = np.arange(offset_x, w, spacing)
            if len(x_indices) > 0:
                frame[:, x_indices, 1] = 255 # Green Channel

            # Draw horizontal lines
            y_indices = np.arange(offset_y, h, spacing)
            if len(y_indices) > 0:
                frame[y_indices, :, 1] = 255

        elif pattern == 'scan':
            # Vectorized Scan Line
            y = int((t * 150) % h)
            thickness = 10
            y1 = max(0, y - thickness)
            y2 = min(h, y + thickness)
            frame[y1:y2, :, :] = 255
        elif pattern == 'radial':
            center = (w // 2, h // 2)
            for r in range(int((t * 50) % 100), max(w, h), 100):
                cv2.circle(frame, center, r, (255, 0, 0), 3)
        elif pattern == 'tunnel':
            center = (w // 2, h // 2)
            for i in range(10):
                r = int(((t + i * 0.1) % 1.0) * max(w, h))
                cv2.rectangle(frame, (center[0]-r, center[1]-r), (center[0]+r, center[1]+r), (255, 255, 255), 2)
        elif pattern == 'plasma':
            # Fast plasma-like effect using sine waves
            if self.cached_plasma_grid is None or self.cached_plasma_grid[0].shape != (h, w):
                x = np.linspace(0, 10, w, dtype=np.float32)
                y = np.linspace(0, 10, h, dtype=np.float32)
                self.cached_plasma_grid = np.meshgrid(x, y)

            X, Y = self.cached_plasma_grid
            res = np.sin(X + t) + np.sin(Y + t*0.5) + np.sin((X + Y + t)*0.5)
            res = ((res + 3) / 6 * 255).astype(np.uint8)
            frame = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        elif pattern == 'vortex':
            center = (w // 2, h // 2)
            # Semi-vectorized: precalculate angles
            angles = np.radians(np.arange(0, 360, 15) + t * 80)
            cos_a = np.cos(angles)
            sin_a = np.sin(angles)
            max_d = max(w, h)
            for i in range(len(angles)):
                end_x = int(center[0] + max_d * cos_a[i])
                end_y = int(center[1] + max_d * sin_a[i])
                cv2.line(frame, center, (end_x, end_y), (255, 255, 0), 2, cv2.LINE_AA)
        elif pattern == 'waves':
            # Vectorized Waves using NumPy
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)

            # Simple wave interference
            wave1 = np.sin(X * 0.05 + t * 2)
            wave2 = np.sin(Y * 0.04 + t * 1.5)
            res = ((wave1 + wave2 + 2) / 4 * 255).astype(np.uint8)
            frame = cv2.applyColorMap(res, cv2.COLORMAP_OCEAN)
        elif pattern == 'polytunnel':
            center = (w // 2, h // 2)
            for i in range(12):
                z = (t * 0.5 + i * 0.1) % 1.2
                if z < 0.1: continue
                size = int((1.0 / z) * 50)
                rot = t * 2 + i * 0.5
                pts = []
                num_sides = 6
                for s in range(num_sides):
                    ang = rot + s * (2 * np.pi / num_sides)
                    pts.append([center[0] + size * np.cos(ang), center[1] + size * np.sin(ang)])
                cv2.polylines(frame, [np.array(pts, np.int32)], True, (200, 0, 255), 2, cv2.LINE_AA)
        elif pattern == 'stardust':
            # 3D-like forward motion of polygonal stars
            for i in range(30):
                seed = (i * 12345)
                speed = 0.2 + (seed % 10) / 10.0
                offset = (t * speed + i * 0.03) % 1.0
                if offset < 0.01: continue
                # Project from center
                size = int(offset * 20)
                dist = offset * max(w, h) * 0.8
                ang = (seed % 360) * (np.pi / 180)
                x = int(w // 2 + dist * np.cos(ang))
                y = int(h // 2 + dist * np.sin(ang))

                # Small triangle star
                star_pts = []
                for s in range(3):
                    sang = t * 5 + s * (2 * np.pi / 3)
                    star_pts.append([x + size * np.cos(sang), y + size * np.sin(sang)])
                cv2.polylines(frame, [np.array(star_pts, np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        elif pattern == 'hypergrid':
            # Perspective moving grid
            horizon = h // 2
            num_lines = 15
            for i in range(num_lines):
                # Vertical perspective lines
                x_start = int(w // 2 + (i - num_lines // 2) * (w // 20))
                cv2.line(frame, (w // 2, horizon), (x_start * 2 - w // 2, h), (100, 0, 200), 1, cv2.LINE_AA)

            # Horizontal moving lines
            for i in range(8):
                y_off = (t * 50 + i * (h // 8)) % (h // 2)
                y = horizon + int(y_off)
                # Map y to width (simulated perspective)
                scale = (y - horizon) / (h // 2.0)
                if scale > 0:
                    width = int(w * scale)
                    cv2.line(frame, (w // 2 - width // 2, y), (w // 2 + width // 2, y), (150, 0, 255), 1, cv2.LINE_AA)
        elif pattern == 'prism_move':
            # Floating prisms moving through space
            for i in range(5):
                path_t = t * 0.2 + i * 0.4
                x = int((np.cos(path_t * 3) * 0.4 + 0.5) * w)
                y = int((np.sin(path_t * 2) * 0.4 + 0.5) * h)
                z_scale = 0.5 + 0.5 * np.sin(path_t * 5)
                size = int(30 + 70 * z_scale)

                # Draw a hexagonal prism wireframe
                pts_top = []
                pts_bot = []
                num_sides = 6
                rot = t + i
                for s in range(num_sides):
                    ang = rot + s * (2 * np.pi / num_sides)
                    pts_top.append([x + size * np.cos(ang), y + size * np.sin(ang)])
                    pts_bot.append([x + (size+20) * np.cos(ang), y + (size+20) * np.sin(ang)])

                color = (255, 0, 255)
                cv2.polylines(frame, [np.array(pts_top, np.int32)], True, color, 2, cv2.LINE_AA)
                cv2.polylines(frame, [np.array(pts_bot, np.int32)], True, color, 1, cv2.LINE_AA)
                for s in range(num_sides):
                    cv2.line(frame, tuple(np.int32(pts_top[s])), tuple(np.int32(pts_bot[s])), color, 1, cv2.LINE_AA)
        elif pattern == 'nebula':
            # Multi-layered noise nebula
            if self.cached_nebula_grid is None or self.cached_nebula_grid[0].shape != (h, w):
                x = np.linspace(0, 4, w, dtype=np.float32)
                y = np.linspace(0, 4, h, dtype=np.float32)
                self.cached_nebula_grid = np.meshgrid(x, y)

            X, Y = self.cached_nebula_grid

            # Layer 1
            n1 = np.sin(X + t * 0.2) * np.cos(Y - t * 0.3)
            # Layer 2 (Faster, smaller)
            n2 = np.sin(X * 2 - t * 0.5) * np.sin(Y * 2 + t * 0.4)

            res = ((n1 + n2 + 2) / 4 * 255).astype(np.uint8)
            frame = cv2.applyColorMap(res, cv2.COLORMAP_MAGMA)
            # Darken for space feel
            frame = (frame * 0.6).astype(np.uint8)
        elif pattern == 'blackhole':
            center = (w // 2, h // 2)
            # Central event horizon
            cv2.circle(frame, center, int(30 + 5 * np.sin(t*5)), (0, 0, 0), -1)
            # Accretion disk particles
            for i in range(40):
                angle = (t * 2 + i * (2 * np.pi / 40))
                dist = (1.5 - ((t * 0.5 + i * 0.05) % 1.0)) * max(w, h) * 0.3 + 40
                if dist < 40: continue

                # Whirlpool effect
                x = int(center[0] + dist * np.cos(angle + 100/dist))
                y = int(center[1] + dist * np.sin(angle + 100/dist))

                color = (200, 100, 255)
                cv2.circle(frame, (x, y), 2, color, -1, cv2.LINE_AA)
                # Connecting lines for flow
                nx = int(center[0] + (dist-10) * np.cos(angle + 100/(dist-10)))
                ny = int(center[1] + (dist-10) * np.sin(angle + 100/(dist-10)))
                cv2.line(frame, (x, y), (nx, ny), (100, 50, 150), 1, cv2.LINE_AA)

        if frame.shape[:2] != (orig_h, orig_w):
            return cv2.resize(frame, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return frame.copy()

    def get_generative_frame(self, h, w):
        self.noise_offset += 0.05
        # Create a moving cloud pattern
        x = np.linspace(0, 5, w)
        y = np.linspace(0, 5, h)
        X, Y = np.meshgrid(x, y)
        pattern = np.sin(X + self.noise_offset) * np.cos(Y + self.noise_offset * 0.5)
        pattern = ((pattern + 1) * 127).astype(np.uint8)
        # Add some color
        colored = cv2.applyColorMap(pattern, cv2.COLORMAP_MAGMA)
        return colored

    def blend_frames(self, base, overlay, mask_img, mode):
        # Performance: Use GPU (UMat) for blending
        is_umat = isinstance(base, cv2.UMat)
        u_base = base if is_umat else cv2.UMat(base)
        u_overlay = overlay if isinstance(overlay, cv2.UMat) else cv2.UMat(overlay)
        u_mask = mask_img if isinstance(mask_img, cv2.UMat) else cv2.UMat(mask_img)

        # Optimized integer-based blending on GPU
        if mode == 'normal':
            # Full alpha blending: res = overlay * (mask/255) + base * (1 - mask/255)
            u_temp1 = cv2.multiply(u_overlay, u_mask, scale=1.0/255.0)
            u_inv_mask = cv2.bitwise_not(u_mask)
            u_temp2 = cv2.multiply(u_base, u_inv_mask, scale=1.0/255.0)
            u_res = cv2.add(u_temp1, u_temp2)
        elif mode == 'add':
            u_temp1 = cv2.multiply(u_overlay, u_mask, scale=1.0/255.0)
            u_res = cv2.add(u_base, u_temp1)
        elif mode == 'screen':
            u_temp1 = cv2.multiply(u_overlay, u_mask, scale=1.0/255.0)
            u_inv_base = cv2.bitwise_not(u_base)
            u_inv_temp1 = cv2.bitwise_not(u_temp1)
            u_res = cv2.bitwise_not(cv2.multiply(u_inv_base, u_inv_temp1, scale=1.0/255.0))
        elif mode == 'multiply':
            u_temp1 = cv2.multiply(u_overlay, u_mask, scale=1.0/255.0)
            u_inv_mask = cv2.bitwise_not(u_mask)
            u_res = cv2.multiply(u_base, cv2.add(u_temp1, u_inv_mask), scale=1.0/255.0)
        else:
            return base

        return u_res if is_umat else u_res.get()

    def normalize_points_to_reference(self, points):
        """Converts points from current frame space to marker-config reference space."""
        if self.last_homography is not None and len(points) > 0:
            try:
                inv_h = np.linalg.inv(self.last_homography)
                pts = np.float32(points).reshape(-1, 1, 2)
                ref_pts = cv2.perspectiveTransform(pts, inv_h).reshape(-1, 2)
                return [ (float(p[0]), float(p[1])) for p in ref_pts]
            except Exception as e:
                print(f"Normalization error: {e}")
        return points

    def stop(self): self._running = False

    def run_boundary_detection(self):
        # We MUST keep resolution consistent across all setup steps
        # Use (9999,9999) as a trigger for max available res in TrackingThread
        self.requested_camera_res = (9999, 9999)
        self._boundary_step = 0
        self._boundary_captures = []
        self._sls_curr_wait = 0
        self._run_boundary_detection_flag = True

    def stop_calibration(self):
        self._run_calibration_flag = False
        self._run_sls_flag = False
        self._run_boundary_detection_flag = False
        self.show_calibration_pattern = False
        self.show_calibration_verify = False
        # Do NOT revert resolution to avoid FOV shift/crop
        # FOV stability is paramount for projection mapping

    def run_auto_calibration(self):
        self.requested_camera_res = (9999, 9999)
        self._run_calibration_flag = True

    def run_one_click_sync(self):
        # The scan room process now also updates the boundary mask
        self.run_room_scan()

    def run_room_scan(self):
        self.requested_camera_res = (9999, 9999) # Request max FOV
        self._sls_patterns_x, self._sls_patterns_y = generate_gray_code_patterns(self.projector_width, self.projector_height)
        self._sls_step = 0
        self._sls_captures_x = []
        self._sls_captures_y = []
        self._sls_curr_wait = 0
        self._run_sls_flag = True

    def warp_full_frame_to_projector(self, frame, w_cam, h_cam, w_target, h_target):
        if self.h_c2p is not None:
            # h_c2p maps to native projector resolution
            # We need to scale it to target resolution
            scale_w = w_target / self.projector_width
            scale_h = h_target / self.projector_height
            S = np.array([[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]], dtype=np.float32)
            M = S @ self.h_c2p
            return cv2.warpPerspective(frame, M, (w_target, h_target))
        return cv2.resize(frame, (w_target, h_target))

    def transform_to_projector(self, pts, target_w=None, target_h=None):
        """Transform normalized camera coordinates [0-1] to projector internal coordinates."""
        if target_w is None: target_w = self.projector_width
        if target_h is None: target_h = self.projector_height

        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)

        # Map to Calibration Resolution space for internal math
        if self.calibration_camera_res:
            cal_w, cal_h = self.calibration_camera_res
        else:
            # Fallback if not calibrated: use current res
            cal_w, cal_h = self._current_camera_res if self._current_camera_res[0] > 0 else (9999, 9999)

        # Ensure we are using absolute pixels relative to the calibration FOV
        pts_cal = pts_arr * [cal_w, cal_h]

        # Prefer Dense LUT if available
        if self.sls_lut_x is not None:
            try:
                res = []
                for p in pts_cal:
                    ix, iy = int(round(p[0])), int(round(p[1]))
                    if 0 <= ix < cal_w and 0 <= iy < cal_h and self.sls_valid_mask[iy, ix]:
                        # SLS LUT stores normalized [0-1] projector coords
                        res.append([self.sls_lut_x[iy, ix] * target_w, self.sls_lut_y[iy, ix] * target_h])
                    else:
                        # Fallback to RBF for out-of-bounds or invalid pixels
                        if self.rbf_x:
                            # RBF outputs normalized [0-1] projector coords
                            rx = self.rbf_x(p.reshape(1, 2))[0] * target_w
                            ry = self.rbf_y(p.reshape(1, 2))[0] * target_h
                            res.append([rx, ry])
                        else:
                            # Proportional fallback
                            res.append([p[0] * (target_w / cal_w), p[1] * (target_h / cal_h)])
                return np.array(res).reshape(-1, 2)
            except:
                pass

        if self.rbf_x is not None:
            try:
                # RBF outputs normalized [0-1] projector coords
                tx = self.rbf_x(pts_cal) * target_w
                ty = self.rbf_y(pts_cal) * target_h
                return np.stack([tx, ty], axis=1).reshape(-1, 2)
            except:
                pass

        if self.h_c2p is not None:
            try:
                pts_reshaped = np.float32(pts_cal).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts_reshaped, self.h_c2p)
                res = transformed.reshape(-1, 2)
                # Homography outputs normalized [0-1] projector coords
                res[:, 0] *= target_w
                res[:, 1] *= target_h
                return res
            except:
                pass

        # Fallback: Simple proportional scaling if not calibrated
        # Since pts_arr is normalized, we just scale by target dimensions
        return pts_arr * [target_w, target_h]

    def set_warp_points(self, points, res=None):
        self.warp_points = points
        if res: self.warp_grid_res = res
        self._warp_map_dirty = True

        # Check if identity
        self._warp_is_identity = True
        res = self.warp_grid_res
        for i, p in enumerate(self.warp_points):
            expected_x = (i % res) / (res - 1)
            expected_y = (i // res) / (res - 1)
            if abs(p[0] - expected_x) > 0.005 or abs(p[1] - expected_y) > 0.005:
                self._warp_is_identity = False
                break
    def cleanup_resources(self):
        # Aggressively stop unused players
        current_cues = {mask.video_path for mask in self.masks if mask.video_path}
        fade_cues = {f['prev_path'] for f in self.fades.values()}
        needed_cues = current_cues.union(fade_cues)

        with QMutexLocker(self.player_mutex):
            for path in list(self.video_players.keys()):
                if path not in needed_cues:
                    print(f"Purging player for: {path}")
                    self.video_players[path].stop()
                    del self.video_players[path]

        # Cleanup trail buffers for removed masks
        mask_ids = {id(mask) for mask in self.masks}
        for mid in list(self.trail_buffers.keys()):
            if mid not in mask_ids:
                del self.trail_buffers[mid]

    def set_masks(self, masks):
        with QMutexLocker(self.mask_mutex):
            # Sort masks by Z-order once when they are updated
            self.masks = sorted(masks, key=lambda m: m.z_order)
        self.update_video_speeds()
        self.cleanup_resources()

    def set_pnp_enabled(self, enabled):
        self.pnp_enabled = enabled

    def set_occlusion_enabled(self, enabled):
        self.occlusion_enabled = enabled
