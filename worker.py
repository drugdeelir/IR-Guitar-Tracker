
import cv2
import numpy as np
import time
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker
from PyQt5.QtGui import QImage
from itertools import combinations
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
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.video_path)
                    if not self.cap.isOpened():
                        time.sleep(1.0)
                        continue

                ret, frame = self.cap.read()
                if not ret:
                    # Robust looping: try to seek back to start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

                    if not ret:
                        # If seek failed, try re-opening the capture
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.video_path)
                        ret, frame = self.cap.read()

                    if not ret:
                        # Still no frame? Sleep and try again in next iteration
                        time.sleep(1.0)
                        continue

                with QMutexLocker(self.mutex):
                    self.latest_frame = frame
                    self.frame_id += 1

                time.sleep(max(0.001, 1.0 / (self.fps * self.playback_speed)))
        except Exception as e:
            print(f"VideoPlayer Critical Error ({self.video_path}): {e}")

    def get_frame(self):
        with QMutexLocker(self.mutex):
            if self.latest_frame is None:
                return None, 0
            return self.latest_frame.copy(), self.frame_id

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
            start_time = time.time()

            if self.worker._camera_changed or self.worker.requested_camera_res != self.worker._current_camera_res:
                if self.worker._camera_changed:
                    if main_cap: main_cap.release()
                    main_cap = cv2.VideoCapture(self.worker.video_source)
                    self.worker._camera_changed = False

                if main_cap and main_cap.isOpened():
                    w_req, h_req = self.worker.requested_camera_res
                    main_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_req)
                    main_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_req)
                    # Read back actual resolution
                    act_w = int(main_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    act_h = int(main_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.worker._current_camera_res = (act_w, act_h)

                    # If we requested max res, update the request to match reality
                    # to prevent constant re-initialization
                    if w_req > 5000:
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
            tracked_points = self.worker.get_tracked_points(main_frame)

            with QMutexLocker(self.worker.tracking_mutex):
                self.worker.last_tracked_points_internal = tracked_points
                self.worker.last_homography_internal = self.worker.last_homography
                self.worker.confidence_internal = self.worker.confidence
                self.worker.tracking_frame_count += 1

            self.worker.trackers_detected.emit(len(tracked_points))
            self.worker.trackers_ready.emit(tracked_points)

            if len(tracked_points) >= 2:
                current_dist = np.linalg.norm(np.array(tracked_points[0]) - np.array(tracked_points[1]))
                if self.worker._calibrate_depth_flag:
                    self.worker.baseline_distance = current_dist
                    self.worker._calibrate_depth_flag = False

                if self.worker.baseline_distance > 0:
                    self.worker.proximity_val = current_dist / self.worker.baseline_distance

            # HUD Data preparation (camera side)
            with QMutexLocker(self.worker.latest_main_frame_mutex):
                self.worker.latest_main_frame = main_frame.copy()
                self.worker.latest_main_frame_id += 1
                self.worker.latest_tracked_points_for_ui = tracked_points

            # Handle Calibration Flags (some need to run in camera thread)
            if self.worker._capture_still_frame_flag:
                rgb = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                self.worker.still_frame_ready.emit(QImage(rgb.data, w_cam, h_cam, w_cam * 3, QImage.Format_RGB888).copy(), tracked_points)
                self.worker._capture_still_frame_flag = False

            elapsed = time.time() - start_time
            sleep_time = max(1, int((frame_time - elapsed) * 1000))
            QThread.msleep(sleep_time)

        if main_cap:
            main_cap.release()

class Worker(QObject):
    frame_ready = pyqtSignal(QImage)
    projector_frame_ready = pyqtSignal(QImage)
    still_frame_ready = pyqtSignal(QImage, list)
    trackers_detected = pyqtSignal(int)
    trackers_ready = pyqtSignal(list)
    camera_error = pyqtSignal(int)
    system_warning = pyqtSignal(str)
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
        self.requested_camera_res = (640, 480)
        self._current_camera_res = (640, 480)
        self.baseline_distance = 0
        self.depth_sensitivity = 1.0
        self._calibrate_depth_flag = False
        self._capture_still_frame_flag = False
        self.marker_config = None
        self.bpm = 120.0
        self.last_tracked_points = None
        self.last_tracked_points_internal = []
        self.last_homography = None
        self.last_homography_internal = None
        self.confidence_internal = 0.0
        self.marker_fingerprint = []
        self.roi_padding = 50
        self.tracking_mutex = QMutex()

        # Smoothing and Confidence
        self.kalman_filters = []
        self.smoothed_points = None
        self.smoothing_factor = 0.5
        self.confidence = 0.0
        self.confidence_gain = 0.2
        self.confidence_decay = 0.1

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
        self._sls_wait_frames = 5
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
        self.throttle_level = 0.0 # 0.0 to 1.0 (degrade quality)

        # Caching
        self.static_warp_cache = {} # mask_id -> (video_frame_id, warped_frame, mask_img)

        # Reusable Buffers
        self.projector_buffer = None
        self.mask_buffer = None
        self.latest_main_frame = None
        self.latest_main_frame_mutex = QMutex()
        self.latest_main_frame_id = 0
        self._last_captured_frame_id = -1
        self.latest_tracked_points_for_ui = []
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        self.cached_plasma_grid = None
        self.cached_nebula_grid = None
        self.generator_buffer = None
        self.blend_temp1 = None
        self.blend_temp2 = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

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

    def init_kalman(self, count):
        self.kalman_filters = []
        for _ in range(count):
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
            self.kalman_filters.append(kf)

    def set_marker_points(self, points):
        with QMutexLocker(self.tracking_mutex):
            self.marker_config = [ (p.x(), p.y()) for p in points]
            if len(self.marker_config) > 1:
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
            self.marker_fingerprint = []
            self.last_tracked_points = None
            self.smoothed_points = None

    def set_auto_threshold(self, enabled):
        self.auto_threshold = enabled

    def set_smoothing(self, value):
        self.smoothing_factor = value

    def capture_still_frame(self):
        self._capture_still_frame_flag = True

    def calibrate_depth(self):
        self._calibrate_depth_flag = True

    def set_h_c2p(self, matrix_list):
        if matrix_list is None:
            self.h_c2p = None
            self.rbf_x = None
            self.rbf_y = None
        else:
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

    def apply_fx(self, frame, mask, live_only=False):
        mask_id = id(mask)
        lfo_val = 1.0
        lfo_enabled = mask.fx_params.get('lfo_enabled')
        if lfo_enabled:
            lfo_val = self.get_lfo_value(mask)

        if not mask.active_fx and mask.design_overlay == 'none':
            return frame

        # Performance: skip FX if heavily throttled and not essential
        if self.throttle_level > 0.9 and not live_only:
             # Just return original if we are dying
             return frame

        live_fx = {'strobe', 'hue_cycle', 'glitch', 'trails', 'feedback', 'matrix', 'ooze', 'scanline', 'vhs'}

        # Skip heavy procedural FX if throttled
        if self.throttle_level > 0.7:
             live_fx.difference_update({'matrix', 'ooze', 'vhs'})

        # Since frame_cue is already a copy from VideoPlayer or a generator,
        # we can modify it in-place to save performance.
        processed = frame
        h, w = processed.shape[:2]

        if not live_only:
            if 'mirror_h' in mask.active_fx:
                left = processed[:, :w//2]
                processed[:, w//2:] = cv2.flip(left, 1)
            if 'mirror_v' in mask.active_fx:
                top = processed[:h//2, :]
                processed[h//2:, :] = cv2.flip(top, 0)
            if 'kaleidoscope' in mask.active_fx:
                # 4-way symmetry
                quad = processed[:h//2, :w//2]
                processed[:h//2, w//2:] = cv2.flip(quad, 1)
                processed[h//2:, :w//2] = cv2.flip(quad, 0)
                processed[h//2:, w//2:] = cv2.flip(quad, -1)

        if 'strobe' in mask.active_fx:
            trigger = False
            if self.audio_reactive_target == 'strobe':
                if self.audio_bands[0] > 0.6: # Bass trigger
                    trigger = True
            else:
                period = 60.0 / self.bpm
                if (time.time() % period) < (period / 2.0):
                    trigger = True

            if trigger:
                processed = np.zeros_like(processed)

        if 'rgb_shift' in mask.active_fx:
            mod = 1.0
            if self.proximity_mode == 'rgb_shift': mod = self.proximity_val
            if 'rgb_shift' in self.audio_param_mappings:
                mod *= (self.audio_bands[self.audio_param_mappings['rgb_shift']] * 5)

            shift = int(10 * mod * (lfo_val if mask.fx_params.get('lfo_target') == 'rgb_shift' else 1.0))
            b, g, r = cv2.split(processed)
            b = np.roll(b, shift, axis=1)
            r = np.roll(r, -shift, axis=1)
            processed = cv2.merge([b, g, r])

        if 'glitch' in mask.active_fx:
            mod = 1.0
            if self.proximity_mode == 'glitch': mod = self.proximity_val
            if self.audio_reactive_target == 'glitch': mod *= (self.audio_bands[0] * 2)

            for _ in range(int(3 * mod)):
                y = np.random.randint(0, h-10)
                sh = np.random.randint(-20, 20)
                processed[y:y+10, :] = np.roll(processed[y:y+10, :], sh, axis=1)

        if 'trails' in mask.active_fx:
            if mask_id in self.trail_buffers:
                processed = cv2.addWeighted(processed, 0.4, self.trail_buffers[mask_id], 0.6, 0)
            self.trail_buffers[mask_id] = processed.copy()

        if 'feedback' in mask.active_fx:
            if mask_id in self.trail_buffers:
                # Zoom in slightly and rotate last frame
                prev = self.trail_buffers[mask_id]
                M = cv2.getRotationMatrix2D((w//2, h//2), 1, 1.02)
                prev = cv2.warpAffine(prev, M, (w, h))
                processed = cv2.addWeighted(processed, 0.7, prev, 0.3, 0)
            self.trail_buffers[mask_id] = processed.copy()

        if 'hue_cycle' in mask.active_fx:
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV).astype(np.float32)
            shift = (time.time() * (self.bpm / 60.0) * 30) % 180
            if mask.fx_params.get('lfo_target') == 'hue':
                shift *= lfo_val
            hsv[:,:,0] = (hsv[:,:,0] + shift) % 180
            processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if not live_only:
            if 'blur' in mask.active_fx:
                mod = 1.0
                if 'blur' in self.audio_param_mappings:
                    mod *= (self.audio_bands[self.audio_param_mappings['blur']] * 3)

                # Performance: Throttle blur kernel size
                base_size = 15 * (1.0 - self.throttle_level * 0.8)
                ksize = int(base_size * mod * (lfo_val if mask.fx_params.get('lfo_target') == 'blur' else 1.0))
                if ksize % 2 == 0: ksize += 1
                if ksize > 0:
                    if self.throttle_level > 0.5:
                        # Faster blur
                        processed = cv2.blur(processed, (max(1, ksize), max(1, ksize)))
                    else:
                        processed = cv2.GaussianBlur(processed, (max(1, ksize), max(1, ksize)), 0)

            if 'invert' in mask.active_fx:
                processed = cv2.bitwise_not(processed)

            if 'edges' in mask.active_fx:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            if 'tint' in mask.active_fx:
                mod = 1.0
                if 'tint' in self.audio_param_mappings:
                    mod *= (self.audio_bands[self.audio_param_mappings['tint']] * 3)

                alpha = 0.3 * mod * (lfo_val if mask.fx_params.get('lfo_target') == 'tint' else 1.0)
                tint = np.full_like(processed, mask.tint_color)
                processed = cv2.addWeighted(processed, 1.0 - alpha, tint, alpha, 0)

            if 'duotone' in mask.active_fx:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                comp = (255 - mask.tint_color[0], 255 - mask.tint_color[1], 255 - mask.tint_color[2])
                lut = np.zeros((256, 1, 3), dtype=np.uint8)
                for i in range(256):
                    a = i / 255.0
                    lut[i, 0, 0] = int(comp[0] * (1 - a) + mask.tint_color[0] * a)
                    lut[i, 0, 1] = int(comp[1] * (1 - a) + mask.tint_color[1] * a)
                    lut[i, 0, 2] = int(comp[2] * (1 - a) + mask.tint_color[2] * a)
                processed = cv2.LUT(cv2.merge([gray, gray, gray]), lut)

            if 'chromakey' in mask.active_fx:
                # Simple green screen removal
                hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
                lower = np.array([40, 40, 40])
                upper = np.array([80, 255, 255])
                m = cv2.inRange(hsv, lower, upper)
                processed[m > 0] = [0, 0, 0]

            if 'pixelate' in mask.active_fx:
                # Scale down and back up
                div = 16
                small = cv2.resize(processed, (max(1, w // div), max(1, h // div)), interpolation=cv2.INTER_NEAREST)
                processed = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

            if 'chroma_aberration' in mask.active_fx:
                shift = 5
                b, g, r = cv2.split(processed)
                b = np.roll(b, shift, axis=0)
                r = np.roll(r, -shift, axis=1)
                processed = cv2.merge([b, g, r])

        # Live FX (always applied)
        if any(fx in live_fx for fx in mask.active_fx):
            if 'strobe' in mask.active_fx:
                trigger = False
                if self.audio_reactive_target == 'strobe':
                    if self.audio_bands[0] > 0.6: # Bass trigger
                        trigger = True
                else:
                    period = 60.0 / self.bpm
                    if (time.time() % period) < (period / 2.0):
                        trigger = True

                if trigger:
                    processed = np.zeros_like(processed)

            if 'rgb_shift' in mask.active_fx:
                mod = 1.0
                if self.proximity_mode == 'rgb_shift': mod = self.proximity_val
                if 'rgb_shift' in self.audio_param_mappings:
                    mod *= (self.audio_bands[self.audio_param_mappings['rgb_shift']] * 5)

                shift = int(10 * mod * (lfo_val if mask.fx_params.get('lfo_target') == 'rgb_shift' else 1.0))
                b, g, r = cv2.split(processed)
                b = np.roll(b, shift, axis=1)
                r = np.roll(r, -shift, axis=1)
                processed = cv2.merge([b, g, r])

            if 'glitch' in mask.active_fx:
                mod = 1.0
                if self.proximity_mode == 'glitch': mod = self.proximity_val
                if self.audio_reactive_target == 'glitch': mod *= (self.audio_bands[0] * 2)

                for _ in range(int(3 * mod)):
                    y = np.random.randint(0, h-10)
                    sh = np.random.randint(-20, 20)
                    processed[y:y+10, :] = np.roll(processed[y:y+10, :], sh, axis=1)

            if 'trails' in mask.active_fx:
                if mask_id in self.trail_buffers:
                    processed = cv2.addWeighted(processed, 0.4, self.trail_buffers[mask_id], 0.6, 0)
                self.trail_buffers[mask_id] = processed.copy()

            if 'feedback' in mask.active_fx:
                if mask_id in self.trail_buffers:
                    # Zoom in slightly and rotate last frame
                    prev = self.trail_buffers[mask_id]
                    M = cv2.getRotationMatrix2D((w//2, h//2), 1, 1.02)
                    prev = cv2.warpAffine(prev, M, (w, h))
                    processed = cv2.addWeighted(processed, 0.7, prev, 0.3, 0)
                self.trail_buffers[mask_id] = processed.copy()

            if 'hue_cycle' in mask.active_fx:
                hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV).astype(np.float32)
                shift = (time.time() * (self.bpm / 60.0) * 30) % 180
                if mask.fx_params.get('lfo_target') == 'hue':
                    shift *= lfo_val
                hsv[:,:,0] = (hsv[:,:,0] + shift) % 180
                processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            if 'ooze' in mask.active_fx:
                t = time.time()
                for x in range(0, w, 20):
                    length = int((np.sin(t + x * 0.1) * 0.5 + 0.5) * h)
                    cv2.line(processed, (x, 0), (x, length), (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.circle(processed, (x, length), 5, (0, 255, 100), -1)

            if 'matrix' in mask.active_fx:
                t = time.time()
                for x in range(0, w, 15):
                    speed = 1.0 + (np.sin(x) * 0.5 + 0.5)
                    y = int((t * 200 * speed) % (h + 100)) - 100
                    for i in range(10):
                        alpha = (10 - i) / 10.0
                        color = (0, int(255 * alpha), 0)
                        cv2.putText(processed, chr(np.random.randint(33, 126)), (x, y - i*15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if 'vhs' in mask.active_fx:
                jitter = np.random.randint(-5, 5)
                processed = np.roll(processed, jitter, axis=1)
                y = np.random.randint(0, h)
                processed[y:y+2, :] = cv2.add(processed[y:y+2, :], (50, 50, 50, 0))
                b, g, r = cv2.split(processed)
                r = np.roll(r, 3, axis=1)
                processed = cv2.merge([b, g, r])

            if 'scanline' in mask.active_fx:
                t = time.time()
                y_pos = int((t * 100) % h)
                cv2.line(processed, (0, y_pos), (w, y_pos), (255, 255, 255), 1)
                processed[::2, :] = cv2.convertScaleAbs(processed[::2, :], alpha=0.7)

        return processed

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
                cv2.circle(mask, (int(center[0] + x * (h//50)), int(center[1] + y * (h//50))), h // 40, 255, -1)
        else:
            mask.fill(255)

        self._design_cache[cache_key] = mask
        return mask

    def get_tracked_points(self, frame):
        # Already called from TrackingThread, but let's ensure we use tracking_mutex for shared state
        with QMutexLocker(self.tracking_mutex):
            marker_cfg = self.marker_config
            marker_fp = self.marker_fingerprint

        if marker_cfg is None:
            return []

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

        if self.auto_threshold:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
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

        if marker_cfg and len(marker_cfg) > 1 and len(detected_points) >= len(marker_cfg):
            # If we were tracking, prioritize points near last known location
            if self.confidence > 0.5 and self.last_tracked_points:
                center = np.mean(self.last_tracked_points, axis=0)
                detected_points.sort(key=lambda p: np.linalg.norm(np.array(p) - center))

            # Limit points to check to avoid O(N^K) explosion (N=12, K=4 -> 495 combos)
            limit = 12 if len(marker_cfg) >= 4 else 15
            points_to_check = detected_points[:limit]

            num_markers = len(marker_cfg)

            # Optimization: Pre-calculate pairwise distances between detected points
            # to avoid redundant norm calculations in the combinations loop.
            dist_matrix = {}
            for i, j in combinations(range(len(points_to_check)), 2):
                d = np.linalg.norm(np.array(points_to_check[i]) - np.array(points_to_check[j]))
                dist_matrix[(i, j)] = d

            for indices in combinations(range(len(points_to_check)), num_markers):
                # Quick bounding box check
                pts_arr = np.array([points_to_check[i] for i in indices])
                if self.confidence > 0.5 and self.last_tracked_points is not None:
                    # If tracking, current combo shouldn't be too far from last known size
                    prev_pts = np.array(self.last_tracked_points)
                    prev_size = np.max(prev_pts, axis=0) - np.min(prev_pts, axis=0)
                    curr_size = np.max(pts_arr, axis=0) - np.min(pts_arr, axis=0)
                    if any(curr_size > prev_size * 2.0) or any(curr_size < prev_size * 0.5):
                        continue

                current_distances = []
                for i, j in combinations(range(num_markers), 2):
                    # Map local combo indices back to dist_matrix keys
                    idx1, idx2 = sorted((indices[i], indices[j]))
                    current_distances.append(dist_matrix[(idx1, idx2)])

                current_fingerprint = sorted(current_distances)

                if len(current_fingerprint) == len(marker_fp):
                    point_combo = [points_to_check[i] for i in indices]
                    is_match = True
                    for i in range(len(current_fingerprint)):
                        if not np.isclose(current_fingerprint[i], marker_fp[i], rtol=0.15):
                            is_match = False
                            break
                    if is_match:
                        src_pts = np.float32(marker_cfg).reshape(-1, 1, 2)
                        dst_pts = np.float32(point_combo).reshape(-1, 1, 2)
                        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if matrix is not None:
                            self.last_homography = matrix
                            transformed_src = cv2.perspectiveTransform(src_pts, matrix)
                            ordered_points = []
                            remaining_dst = list(point_combo)
                            for i in range(num_markers):
                                pred = transformed_src[i][0]
                                closest = min(remaining_dst, key=lambda p: np.linalg.norm(np.array(p) - pred))
                                ordered_points.append(closest)
                                remaining_dst.remove(closest)

                            # Kalman Correction
                            kalman_pts = []
                            for i, pt in enumerate(ordered_points):
                                self.kalman_filters[i].correct(np.array([[np.float32(pt[0])], [np.float32(pt[1])]]))
                                pred = self.kalman_filters[i].predict()
                                kalman_pts.append((pred[0, 0], pred[1, 0]))

                            if self.smoothed_points is None:
                                self.smoothed_points = np.array(kalman_pts, dtype=np.float32)
                            else:
                                alpha = 1.0 - self.smoothing_factor
                                self.smoothed_points = self.smoothed_points * (1.0 - alpha) + np.array(kalman_pts, dtype=np.float32) * alpha

                            self.confidence = min(1.0, self.confidence + self.confidence_gain)
                            self.last_tracked_points = [tuple(p.astype(int)) for p in self.smoothed_points]

                            # Predictive ROI Scaling based on velocity
                            max_v = 0
                            for kf in self.kalman_filters:
                                v = np.linalg.norm(kf.statePost[2:])
                                max_v = max(max_v, v)
                            self.roi_padding = int(50 + max_v * 5)

                            return self.last_tracked_points

        if self.last_tracked_points is not None:
            if roi_x != 0 or roi_y != 0 or roi_w != w or roi_h != h:
                self.last_tracked_points = None
                return self.get_tracked_points(frame)

        self.confidence = max(0.0, self.confidence - self.confidence_decay)
        if self.confidence > 0.1 and self.last_tracked_points:
            return self.last_tracked_points

        return detected_points

    def _update_warp_maps(self, w, h):
        if w <= 0 or h <= 0: return
        self.map_x = np.zeros((h, w), dtype=np.float32)
        self.map_y = np.zeros((h, w), dtype=np.float32)

        wp = np.array(self.warp_points)
        wp[:, 0] *= w
        wp[:, 1] *= h

        res = self.warp_grid_res

        for r in range(res - 1):
            for c in range(res - 1):
                # Quad corners in the point list
                p1 = r * res + c
                p2 = r * res + (c + 1)
                p3 = (r + 1) * res + (c + 1)
                p4 = (r + 1) * res + c

                dst_q = np.float32([wp[p1], wp[p2], wp[p3], wp[p4]])

                # Source quad in linear grid
                src_x_start = c * (w // (res - 1))
                src_y_start = r * (h // (res - 1))
                src_x_end = (c + 1) * (w // (res - 1))
                src_y_end = (r + 1) * (h // (res - 1))

                if c == res - 2: src_x_end = w
                if r == res - 2: src_y_end = h

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

                # Clip to image boundaries
                min_x, max_x = max(0, min_x), min(w, max_x)
                min_y, max_y = max(0, min_y), min(h, max_y)

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
            # Use fixed 720p internal for performance, then upscale
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
                self.static_warp_cache.clear()

            if self._run_boundary_detection_flag:
                # Ensure resolution is stable before capturing
                if main_frame.shape[1] != self.requested_camera_res[0] or main_frame.shape[0] != self.requested_camera_res[1]:
                    # Still waiting for camera thread to catch up
                    self._sls_curr_wait = 0
                    continue

                if self._boundary_step == 0:
                    # Capture Black Frame
                    self.projector_buffer.fill(0)
                    if self._sls_curr_wait >= self._sls_wait_frames:
                        if curr_frame_id > self._last_captured_frame_id:
                            self._boundary_captures.append(self.boost_contrast(main_frame))
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
                            self._boundary_captures.append(self.boost_contrast(main_frame))
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
                    diff = cv2.absdiff(white, black)
                    # Use Otsu's thresholding for IR cameras
                    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # Fallback if Otsu results in near-zero coverage
                    if np.mean(thresh) < 1.0:
                         _, thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)

                    # Clean up
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    thresh = cv2.dilate(thresh, kernel, iterations=2)

                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Find largest contour
                        main_contour = max(contours, key=cv2.contourArea)
                        area_ratio = cv2.contourArea(main_contour) / (w_cam * h_cam)

                        if area_ratio < 0.05:
                            msg = f"Detected projector area is very small ({area_ratio:.1%}). Ensure the camera has a clear view of the projection."
                            self.system_warning.emit(msg)

                        # Simplify to a polygon (e.g. 8 points for some flexibility)
                        peri = cv2.arcLength(main_contour, True)
                        approx = cv2.approxPolyDP(main_contour, 0.01 * peri, True)

                        # Ensure we have at least 4 points
                        if len(approx) < 4:
                            rect = cv2.minAreaRect(main_contour)
                            approx = cv2.boxPoints(rect)

                        pts = [ (float(p[0][0]), float(p[0][1])) for p in approx ]
                        self.projector_boundary = pts
                        self.boundary_detected.emit(pts)
                    else:
                        print("Error: No projector light detected!")
                        self.boundary_detected.emit([])

                    self._run_boundary_detection_flag = False
                    self.requested_camera_res = (640, 480) # Revert to performance mode
                    self._boundary_step = 0
                    self._boundary_captures = []
            elif self._run_sls_flag:
                # Ensure resolution is stable before capturing
                if main_frame.shape[1] != self.requested_camera_res[0] or main_frame.shape[0] != self.requested_camera_res[1]:
                    # Still waiting for camera thread to catch up
                    self._sls_curr_wait = 0
                    continue

                # Structured Light Scanning takes priority
                total_x = len(self._sls_patterns_x)
                total_y = len(self._sls_patterns_y)

                if self._sls_step < total_x:
                    pattern = self._sls_patterns_x[self._sls_step]
                    self.projector_buffer = cv2.merge([pattern, pattern, pattern])
                    if self._sls_curr_wait >= self._sls_wait_frames:
                        if curr_frame_id > self._last_captured_frame_id:
                            gray = self.boost_contrast(main_frame)
                            self._sls_captures_x.append(gray)
                            self._sls_step += 1
                            self._sls_curr_wait = 0
                            self._last_captured_frame_id = curr_frame_id
                    else:
                        self._sls_curr_wait += 1
                elif self._sls_step < total_x + total_y:
                    idx = self._sls_step - total_x
                    pattern = self._sls_patterns_y[idx]
                    self.projector_buffer = cv2.merge([pattern, pattern, pattern])
                    if self._sls_curr_wait >= self._sls_wait_frames:
                        if curr_frame_id > self._last_captured_frame_id:
                            gray = self.boost_contrast(main_frame)
                            self._sls_captures_y.append(gray)
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

                    # Store dense LUT
                    self.sls_lut_x = proj_x.astype(np.float32)
                    self.sls_lut_y = proj_y.astype(np.float32)
                    self.sls_valid_mask = valid.astype(np.uint8)

                    # Collect mapping points for RBF (sub-sampled)
                    calib_data = []
                    step = 10 # Sample every 10 pixels
                    # Ensure indices are within bounds
                    h_idx, w_idx = valid.shape
                    for r in range(0, h_idx, step):
                        for c in range(0, w_idx, step):
                            if valid[r, c]:
                                calib_data.append([float(c), float(r),
                                                   float(proj_x[r, c]), float(proj_y[r, c])])

                    if len(calib_data) > 10:
                        self.init_rbf_from_points(calib_data)
                        # Estimate homography from sample points for fallback
                        pts = np.array(calib_data)
                        self.h_c2p, _ = cv2.findHomography(pts[:100, :2], pts[:100, 2:])
                        self.calibration_complete.emit(True)
                    else:
                        self.calibration_complete.emit(False)

                    self._run_sls_flag = False
                    self.requested_camera_res = (640, 480) # Revert to performance mode
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
                self.projector_buffer.fill(0)

            projector_output = self.projector_buffer

            # Handle Auto-Calibration logic (Multi-frame averaging)
            if self._run_calibration_flag:
                # Ensure resolution is stable before capturing
                if main_frame.shape[1] != self.requested_camera_res[0] or main_frame.shape[0] != self.requested_camera_res[1]:
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

                    proj_pts = []
                    for r in range(1, 7):
                        for c in range(1, 10):
                            proj_pts.append([start_x + c * sq_w, start_y + r * sq_h])

                    proj_pts = np.array(proj_pts, dtype=np.float32)
                    cam_pts = avg_corners.reshape(-1, 2)

                    # Store as point list for RBF
                    calib_data = []
                    for i in range(len(cam_pts)):
                        calib_data.append([float(cam_pts[i, 0]), float(cam_pts[i, 1]),
                                           float(proj_pts[i, 0]), float(proj_pts[i, 1])])

                    self.init_rbf_from_points(calib_data)
                    # Also keep homography as a robust fallback
                    self.h_c2p, _ = cv2.findHomography(cam_pts, proj_pts)

                    self._run_calibration_flag = False
                    self.requested_camera_res = (640, 480) # Revert to performance mode
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

                splash_frame = self.splash_player.get_frame()
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
                for point in ui_tracked_points:
                    cv2.circle(main_frame, point, 5, (0, 0, 255), -1)

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

            self.update_particles(tracked_points, h, w, w_cam, h_cam)
            self.draw_particles(projector_output)

            # Performer Occlusion Logic
            if self.occlusion_enabled:
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

            # Generator frame cache for this rendering cycle
            cycle_generator_cache = {}

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

                if mask.video_path == "generative":
                    if "generative" not in cycle_generator_cache:
                        cycle_generator_cache["generative"] = self.get_generative_frame(h, w)
                    frame_cue = cycle_generator_cache["generative"].copy()
                    frame_id = self.frame_count # Use rendering frame count as ID
                elif mask.video_path.startswith("generator:"):
                    pattern = mask.video_path.split(":")[-1]
                    if mask.video_path not in cycle_generator_cache:
                        cycle_generator_cache[mask.video_path] = self.get_vj_generator(pattern, h, w)
                    frame_cue = cycle_generator_cache[mask.video_path].copy()
                    frame_id = self.frame_count
                else:
                    with QMutexLocker(self.player_mutex):
                        if mask.video_path not in self.video_players:
                            player = VideoPlayer(mask.video_path)
                            player.start()
                            self.video_players[mask.video_path] = player

                        player = self.video_players[mask.video_path]
                        frame_cue, frame_id = player.get_frame()

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
                    # Performance Optimization: Downscale video frame if it's larger than the projector resolution
                    # This significantly speeds up all subsequent FX processing and warping.
                    fh, fw = frame_cue.shape[:2]
                    if fw > w or fh > h:
                        frame_cue = cv2.resize(frame_cue, (w, h), interpolation=cv2.INTER_LINEAR)

                    # Performance: Only apply FX if not fully throttled or essential
                    frame_cue = self.apply_fx(frame_cue, mask)

                    if mask.design_overlay != 'none':
                        design_m = self.get_design_mask(mask.design_overlay, frame_cue.shape[0], frame_cue.shape[1])
                        design_m_3ch = cv2.merge([design_m, design_m, design_m])
                        frame_cue = cv2.bitwise_and(frame_cue, design_m_3ch)

                    effective_frame_cue = frame_cue
                    if mask.type == 'dynamic':
                        if curr_confidence < 1.0:
                            effective_frame_cue = cv2.convertScaleAbs(frame_cue, alpha=curr_confidence)

                    effective_opacity = mask.opacity * curr_fade
                    if mask.fx_params.get('lfo_enabled') and mask.fx_params.get('lfo_target') == 'opacity':
                        effective_opacity *= lfo_val

                    if effective_opacity < 1.0:
                        effective_frame_cue = cv2.convertScaleAbs(effective_frame_cue, alpha=effective_opacity)

                    # Safety Fallback
                    is_safe = True
                    if mask.type == 'dynamic' and curr_confidence < 0.2 and self.safety_mode_enabled:
                        is_safe = False
                        # Override frame with fallback generator
                        frame_cue = self.get_vj_generator(self.fallback_generator, h, w)

                    # Static Caching Check
                    cache_key = (mask_id, tuple(map(tuple, mask.source_points)))
                    warped_cue = None
                    mask_img = None

                    if mask.type == 'static' and cache_key in self.static_warp_cache:
                        cached_fid, cached_warped, cached_mask_img = self.static_warp_cache[cache_key]
                        if cached_fid == frame_id:
                            warped_cue = cached_warped
                            mask_img = cached_mask_img

                    if warped_cue is None:
                        if mask.type == 'dynamic' and ((mask.is_linked and curr_homography is not None) or not is_safe):
                            src_pts = np.float32(mask.source_points)

                            if is_safe:
                                # 3D PnP Perspective Logic
                                if self.pnp_enabled and self.marker_config and len(self.marker_config) == 4 and len(tracked_points) == 4:
                                    # Model points (markers in their reference plane, Z=0)
                                    model_pts = np.array([ [p[0], p[1], 0] for p in self.marker_config ], dtype=np.float32)
                                    image_pts = np.array(tracked_points, dtype=np.float32)

                                    success, rvec, tvec = cv2.solvePnP(model_pts, image_pts, self.camera_matrix, self.dist_coeffs)

                                    if success:
                                        # Project the mask points based on 3D pose
                                        # Mask points are relative to the reference markers
                                        mask_model_pts = np.array([ [p[0], p[1], 0] for p in mask.source_points ], dtype=np.float32)
                                        dst_pts_raw, _ = cv2.projectPoints(mask_model_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                                        dst_pts_raw = dst_pts_raw.reshape(-1, 2)
                                    else:
                                        dst_pts_raw = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), curr_homography).reshape(-1, 2)
                                else:
                                    # Fallback to standard Homography
                                    dst_pts_raw = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), curr_homography).reshape(-1, 2)

                                # Audio Scaling
                                if self.audio_reactive_target == 'scale':
                                    center = np.mean(dst_pts_raw, axis=0)
                                    scale_factor = 1.0 + self.audio_bands[0] * 0.5
                                    dst_pts_cam = (dst_pts_raw - center) * scale_factor + center
                                else:
                                    dst_pts_cam = dst_pts_raw
                            else:
                                # Tracking lost or not linked: stay at source points or fallback
                                dst_pts_cam = src_pts

                            # Transform camera coordinates to internal render coordinates
                            dst_pts = self.transform_to_projector(dst_pts_cam, w_cam, h_cam, target_w=w, target_h=h)

                            # Warp video to the dynamic polygon
                            video_corners = np.float32([[0, 0], [frame_cue.shape[1], 0], [frame_cue.shape[1], frame_cue.shape[0]], [0, frame_cue.shape[0]]])

                            if len(dst_pts) == 4:
                                dst_pts_warp = np.float32(dst_pts).reshape(-1, 2)
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
                                warped_cue = cv2.warpPerspective(effective_frame_cue, matrix, (w, h))

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
                                if mask.feather > 0:
                                    k = int(mask.feather) | 1
                                    cv2.boxFilter(curr_mask_img, -1, (k, k), dst=curr_mask_img)
                                mask_img = curr_mask_img

                        elif mask.type == 'static' or (mask.type == 'dynamic' and not mask.is_linked):
                            effective_frame_cue = frame_cue

                            effective_opacity = mask.opacity
                        if mask.fx_params.get('lfo_enabled') and mask.fx_params.get('lfo_target') == 'opacity':
                            lfo_val = self.get_lfo_value(mask)
                            effective_opacity *= lfo_val

                        if effective_opacity < 1.0:
                            effective_frame_cue = cv2.convertScaleAbs(frame_cue, alpha=effective_opacity)

                            if not mask.source_points:
                                warped_cue = cv2.resize(effective_frame_cue, (w, h))
                                mask_img = np.full((h, w, 3), 255, dtype=np.uint8)
                            else:
                                src_pts = np.float32([[0, 0], [frame_cue.shape[1], 0], [frame_cue.shape[1], frame_cue.shape[0]], [0, frame_cue.shape[0]]])
                                dst_pts = self.transform_to_projector(mask.source_points, w_cam, h_cam, target_w=w, target_h=h)

                                if len(dst_pts) == 4:
                                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                                else:
                                    min_x, min_y = np.min(dst_pts, axis=0)
                                    max_x, max_y = np.max(dst_pts, axis=0)
                                    bbox_pts = np.float32([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
                                    matrix = cv2.getPerspectiveTransform(src_pts, bbox_pts)

                                warped_cue = cv2.warpPerspective(effective_frame_cue, matrix, (w, h))
                                curr_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
                                cv2.fillPoly(curr_mask_img, [np.int32(dst_pts)], (255, 255, 255))
                                if mask.feather > 0:
                                    k = int(mask.feather) | 1
                                    cv2.boxFilter(curr_mask_img, -1, (k, k), dst=curr_mask_img)
                                mask_img = curr_mask_img

                        # Store in cache if static
                        if mask.type == 'static' and warped_cue is not None:
                            self.static_warp_cache[cache_key] = (frame_id, warped_cue, mask_img)

                    if warped_cue is not None and mask_img is not None:
                        projector_output = self.blend_frames(projector_output, warped_cue, mask_img, mask.blend_mode)

                # Draw outlines on projector during calibration/alignment
                if self.show_camera_on_projector:
                    # Determine current points for outline
                    if mask.type == 'dynamic' and mask.is_linked and curr_homography is not None:
                        src_pts = np.float32(mask.source_points).reshape(-1, 1, 2)
                        draw_pts_cam = cv2.perspectiveTransform(src_pts, curr_homography).reshape(-1, 2)
                    else:
                        draw_pts_cam = np.array(mask.source_points)

                    draw_pts = self.transform_to_projector(draw_pts_cam, w_cam, h_cam, target_w=w, target_h=h).astype(np.int32)

                    if len(draw_pts) >= 3:
                        cv2.polylines(projector_output, [draw_pts], True, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(projector_output, f"{mask.tag or mask.name}", (draw_pts[0][0], draw_pts[0][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if self._capture_still_frame_flag:
                rgb = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                self.still_frame_ready.emit(QImage(rgb.data, w_cam, h_cam, w_cam * 3, QImage.Format_RGB888).copy(), tracked_points)
                self._capture_still_frame_flag = False

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
                                draw_pts = cv2.perspectiveTransform(src_pts, curr_homography).reshape(-1, 2).astype(np.int32)
                            except:
                                draw_pts = np.int32(mask.source_points)
                        else:
                            draw_pts = np.int32(mask.source_points)

                        if len(draw_pts) >= 2:
                            color = (0, 255, 0) if mask.is_linked else (0, 255, 255)
                            cv2.polylines(main_frame, [draw_pts], True, color, 2, cv2.LINE_AA)
                            if mask.is_linked:
                                cv2.putText(main_frame, f"LINKED: {mask.name}", tuple(draw_pts[0]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            rgb_main = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(QImage(rgb_main.data, w_cam, h_cam, w_cam * 3, QImage.Format_RGB888).copy())

            # Apply Master FX to the composition before warping
            if self.master_active_fx or self.master_brightness != 0 or self.master_contrast != 0 or self.master_saturation != 100:
                master_proxy = Mask("Master", [], None)
                master_proxy.active_fx = self.master_active_fx
                master_proxy.tint_color = self.master_tint_color
                projector_output = self.apply_fx(projector_output, master_proxy)

                # Apply Brightness/Contrast
                if self.master_brightness != 0 or self.master_contrast != 0:
                    alpha = (self.master_contrast + 100.0) / 100.0
                    beta = self.master_brightness
                    projector_output = cv2.convertScaleAbs(projector_output, alpha=alpha, beta=beta)

                # Apply Saturation
                if self.master_saturation != 100:
                    hsv = cv2.cvtColor(projector_output, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:,:,1] *= (self.master_saturation / 100.0)
                    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
                    projector_output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # Apply Grain
                if self.master_grain > 0:
                    noise = np.random.randint(0, self.master_grain, (h, w, 3), dtype=np.uint8)
                    projector_output = cv2.add(projector_output, noise)

                # Apply Bloom (Simple)
                if self.master_bloom > 0:
                    # Extract highlights
                    gray = cv2.cvtColor(projector_output, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                    highlights = cv2.bitwise_and(projector_output, cv2.merge([mask, mask, mask]))
                    # Blur highlights
                    k = int(self.master_bloom / 2) | 1
                    bloom = cv2.GaussianBlur(highlights, (k, k), 0)
                    projector_output = cv2.addWeighted(projector_output, 1.0, bloom, 0.5, 0)

            # Apply Master Fader
            if self.master_fader < 1.0:
                projector_output = cv2.convertScaleAbs(projector_output, alpha=self.master_fader)

            # Apply Projector Boundary Global Clip (in internal resolution)
            if self.projector_boundary:
                clip_mask = np.zeros((h, w), dtype=np.uint8)
                proj_pts = self.transform_to_projector(self.projector_boundary, w_cam, h_cam, target_w=w, target_h=h)
                cv2.fillPoly(clip_mask, [np.int32(proj_pts)], 255)
                projector_output = cv2.bitwise_and(projector_output, cv2.merge([clip_mask, clip_mask, clip_mask]))

            # Apply Performer Occlusion
            if self.occlusion_enabled and self.occlusion_mask is not None:
                # Black out the performer
                inv_mask = cv2.bitwise_not(self.occlusion_mask)
                if len(inv_mask.shape) == 2:
                    inv_mask = cv2.merge([inv_mask, inv_mask, inv_mask])
                projector_output = cv2.bitwise_and(projector_output, inv_mask)

            # 9-point grid warping (piecewise perspective optimization)
            if self._warp_is_identity:
                warped_output = projector_output
            else:
                if self.map_x is None or self.map_x.shape[:2] != (h, w) or self._warp_map_dirty:
                    self._update_warp_maps(w, h)

                warped_output = cv2.remap(projector_output, self.map_x, self.map_y, cv2.INTER_LINEAR)

            # Final Upscale to Projector Resolution
            if (w, h) != (w_proj, h_proj):
                warped_output = cv2.resize(warped_output, (w_proj, h_proj), interpolation=cv2.INTER_LINEAR)

            rgb_proj = cv2.cvtColor(warped_output, cv2.COLOR_BGR2RGB)
            self.projector_frame_ready.emit(QImage(rgb_proj.data, w_proj, h_proj, w_proj * 3, QImage.Format_RGB888).copy())

            elapsed = time.time() - start_time

            # Auto-Throttle Logic
            if elapsed > frame_time:
                self.throttle_level = min(1.0, self.throttle_level + 0.05)
            else:
                self.throttle_level = max(0.0, self.throttle_level - 0.01)

            sleep_time = max(1, int((frame_time - elapsed) * 1000))
            QThread.msleep(sleep_time)

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
            proj_pts = self.transform_to_projector(tracked_points, w_cam, h_cam, target_w=w, target_h=h)
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
        for p in self.particles:
            color = (0, 255, 0) if self.particle_preset == 'rain' else (255, 255, 200)
            alpha = int(p['life'] * 255)
            # Use anti-aliasing for smoother particles
            cv2.circle(frame, (int(p['x']), int(p['y'])), 2, color, -1, cv2.LINE_AA)

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
            spacing = 40
            offset = int((t * 20) % spacing)
            for x in range(offset, w, spacing):
                cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 2)
            for y in range(offset, h, spacing):
                cv2.line(frame, (0, y), (w, y), (0, 255, 0), 2)
        elif pattern == 'scan':
            y = int((t * 100) % h)
            cv2.line(frame, (0, y), (w, y), (255, 255, 255), 5)
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
            for i in range(0, 360, 20):
                angle = np.radians(i + t * 100)
                end_x = int(center[0] + max(w, h) * np.cos(angle))
                end_y = int(center[1] + max(w, h) * np.sin(angle))
                cv2.line(frame, center, (end_x, end_y), (255, 255, 0), 2)
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
        # Ensure mask_img has the same number of channels as base
        if len(mask_img.shape) == 2:
            mask_img = cv2.merge([mask_img, mask_img, mask_img])

        if self.blend_temp1 is None or self.blend_temp1.shape != base.shape:
            self.blend_temp1 = np.zeros_like(base)
            self.blend_temp2 = np.zeros_like(base)

        # Optimized integer-based blending to avoid slow float conversions
        if mode == 'normal':
            # Full alpha blending: res = overlay * (mask/255) + base * (1 - mask/255)
            cv2.multiply(overlay, mask_img, dst=self.blend_temp1, scale=1.0/255.0)
            cv2.bitwise_not(mask_img, dst=self.blend_temp2)
            cv2.multiply(base, self.blend_temp2, dst=self.blend_temp2, scale=1.0/255.0)
            return cv2.add(self.blend_temp1, self.blend_temp2, dst=base)
        elif mode == 'add':
            cv2.multiply(overlay, mask_img, dst=self.blend_temp1, scale=1.0/255.0)
            return cv2.add(base, self.blend_temp1, dst=base)
        elif mode == 'screen':
            # res = 1 - (1 - base) * (1 - overlay*mask)
            cv2.multiply(overlay, mask_img, dst=self.blend_temp1, scale=1.0/255.0)
            cv2.bitwise_not(base, dst=self.blend_temp2)
            cv2.bitwise_not(self.blend_temp1, dst=self.blend_temp1)
            cv2.multiply(self.blend_temp2, self.blend_temp1, dst=self.blend_temp1, scale=1.0/255.0)
            return cv2.bitwise_not(self.blend_temp1, dst=base)
        elif mode == 'multiply':
            # res = base * (overlay*mask + (255-mask)) / 255
            cv2.multiply(overlay, mask_img, dst=self.blend_temp1, scale=1.0/255.0)
            cv2.bitwise_not(mask_img, dst=self.blend_temp2)
            cv2.add(self.blend_temp1, self.blend_temp2, dst=self.blend_temp1)
            return cv2.multiply(base, self.blend_temp1, dst=base, scale=1.0/255.0)

        return base

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
        self.requested_camera_res = (9999, 9999) # Request max FOV
        self._boundary_step = 0
        self._boundary_captures = []
        self._sls_curr_wait = 0
        self._run_boundary_detection_flag = True

    def stop_calibration(self):
        self._run_calibration_flag = False
        self._run_sls_flag = False
        self._run_boundary_detection_flag = False
        self.requested_camera_res = (640, 480)
        self.show_calibration_pattern = False
        self.show_calibration_verify = False

    def run_auto_calibration(self):
        self.requested_camera_res = (9999, 9999) # Request max FOV
        self._run_calibration_flag = True

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

    def transform_to_projector(self, pts, w_cam=640, h_cam=480, target_w=None, target_h=None):
        if target_w is None: target_w = self.projector_width
        if target_h is None: target_h = self.projector_height

        scale_w = target_w / self.projector_width
        scale_h = target_h / self.projector_height

        # Prefer Dense LUT if available
        if self.sls_lut_x is not None:
            try:
                pts_arr = np.array(pts, dtype=np.float32)
                res = []
                for p in pts_arr.reshape(-1, 2):
                    ix, iy = int(round(p[0])), int(round(p[1]))
                    if 0 <= ix < w_cam and 0 <= iy < h_cam and self.sls_valid_mask[iy, ix]:
                        res.append([self.sls_lut_x[iy, ix] * scale_w, self.sls_lut_y[iy, ix] * scale_h])
                    else:
                        # Fallback to RBF for out-of-bounds or invalid pixels
                        if self.rbf_x:
                            rx = self.rbf_x(p.reshape(1,2))[0] * scale_w
                            ry = self.rbf_y(p.reshape(1,2))[0] * scale_h
                            res.append([rx, ry])
                        else:
                            # Proportional fallback
                            res.append([p[0] * (target_w / w_cam), p[1] * (target_h / h_cam)])
                return np.array(res).reshape(-1, 2)
            except:
                pass

        if self.rbf_x is not None:
            try:
                pts_arr = np.array(pts, dtype=np.float32)
                if pts_arr.ndim == 1:
                    pts_arr = pts_arr.reshape(1, 2)

                tx = self.rbf_x(pts_arr) * scale_w
                ty = self.rbf_y(pts_arr) * scale_h
                return np.stack([tx, ty], axis=1).reshape(-1, 2)
            except:
                pass

        if self.h_c2p is not None:
            try:
                pts_reshaped = np.float32(pts).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts_reshaped, self.h_c2p)
                res = transformed.reshape(-1, 2)
                res[:, 0] *= scale_w
                res[:, 1] *= scale_h
                return res
            except:
                pass

        # Fallback: Simple proportional scaling if not calibrated
        # This prevents the "top-left corner" issue when calibration is missing.
        scale_x = target_w / max(1, w_cam)
        scale_y = target_h / max(1, h_cam)

        pts_arr = np.array(pts, dtype=np.float32)
        if pts_arr.ndim == 1:
            return np.array([pts_arr[0] * scale_x, pts_arr[1] * scale_y])

        pts_arr_c = pts_arr.copy()
        pts_arr_c[:, 0] *= scale_x
        pts_arr_c[:, 1] *= scale_y
        return pts_arr_c

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
