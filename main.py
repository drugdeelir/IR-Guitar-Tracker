import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
# Note: DO NOT disable MSMF — it's the only backend that works for camera 0 on this machine

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QByteArray, QEventLoop, QPoint, QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QShortcut,
    QSlider,
    QSplitter,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mask import Mask
from splash import SplashScreen
from widgets import MarkerSelectionDialog, PolygonMaskDialog, ProjectorWindow, VideoDisplay
from worker import Worker

try:
    import mido
except ImportError:
    mido = None

SETTINGS_PATH = Path("settings.json")

__version__ = "1.0.1"
_SETTINGS_VERSION = 4


def configure_opencv_logging():
    from logging.handlers import RotatingFileHandler
    _fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                             datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    root.addHandler(_ch)

    # Rotating file handler — survives show days without filling disk
    _fh = RotatingFileHandler("ir_tracker.log", maxBytes=2_000_000, backupCount=3,
                               encoding="utf-8")
    _fh.setFormatter(_fmt)
    root.addHandler(_fh)

    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except AttributeError:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass  # OpenCV build does not expose logging API; silently skip


def _migrate_settings(d: dict) -> dict:
    """Migrate settings dict to current version."""
    version = d.get("version", 1)
    if version < 2:
        # v1→v2: masks moved from flat list to structured dicts
        d["version"] = 2
    if version < 3:
        # v2→v3: added render_order and locked fields to masks
        for mask_dict in d.get("masks", []):
            mask_dict.setdefault("render_order", 0)
            mask_dict.setdefault("locked", False)
        d["version"] = 3
    if version < 4:
        # v3→v4: added opacity, blend_mode, loop_mode, fade_in, fade_out, enabled, label_color
        for mask_dict in d.get("masks", []):
            mask_dict.setdefault("enabled", True)
            mask_dict.setdefault("opacity", 1.0)
            mask_dict.setdefault("blend_mode", "normal")
            mask_dict.setdefault("loop_mode", "loop")
            mask_dict.setdefault("label_color", None)
            mask_dict.setdefault("fade_in", 0.0)
            mask_dict.setdefault("fade_out", 0.0)
        d["version"] = 4
    return d


# Alias so CameraScanThread can import without circular issues
_scan_cameras = None  # will be set after get_available_cameras is defined


class StartupWizardDialog(QDialog):
    def __init__(self, cameras, screens, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Startup Wizard")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Let's quickly configure your tracking session."))

        form = QFormLayout()
        self.camera_combo = QComboBox()
        if cameras:
            self.camera_combo.addItems([f"Camera {i}" for i in cameras])
            camera_idx = settings.get("camera_index", 0)
            self.camera_combo.setCurrentIndex(min(camera_idx, len(cameras) - 1))
        else:
            self.camera_combo.addItem("No camera detected")
            self.camera_combo.setEnabled(False)
        form.addRow("Camera", self.camera_combo)

        self.projector_combo = QComboBox()
        self.projector_combo.addItems(
            [screen.name() or f"Screen {i + 1}" for i, screen in enumerate(screens)]
        )
        if screens:
            proj_idx = settings.get("projector_index", 0)
            self.projector_combo.setCurrentIndex(min(proj_idx, len(screens) - 1))
        form.addRow("Projector", self.projector_combo)

        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["Manual", "Auto (Otsu)"])
        self.threshold_combo.setCurrentIndex(settings.get("threshold_mode", 0))
        form.addRow("Threshold Mode", self.threshold_combo)

        self.auto_sync_checkbox = QCheckBox("Enable auto-sync of marker links")
        self.auto_sync_checkbox.setChecked(settings.get("auto_sync_enabled", True))
        form.addRow("", self.auto_sync_checkbox)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

def _get_camera_backends(include_fallback_any=False):
    """Return backends for one-shot camera index probing.
    Uses DSHOW on Windows to reduce noisy backend warnings during probing.
    Worker._camera_backends() handles sustained capture and prefers MSMF."""
    is_windows = platform.system().lower() == "windows"
    if not is_windows:
        return [cv2.CAP_ANY]

    preferred = ["CAP_DSHOW"]
    if include_fallback_any:
        preferred.append("CAP_ANY")

    backends = []
    for name in preferred:
        backend = getattr(cv2, name, None)
        if backend is not None and backend not in backends:
            backends.append(backend)
    return backends or [cv2.CAP_ANY]


def _open_capture(index, backend):
    try:
        return cv2.VideoCapture(index, backend)
    except TypeError:
        return cv2.VideoCapture(index)


def get_available_cameras(max_probe=8):
    arr = []
    backends = _get_camera_backends(include_fallback_any=False)
    misses_after_first = 0
    misses_before_first = 0

    for index in range(max_probe):
        opened = False
        for backend in backends:
            cap = _open_capture(index, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    continue
                opened = True
                cap.release()
                break
            cap.release()
        if opened:
            arr.append(index)
            misses_after_first = 0
        elif arr:
            misses_after_first += 1
            if misses_after_first >= 3:
                break
        else:
            misses_before_first += 1
            if misses_before_first >= 4 and index >= 3:
                break
    return arr


_scan_cameras = get_available_cameras


class CameraScanThread(QThread):
    cameras_found = pyqtSignal(list)

    def run(self):
        cameras = _scan_cameras()
        self.cameras_found.emit(cameras)


class ProjectionMappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("ProjectionMappingApp")
        self.logger.info(
            "IR Guitar Tracker v%s | OpenCV %s | Qt %s | NumPy %s | Python %s",
            __version__,
            cv2.__version__,
            QtCore.PYQT_VERSION_STR if hasattr(QtCore, 'PYQT_VERSION_STR') else '?',
            np.__version__,
            sys.version.split()[0],
        )
        self.setWindowTitle(f"IR Guitar Tracker v{__version__}")
        self.setGeometry(100, 100, 1200, 800)
        self.masks = []
        self.selected_markers = []
        self.reference_markers = []
        self.latest_camera_qimage = None
        self._is_closing = False
        self._camera_retry_count = 0
        self.settings = self.load_settings()

        self.setStatusBar(QStatusBar(self))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.video_display = VideoDisplay()
        self.projector_window = ProjectorWindow()

        self.worker = Worker()
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.create_control_panel()

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.control_scroll)
        self.main_splitter.addWidget(self.video_display)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 4)
        self.main_splitter.setSizes([620, 900])
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.layout.addWidget(self.main_splitter)
        self.video_display.mask_point_added.connect(self.add_mask_point_to_list)
        self.projector_window.show()

        self.worker.frame_ready.connect(self.video_display.set_image, Qt.QueuedConnection)
        self.worker.frame_ready.connect(self.cache_latest_frame, Qt.QueuedConnection)
        self.worker.projector_frame_ready.connect(self.projector_window.set_image, Qt.QueuedConnection)
        self.worker.projector_frame_ready.connect(self.update_projector_preview, Qt.QueuedConnection)
        self.projector_window.warp_points_changed.connect(self.worker.set_warp_points)
        self.worker.trackers_detected.connect(self.update_tracker_label, Qt.QueuedConnection)
        self.worker.camera_error.connect(self.show_camera_error, Qt.QueuedConnection)
        self.worker.performance_updated.connect(self.update_performance_label, Qt.QueuedConnection)
        self.worker.camera_info_updated.connect(self.update_camera_info, Qt.QueuedConnection)
        self.worker.markers_calibrated.connect(self._on_markers_calibrated, Qt.QueuedConnection)
        self.worker.worker_stopped.connect(self._on_worker_stopped, Qt.QueuedConnection)
        self.worker.calibration_progress.connect(self._on_calibration_progress, Qt.QueuedConnection)
        self.worker.performance_degraded.connect(self._on_performance_degraded, Qt.QueuedConnection)
        # Improvement 49: connect new worker signals
        self.worker.tracking_state_changed.connect(self._on_tracking_state_changed, Qt.QueuedConnection)
        self.worker.calibration_restored.connect(self._on_calibration_restored, Qt.QueuedConnection)
        self.worker.diagnostic_info.connect(self._on_diagnostic_info, Qt.QueuedConnection)
        # Improvement 10: connect guitar multi-candidate signal
        self.worker.guitar_candidates_ready.connect(self._on_guitar_candidates_ready, Qt.QueuedConnection)

        # Screen-change recovery: repopulate projector combo when displays change
        _app = QApplication.instance()
        _app.screenAdded.connect(self._on_screens_changed)
        _app.screenRemoved.connect(self._on_screens_changed)

        self.marker_selection_dialog = MarkerSelectionDialog(self)
        self.marker_selection_dialog.take_picture_button.clicked.connect(
            self.start_marker_capture_countdown
        )
        self.worker.still_frame_ready.connect(self.set_marker_selection_image)

        self._save_settings_timer = QTimer(self)
        self._save_settings_timer.setSingleShot(True)
        self._save_settings_timer.setInterval(2000)
        self._save_settings_timer.timeout.connect(self._do_save_settings)

        self.apply_loaded_settings()
        self.refresh_mask_views()
        self.change_projector(self.projector_combo.currentIndex())
        self.maybe_show_startup_wizard()
        self._auto_create_test_masks()

        self._midi_port_names = []
        self._midi_hotplug_timer = QTimer(self)
        self._midi_hotplug_timer.setInterval(2000)
        self._midi_hotplug_timer.timeout.connect(self._check_midi_hotplug)
        self._midi_hotplug_timer.start()

        # Improvement 50: auto-save every 5 minutes so show config is preserved
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setInterval(300_000)  # 5 minutes
        self._auto_save_timer.timeout.connect(self._auto_save)
        self._auto_save_timer.start()

        # Improvement 51: track calibration time for "last calibrated" display
        self._calibration_timestamp = None
        self._calibration_age_timer = QTimer(self)
        self._calibration_age_timer.setInterval(60_000)  # update every minute
        self._calibration_age_timer.timeout.connect(self._update_calibration_age_label)
        self._calibration_age_timer.start()

        # Improvement 52: lock mode — prevents accidental edits during show
        self._lock_mode = False

        import os as _os
        if _os.environ.get('IRTK_PROFILE_MEMORY'):
            import tracemalloc
            tracemalloc.start()
            self._memory_check_timer = QTimer(self)
            self._memory_check_timer.setInterval(300_000)  # every 5 minutes
            self._memory_check_timer.timeout.connect(self._check_memory)
            self._memory_check_timer.start()

        self.thread.started.connect(self.worker.process_video)
        self.thread.start()

        self._setup_shortcuts()

    def load_settings(self):
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
                data = _migrate_settings(data)
                return self._validate_settings(data)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.logger.warning("Could not load settings: %s", exc)
                # Try backup file
                backup = SETTINGS_PATH.with_name('settings_backup.json')
                if backup.exists():
                    try:
                        data = json.loads(backup.read_text(encoding='utf-8'))
                        data = _migrate_settings(data)
                        self.logger.info("Restored settings from backup")
                        return self._validate_settings(data)
                    except Exception:
                        pass
                return {"_corrupted": True}
        return {}

    def _validate_settings(self, d: dict) -> dict:
        """Coerce and clamp all settings fields to valid types and ranges."""
        def _int(key, default, lo, hi):
            try:
                return max(lo, min(hi, int(d.get(key, default))))
            except (TypeError, ValueError):
                return default

        def _float(key, default, lo, hi):
            try:
                return max(lo, min(hi, float(d.get(key, default))))
            except (TypeError, ValueError):
                return default

        def _bool(key, default):
            v = d.get(key, default)
            if isinstance(v, bool):
                return v
            if isinstance(v, int):
                return bool(v)
            return default

        def _str(key, default, allowed=None):
            v = str(d.get(key, default))
            return v if (allowed is None or v in allowed) else default

        result = dict(d)  # preserve unknown keys
        result.update({
            "ir_threshold": _int("ir_threshold", 200, 0, 255),
            "depth_sensitivity": _int("depth_sensitivity", 100, 1, 500),
            "camera_index": _int("camera_index", 0, 0, 99),
            "projector_index": _int("projector_index", 0, 0, 20),
            "threshold_mode": _int("threshold_mode", 0, 0, 1),
            "auto_sync_enabled": _bool("auto_sync_enabled", True),
            "show_preview_enabled": _bool("show_preview_enabled", True),
            "show_mask_overlays": _bool("show_mask_overlays", True),
            "camera_mode": _str("camera_mode", "native", {"native", "performance", "hd"}),
            "expected_marker_count": _int("expected_marker_count", 4, 1, 8),
        })
        return result

    def save_settings(self):
        settings = {
            # Improvement 46: save correct settings version (was incorrectly saving version 2)
            "version": _SETTINGS_VERSION,
            "ir_threshold": self.ir_threshold_slider.value(),
            "threshold_mode": self.threshold_mode_combo.currentIndex(),
            "depth_sensitivity": self.depth_sensitivity_slider.value(),
            "camera_index": self.camera_combo.currentIndex(),
            "projector_index": self.projector_combo.currentIndex(),
            "warp_points": self.projector_window.get_warp_points_normalized(),
            "auto_sync_enabled": self.auto_sync_checkbox.isChecked(),
            "show_preview_enabled": self.preview_checkbox.isChecked(),
            "camera_mode": self.camera_mode_combo.currentData(),
            "show_mask_overlays": self.show_mask_overlays_checkbox.isChecked(),
            "wizard_completed": True,
            "masks": [m.to_dict() for m in self.masks],
            # Improvement 47: save operator notes
            "operator_notes": getattr(self, '_operator_notes', ''),
            # Improvement 48: save expected marker count
            "expected_marker_count": getattr(self, 'marker_count_spin', None) and
                self.marker_count_spin.value() or 4,
            "window_geometry": self.saveGeometry().toBase64().data().decode(),
            "splitter_sizes": self.main_splitter.sizes(),
        }
        try:
            # Rotate backup files
            if SETTINGS_PATH.exists():
                _backup = SETTINGS_PATH.with_name('settings_backup.json')
                try:
                    import shutil
                    shutil.copy2(str(SETTINGS_PATH), str(_backup))
                except Exception:
                    pass
            _tmp = SETTINGS_PATH.with_suffix('.tmp')
            _tmp.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding='utf-8')
            _tmp.replace(SETTINGS_PATH)
        except OSError as exc:
            self.logger.error("Could not save settings: %s", exc)

    def _do_save_settings(self):
        """Debounced settings save — called 2s after last change."""
        self.save_settings()

    def request_save_settings(self):
        """Trigger a debounced settings save."""
        self._save_settings_timer.start()

    def apply_loaded_settings(self):
        assert not self.thread.isRunning(), "apply_loaded_settings must be called before thread start"
        if self.settings.get("_corrupted"):
            QMessageBox.warning(
                self, "Settings Corrupted",
                "Your settings file was corrupted and could not be loaded.\n"
                "Default settings have been applied. Your show configuration may need to be reconfigured."
            )
        ir_threshold = self.settings.get("ir_threshold", 200)
        self.ir_threshold_slider.setValue(ir_threshold)
        threshold_mode = self.settings.get("threshold_mode", 0)
        self.threshold_mode_combo.setCurrentIndex(threshold_mode)
        self.depth_sensitivity_slider.setValue(self.settings.get("depth_sensitivity", 100))
        self.auto_sync_checkbox.setChecked(self.settings.get("auto_sync_enabled", True))
        self.preview_checkbox.setChecked(self.settings.get("show_preview_enabled", True))
        self.show_mask_overlays_checkbox.setChecked(self.settings.get("show_mask_overlays", True))
        self.projector_preview_label.setVisible(self.preview_checkbox.isChecked())

        camera_mode = self.settings.get("camera_mode", "native")
        mode_index = self.camera_mode_combo.findData(camera_mode)
        if mode_index >= 0:
            self.camera_mode_combo.setCurrentIndex(mode_index)

        self.worker.set_show_mask_overlays(self.show_mask_overlays_checkbox.isChecked())

        cam_idx = self.settings.get("camera_index", 0)
        if self.camera_combo.count() and self.camera_combo.isEnabled():
            self.camera_combo.setCurrentIndex(min(cam_idx, self.camera_combo.count() - 1))

        proj_idx = self.settings.get("projector_index", 0)
        if self.projector_combo.count():
            self.projector_combo.setCurrentIndex(min(proj_idx, self.projector_combo.count() - 1))

        warp_points = self.settings.get("warp_points")
        if isinstance(warp_points, list) and len(warp_points) == 4:
            self.projector_window.warp_points = self.projector_window.deserialize_warp_points(warp_points)
            self.worker.set_warp_points(self.projector_window.get_warp_points_normalized())

        # Restore saved masks (version 2+)
        saved_masks = self.settings.get("masks", [])
        if saved_masks:
            try:
                from mask import Mask as _Mask
                for d in saved_masks:
                    m = _Mask.from_dict(d)
                    self.masks.append(m)
                self.worker.set_masks(self.masks)
                self.refresh_mask_views()
                self.logger.info("Restored %d mask(s) from settings", len(saved_masks))
            except Exception as exc:
                self.logger.warning("Could not restore saved masks: %s", exc)

        # Restore window geometry and splitter
        geom = self.settings.get("window_geometry")
        if geom:
            try:
                self.restoreGeometry(QByteArray.fromBase64(geom.encode()))
            except Exception:
                pass
        splitter_sizes = self.settings.get("splitter_sizes")
        if isinstance(splitter_sizes, list) and len(splitter_sizes) == 2:
            self.main_splitter.setSizes(splitter_sizes)

        # Improvement: restore operator notes
        notes = self.settings.get("operator_notes", "")
        if notes and hasattr(self, 'operator_notes_edit'):
            self.operator_notes_edit.setPlainText(notes)
        self._operator_notes = notes

        # Improvement: restore expected marker count
        marker_count = self.settings.get("expected_marker_count", 4)
        if hasattr(self, 'marker_count_spin'):
            self.marker_count_spin.setValue(int(marker_count))
            self.worker.set_expected_marker_count(int(marker_count))

    def _run_marker_selection_dialog(self, *, use_live_capture=True, reference_pixmap=None, title="Select IR Markers", ir_assist=True):
        self.marker_selection_dialog.setWindowTitle(title)
        self.marker_selection_dialog.clear_selection()
        self.marker_selection_dialog.set_ir_assist_enabled(ir_assist)
        self.marker_selection_dialog.take_picture_button.setVisible(use_live_capture)
        self.marker_selection_dialog.take_picture_button.setEnabled(use_live_capture)
        self.marker_selection_dialog.take_picture_button.setText("Take Picture")

        if reference_pixmap is not None:
            self.marker_selection_dialog.set_pixmap(reference_pixmap)

        # Auto-close the dialog after 3 minutes to prevent it hanging open during a show
        _timeout_ms = 180_000  # 3 minutes
        _timeout_timer = QTimer(self.marker_selection_dialog)
        _timeout_timer.setSingleShot(True)
        _timeout_timer.timeout.connect(self.marker_selection_dialog.reject)
        _timeout_timer.start(_timeout_ms)

        accepted = self.marker_selection_dialog.exec_()
        _timeout_timer.stop()

        if accepted:
            markers = self.marker_selection_dialog.get_selected_points()
            if len(markers) != 4:
                QMessageBox.warning(self, "Marker Selection", "Please select exactly 4 guitar markers.")
                return []
            return markers[:4]
        return []

    def select_reference_guitar_markers(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Guitar Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not image_path:
            return []

        reference_pixmap = QPixmap(image_path)
        if reference_pixmap.isNull():
            QMessageBox.warning(self, "Reference Image", "Could not load the selected image.")
            return []

        return self._run_marker_selection_dialog(
            use_live_capture=False,
            reference_pixmap=reference_pixmap,
            title="Select 4 Reference Markers (uploaded image)",
            ir_assist=False,
        )

    def open_marker_selection_dialog(self):
        markers = self._run_marker_selection_dialog(
            use_live_capture=True,
            reference_pixmap=None,
            title="Select Live IR Markers",
            ir_assist=True,
        )
        if not markers:
            return

        self.selected_markers = markers
        self.worker.set_marker_points(self.selected_markers)
        self.statusBar().showMessage(
            f"Selected {len(self.selected_markers)} markers.", 3000
        )

    def start_marker_capture_countdown(self):
        self.marker_selection_dialog.take_picture_button.setEnabled(False)
        self.countdown_timer = QTimer(self)
        self.countdown_seconds = 7
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

    def update_countdown(self):
        if self.countdown_seconds > 0:
            self.marker_selection_dialog.take_picture_button.setText(
                f"{self.countdown_seconds}..."
            )
            self.countdown_seconds -= 1
        else:
            self.countdown_timer.stop()
            self.marker_selection_dialog.take_picture_button.setText("Take Picture")
            self.marker_selection_dialog.take_picture_button.setEnabled(True)
            self.worker.capture_still_frame()

    def set_marker_selection_image(self, image):
        self.marker_selection_dialog.set_pixmap(QPixmap.fromImage(image))

    def clear_marker_selection(self):
        self.selected_markers = []
        self.worker.clear_marker_config()
        self.statusBar().showMessage("Marker selection cleared.", 3000)

    def maybe_show_startup_wizard(self):
        if self.settings.get("wizard_completed"):
            return

        dialog = StartupWizardDialog(self.available_cameras, self.screens, self.settings, self)
        if dialog.exec_():
            if dialog.camera_combo.isEnabled():
                cam_idx = dialog.camera_combo.currentIndex()
                # Validate selected camera can actually deliver a frame (#40)
                if self.available_cameras:
                    real_idx = self.available_cameras[min(cam_idx, len(self.available_cameras) - 1)]
                    test_cap = cv2.VideoCapture(real_idx)
                    ok, _ = test_cap.read() if test_cap.isOpened() else (False, None)
                    test_cap.release()
                    if not ok:
                        QMessageBox.warning(
                            self, "Camera Not Ready",
                            f"Camera {real_idx} did not return a frame.\n"
                            "Check that it is plugged in and not in use by another application."
                        )
                self.camera_combo.setCurrentIndex(cam_idx)
                self.change_camera(cam_idx)
            if self.projector_combo.count() > 0:
                self.projector_combo.setCurrentIndex(dialog.projector_combo.currentIndex())
                self.change_projector(dialog.projector_combo.currentIndex())
            self.threshold_mode_combo.setCurrentIndex(dialog.threshold_combo.currentIndex())
            self.auto_sync_checkbox.setChecked(dialog.auto_sync_checkbox.isChecked())
            self.settings["wizard_completed"] = True
            self.save_settings()

    def create_control_panel(self):
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)

        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setWidget(self.control_panel)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_scroll.setMinimumWidth(560)
        self.control_scroll.setMaximumWidth(560)

        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip("Select the IR camera to use for tracking.")
        self.camera_mode_combo = QComboBox()
        self.camera_mode_combo.addItem("Native / Driver Default", "native")
        self.camera_mode_combo.addItem("Performance (960x540 @ 30)", "performance")
        self.camera_mode_combo.addItem("High Detail (1280x720 @ 30)", "hd")
        self.camera_mode_combo.setToolTip(
            "Native: camera's own default.\n"
            "Performance: lower resolution for speed.\n"
            "High Detail: 720p for more precise detection."
        )
        self.camera_mode_combo.currentIndexChanged.connect(self.update_camera_mode)
        self.refresh_camera_button = QPushButton("Refresh Cameras")
        self.refresh_camera_button.setToolTip("Re-scan for connected cameras.")
        self.refresh_camera_button.clicked.connect(self.refresh_cameras)
        self.retry_camera_button = QPushButton("Retry Camera")
        self.retry_camera_button.setToolTip("Attempt to reconnect to the currently selected camera.")
        self.retry_camera_button.clicked.connect(self.retry_camera)

        self.available_cameras = []
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        self.refresh_cameras(initial=True)

        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.camera_mode_combo)
        camera_layout.addWidget(self.refresh_camera_button)
        camera_layout.addWidget(self.retry_camera_button)
        camera_group.setLayout(camera_layout)
        self.control_layout.addWidget(camera_group)

        projector_group = QGroupBox("Projector Display")
        projector_layout = QVBoxLayout()
        self.projector_combo = QComboBox()
        self.projector_combo.setToolTip("Select which display is the projector output.")
        self.screens = QApplication.screens()
        self.projector_combo.addItems(
            [screen.name() or f"Screen {i + 1}" for i, screen in enumerate(self.screens)]
        )
        self.projector_combo.currentIndexChanged.connect(self.change_projector)
        self.logger.info("Detected displays: %s", [f"{i}:{s.name()} {s.geometry()}" for i, s in enumerate(self.screens)])
        projector_layout.addWidget(self.projector_combo)

        # Improvement 74: reconnect projector button in case display goes missing
        proj_btn_row = QHBoxLayout()
        self.reconnect_projector_button = QPushButton("Reconnect Projector")
        self.reconnect_projector_button.setToolTip(
            "Re-detect displays and move projector window to the selected screen."
        )
        self.reconnect_projector_button.clicked.connect(self._reconnect_projector)
        # Improvement 75: show/hide projector window button
        self.toggle_projector_button = QPushButton("Hide Projector")
        self.toggle_projector_button.setCheckable(True)
        self.toggle_projector_button.setToolTip("Show or hide the projector output window.")
        self.toggle_projector_button.toggled.connect(self._toggle_projector_visibility)
        proj_btn_row.addWidget(self.reconnect_projector_button)
        proj_btn_row.addWidget(self.toggle_projector_button)
        projector_layout.addLayout(proj_btn_row)
        projector_group.setLayout(projector_layout)
        self.control_layout.addWidget(projector_group)

        self.setup_wizard_button = QPushButton("Run Full Calibration Wizard")
        self.setup_wizard_button.clicked.connect(self.run_full_calibration_wizard_safe)
        self.setup_wizard_button.setToolTip("Step-by-step wizard: detect projector bounds, set background/guitar/amp masks, and calibrate depth.")
        self.control_layout.addWidget(self.setup_wizard_button)

        self.calibrate_button = QPushButton("Calibrate Guitar + Homography")
        self.calibrate_button.setStyleSheet("QPushButton { background-color: #2a5; color: white; font-weight: bold; padding: 8px; }")
        self.calibrate_button.clicked.connect(self._start_calibration)
        self.calibrate_button.setToolTip("Run 3-phase calibration: dark → illuminate → detect markers (Ctrl+C)")
        self.control_layout.addWidget(self.calibrate_button)

        # Improvement 58: Stage-control row — Blackout + Lock Mode
        stage_row = QHBoxLayout()
        self.blackout_button = QPushButton("BLACKOUT")
        self.blackout_button.setCheckable(True)
        self.blackout_button.setStyleSheet(
            "QPushButton { background-color: #8b0000; color: white; font-weight: bold; padding: 8px; border-radius: 5px; }"
            "QPushButton:checked { background-color: #cc0000; color: white; }"
        )
        self.blackout_button.setToolTip("Emergency: send solid black to projector (Esc). Tracking continues.")
        self.blackout_button.toggled.connect(self._toggle_blackout)
        stage_row.addWidget(self.blackout_button)

        self.lock_mode_button = QPushButton("Lock UI")
        self.lock_mode_button.setCheckable(True)
        self.lock_mode_button.setToolTip("Lock all editing controls to prevent accidental changes during a show.")
        self.lock_mode_button.toggled.connect(self._toggle_lock_mode)
        stage_row.addWidget(self.lock_mode_button)
        self.control_layout.addLayout(stage_row)

        # Improvement 59: Screenshot button
        self.screenshot_button = QPushButton("Save Screenshot")
        self.screenshot_button.setToolTip("Save the current projector output frame as a PNG file.")
        self.screenshot_button.clicked.connect(self._save_screenshot)
        self.control_layout.addWidget(self.screenshot_button)

        # Improvement 60: pre-show checklist button
        self.checklist_button = QPushButton("Pre-Show Checklist")
        self.checklist_button.setToolTip("Run a quick check of camera, calibration, MIDI, and cues before going live.")
        self.checklist_button.clicked.connect(self._run_preshow_checklist)
        self.control_layout.addWidget(self.checklist_button)

        self.mapping_tabs = QTabWidget()

        # Page 1: Masks
        masks_page = QWidget()
        masks_layout = QVBoxLayout(masks_page)

        mask_list_group = QGroupBox("Masks")
        mask_list_layout = QVBoxLayout()
        self.mask_list_widget = QListWidget()
        self.mask_list_widget.currentRowChanged.connect(self.on_mask_selection_changed)
        # Improvement 61: double-click to rename mask
        self.mask_list_widget.itemDoubleClicked.connect(self._rename_mask_inline)
        mask_list_layout.addWidget(self.mask_list_widget)

        mask_action_row = QHBoxLayout()
        self.remove_mask_button = QPushButton("Remove")
        self.remove_mask_button.setToolTip("Remove the selected mask and all its cues.")
        self.remove_mask_button.clicked.connect(self.remove_mask)
        # Improvement 62: enable/disable mask toggle
        self.toggle_mask_button = QPushButton("Enable/Disable")
        self.toggle_mask_button.setToolTip("Toggle the selected mask on or off without deleting it.")
        self.toggle_mask_button.clicked.connect(self._toggle_mask_enabled)
        # Improvement 63: rename mask button
        self.rename_mask_button = QPushButton("Rename")
        self.rename_mask_button.setToolTip("Rename the selected mask (F2).")
        self.rename_mask_button.clicked.connect(self._rename_mask_inline)
        mask_action_row.addWidget(self.remove_mask_button)
        mask_action_row.addWidget(self.toggle_mask_button)
        mask_action_row.addWidget(self.rename_mask_button)
        mask_list_layout.addLayout(mask_action_row)

        # Improvement 64: move mask up/down for render order
        mask_order_row = QHBoxLayout()
        self.move_mask_up_button = QPushButton("▲ Up")
        self.move_mask_up_button.setToolTip("Move selected mask up in render order.")
        self.move_mask_up_button.clicked.connect(self._move_mask_up)
        self.move_mask_down_button = QPushButton("▼ Down")
        self.move_mask_down_button.setToolTip("Move selected mask down in render order.")
        self.move_mask_down_button.clicked.connect(self._move_mask_down)
        mask_order_row.addWidget(self.move_mask_up_button)
        mask_order_row.addWidget(self.move_mask_down_button)
        mask_list_layout.addLayout(mask_order_row)

        # Improvement 65: export / import mask configs
        mask_io_row = QHBoxLayout()
        self.export_masks_button = QPushButton("Export Masks…")
        self.export_masks_button.setToolTip("Save all masks and cue paths to a JSON file.")
        self.export_masks_button.clicked.connect(self._export_masks)
        self.import_masks_button = QPushButton("Import Masks…")
        self.import_masks_button.setToolTip("Load masks from a previously exported JSON file.")
        self.import_masks_button.clicked.connect(self._import_masks)
        mask_io_row.addWidget(self.export_masks_button)
        mask_io_row.addWidget(self.import_masks_button)
        mask_list_layout.addLayout(mask_io_row)

        mask_list_group.setLayout(mask_list_layout)
        masks_layout.addWidget(mask_list_group)

        mask_group = QGroupBox("Mask Creation")
        mask_layout = QVBoxLayout()
        self.create_mask_button = QPushButton("Create Mask")
        self.create_mask_button.clicked.connect(self.enter_mask_creation_mode)
        self.finish_mask_button = QPushButton("Finish Mask")
        self.finish_mask_button.clicked.connect(self.finish_mask_creation)
        self.finish_mask_button.setEnabled(False)
        self.cancel_mask_button = QPushButton("Cancel")
        self.cancel_mask_button.clicked.connect(self.cancel_mask_creation)
        self.cancel_mask_button.setEnabled(False)
        self.mask_points_list = QListWidget()

        mask_layout.addWidget(self.create_mask_button)
        mask_layout.addWidget(self.finish_mask_button)
        mask_layout.addWidget(self.cancel_mask_button)
        mask_layout.addWidget(self.mask_points_list)

        self.link_mask_button = QPushButton("Link Mask to Markers")
        self.link_mask_button.clicked.connect(self.link_mask_to_markers)
        self.auto_sync_checkbox = QCheckBox("Auto-sync marker links")
        self.auto_sync_checkbox.setChecked(True)
        mask_layout.addWidget(self.link_mask_button)
        mask_layout.addWidget(self.auto_sync_checkbox)

        mask_group.setLayout(mask_layout)
        masks_layout.addWidget(mask_group)

        # Page 2: Cues + MIDI
        cues_page = QWidget()
        cues_layout = QVBoxLayout(cues_page)

        cue_group = QGroupBox("Cue Queues per Mask")
        cue_layout = QVBoxLayout()
        self.cue_mask_combo = QComboBox()
        self.cue_mask_combo.currentIndexChanged.connect(self.refresh_cues_for_selected_mask)
        self.mask_cue_list_widget = QListWidget()
        self.mask_cue_list_widget.currentRowChanged.connect(self.on_mask_cue_selected)

        self.add_cue_button = QPushButton("Add Video Cue to Mask")
        self.add_cue_button.clicked.connect(self.add_cue)
        self.remove_cue_button = QPushButton("Remove Selected Cue")
        self.remove_cue_button.clicked.connect(self.remove_cue)

        self.cc_spin = QSpinBox()
        self.cc_spin.setRange(0, 127)
        self.cc_spin.setPrefix("CC ")
        self.map_cc_button = QPushButton("Map CC to Selected Cue")
        self.map_cc_button.clicked.connect(self.map_cc_to_selected_cue)

        self.render_all_cues_button = QPushButton("Render All Masks")
        self.render_all_cues_button.clicked.connect(lambda: self.worker.set_active_cue_index(-1))

        # Cue preview row (#35)
        preview_cue_layout = QHBoxLayout()
        self.preview_cue_button = QPushButton("Preview Selected Cue")
        self.preview_cue_button.setToolTip(
            "Play selected cue in the preview panel only — projector output not affected."
        )
        self.preview_cue_button.clicked.connect(self._preview_selected_cue)
        self.stop_preview_cue_button = QPushButton("Stop Preview")
        self.stop_preview_cue_button.clicked.connect(self._stop_cue_preview)
        preview_cue_layout.addWidget(self.preview_cue_button)
        preview_cue_layout.addWidget(self.stop_preview_cue_button)

        cue_layout.addWidget(QLabel("Mask"))
        cue_layout.addWidget(self.cue_mask_combo)
        cue_layout.addWidget(self.mask_cue_list_widget)
        cue_layout.addWidget(self.add_cue_button)
        cue_layout.addWidget(self.remove_cue_button)
        cue_layout.addLayout(preview_cue_layout)

        # Improvement 66: cue advance button (manual next cue)
        cue_advance_row = QHBoxLayout()
        self.advance_cue_button = QPushButton("▶ Next Cue")
        self.advance_cue_button.setToolTip("Manually advance to the next cue on the selected mask.")
        self.advance_cue_button.clicked.connect(self._advance_cue)
        cue_advance_row.addWidget(self.advance_cue_button)

        # Improvement 67: cue loop mode toggle
        self.cue_loop_combo = QComboBox()
        self.cue_loop_combo.addItem("Loop", "loop")
        self.cue_loop_combo.addItem("One-shot", "oneshot")
        self.cue_loop_combo.setToolTip("Loop: video restarts. One-shot: holds on last frame.")
        self.cue_loop_combo.currentIndexChanged.connect(self._update_cue_loop_mode)
        cue_advance_row.addWidget(self.cue_loop_combo)
        cue_layout.addLayout(cue_advance_row)

        cue_layout.addWidget(self.cc_spin)
        cue_layout.addWidget(self.map_cc_button)
        cue_layout.addWidget(self.render_all_cues_button)
        cue_group.setLayout(cue_layout)
        cues_layout.addWidget(cue_group)

        midi_group = QGroupBox("MIDI CC Input (Network MIDI capable)")
        midi_layout = QVBoxLayout()
        self.midi_input_combo = QComboBox()
        self.refresh_midi_button = QPushButton("Refresh MIDI Inputs")
        self.refresh_midi_button.clicked.connect(self.refresh_midi_inputs)
        self.connect_midi_button = QPushButton("Connect MIDI")
        self.connect_midi_button.clicked.connect(self.connect_midi_input)

        # Improvement 68: MIDI activity indicator
        midi_status_row = QHBoxLayout()
        self.midi_status_label = QLabel("MIDI: Disconnected")
        self.midi_status_label.setProperty("class", "status-idle")
        self.midi_activity_label = QLabel("●")
        self.midi_activity_label.setFixedWidth(16)
        self.midi_activity_label.setStyleSheet("color: #444; font-size: 16px;")
        midi_status_row.addWidget(self.midi_activity_label)
        midi_status_row.addWidget(self.midi_status_label)
        midi_status_row.addStretch()
        midi_layout.addLayout(midi_status_row)

        midi_layout.addWidget(self.midi_input_combo)
        midi_layout.addWidget(self.refresh_midi_button)
        midi_layout.addWidget(self.connect_midi_button)
        midi_group.setLayout(midi_layout)
        cues_layout.addWidget(midi_group)

        # Improvement 69: Operator Notes tab for show-day notes
        notes_page = QWidget()
        notes_layout = QVBoxLayout(notes_page)
        notes_layout.addWidget(QLabel("Show notes (saved with settings):"))
        from PyQt5.QtWidgets import QTextEdit
        self.operator_notes_edit = QTextEdit()
        self.operator_notes_edit.setPlaceholderText(
            "Enter pre-show notes, cue call times, tech notes…"
        )
        self.operator_notes_edit.setToolTip("Notes are saved automatically with settings.")
        self.operator_notes_edit.textChanged.connect(self._notes_changed)
        notes_layout.addWidget(self.operator_notes_edit)

        self.mapping_tabs.addTab(masks_page, "Masks")
        self.mapping_tabs.addTab(cues_page, "Cues")
        self.mapping_tabs.addTab(notes_page, "Notes")
        self.control_layout.addWidget(self.mapping_tabs)

        self.midi_inport = None
        self.midi_poll_timer = QTimer(self)
        self.midi_poll_timer.timeout.connect(self.poll_midi_messages)
        self.midi_poll_timer.start(20)
        self.refresh_midi_inputs()

        warping_group = QGroupBox("Projector Warping")
        warping_layout = QVBoxLayout()
        self.enable_warping_button = QPushButton("Enable Warping")
        self.enable_warping_button.setCheckable(True)
        self.enable_warping_button.toggled.connect(self.toggle_warping)
        self.reset_warping_button = QPushButton("Reset Warping")
        self.reset_warping_button.clicked.connect(self.projector_window.reset_warp_points)
        warping_layout.addWidget(self.enable_warping_button)
        warping_layout.addWidget(self.reset_warping_button)
        warping_group.setLayout(warping_layout)
        self.control_layout.addWidget(warping_group)

        ir_group = QGroupBox("IR Tracking")
        ir_layout = QVBoxLayout()
        self.ir_threshold_slider = QSlider(Qt.Horizontal)
        self.ir_threshold_slider.setRange(0, 255)
        self.ir_threshold_slider.setValue(200)
        self.ir_threshold_slider.valueChanged.connect(self.update_ir_threshold)
        self.ir_trackers_label = QLabel("Trackers detected: 0")

        self.threshold_mode_combo = QComboBox()
        self.threshold_mode_combo.addItems(["Manual", "Auto (Otsu)"])
        self.threshold_mode_combo.currentIndexChanged.connect(self.update_threshold_mode)

        # Improvement 53: slider label row (label + value readout)
        ir_thresh_row = QHBoxLayout()
        ir_thresh_row.addWidget(QLabel("IR Threshold:"))
        self.ir_threshold_value_label = QLabel("200")
        self.ir_threshold_value_label.setMinimumWidth(32)
        ir_thresh_row.addWidget(self.ir_threshold_value_label)
        ir_layout.addLayout(ir_thresh_row)
        self.ir_threshold_slider.valueChanged.connect(
            lambda v: self.ir_threshold_value_label.setText(str(v))
        )
        ir_layout.addWidget(self.ir_threshold_slider)
        ir_layout.addWidget(QLabel("Threshold Mode:"))
        ir_layout.addWidget(self.threshold_mode_combo)
        ir_layout.addWidget(self.ir_trackers_label)

        # Improvement 54: configurable expected marker count
        marker_count_row = QHBoxLayout()
        marker_count_row.addWidget(QLabel("Expected Markers:"))
        self.marker_count_spin = QSpinBox()
        self.marker_count_spin.setRange(1, 8)
        self.marker_count_spin.setValue(4)
        self.marker_count_spin.setToolTip(
            "Number of IR markers on the guitar. Change only if using more/fewer markers."
        )
        self.marker_count_spin.valueChanged.connect(self.worker.set_expected_marker_count)
        marker_count_row.addWidget(self.marker_count_spin)
        ir_layout.addLayout(marker_count_row)

        # Improvement 55: calibration state indicator label
        self.calib_state_label = QLabel("State: Not calibrated")
        self.calib_state_label.setProperty("class", "status-error")
        ir_layout.addWidget(self.calib_state_label)

        self.select_markers_button = QPushButton("Select Guitar Markers")
        self.select_markers_button.clicked.connect(self.open_marker_selection_dialog)
        self.select_markers_button.setToolTip("Open camera view to manually identify and select IR marker positions.")
        self.clear_markers_button = QPushButton("Clear Marker Selection")
        self.clear_markers_button.clicked.connect(self.clear_marker_selection)
        self.clear_markers_button.setToolTip("Clear all marker point selections.")

        # Improvement 56: reset calibration without full re-run
        self.reset_calibration_button = QPushButton("Reset Calibration")
        self.reset_calibration_button.setToolTip(
            "Clear the calibration cache and return to uncalibrated mode. "
            "Tracking continues using global blob search until you recalibrate."
        )
        self.reset_calibration_button.clicked.connect(self._reset_calibration)
        ir_layout.addWidget(self.select_markers_button)
        ir_layout.addWidget(self.clear_markers_button)
        ir_layout.addWidget(self.reset_calibration_button)
        ir_group.setLayout(ir_layout)
        self.control_layout.addWidget(ir_group)

        depth_group = QGroupBox("Depth Estimation")
        depth_layout = QVBoxLayout()
        self.calibrate_depth_button = QPushButton("Calibrate Depth")
        self.calibrate_depth_button.clicked.connect(self.calibrate_depth)
        self.depth_sensitivity_slider = QSlider(Qt.Horizontal)
        self.depth_sensitivity_slider.setRange(0, 200)
        self.depth_sensitivity_slider.setValue(100)
        self.depth_sensitivity_slider.valueChanged.connect(self.update_depth_sensitivity)
        self.depth_calibration_label = QLabel("Not calibrated")
        depth_layout.addWidget(self.calibrate_depth_button)
        # Improvement 57: depth slider label with value readout
        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Sensitivity:"))
        self.depth_sensitivity_value_label = QLabel("100")
        self.depth_sensitivity_value_label.setMinimumWidth(32)
        depth_row.addWidget(self.depth_sensitivity_value_label)
        depth_layout.addLayout(depth_row)
        self.depth_sensitivity_slider.valueChanged.connect(
            lambda v: self.depth_sensitivity_value_label.setText(str(v))
        )
        depth_layout.addWidget(self.depth_sensitivity_slider)
        depth_layout.addWidget(self.depth_calibration_label)
        depth_group.setLayout(depth_layout)
        self.control_layout.addWidget(depth_group)

        diagnostics_group = QGroupBox("Diagnostics")
        diagnostics_layout = QVBoxLayout()
        self.performance_label = QLabel("FPS: -- | Frame: -- | D: -- M: -- W: -- R: --")
        self.camera_info_label = QLabel("Camera: waiting for stream...")
        # Improvement 70: FPS history - show min/avg over last 60 readings
        self.fps_history_label = QLabel("FPS avg/min: --/-- (last 60s)")
        self._fps_history = []
        # Improvement 71: tracking state display in diagnostics
        self.tracking_state_diag_label = QLabel("Tracking: idle")
        self.tracking_state_diag_label.setProperty("class", "status-idle")
        # Improvement 72: blob count diagnostic
        self.blob_count_label = QLabel("Blobs: --  Tracked: --")
        diagnostics_layout.addWidget(self.performance_label)
        diagnostics_layout.addWidget(self.fps_history_label)
        diagnostics_layout.addWidget(self.camera_info_label)
        diagnostics_layout.addWidget(self.tracking_state_diag_label)
        diagnostics_layout.addWidget(self.blob_count_label)
        # Improvement 73: About button in diagnostics for easy access
        self.about_button = QPushButton("About IR Guitar Tracker")
        self.about_button.clicked.connect(self._show_about_dialog)
        diagnostics_layout.addWidget(self.about_button)
        diagnostics_group.setLayout(diagnostics_layout)
        self.control_layout.addWidget(diagnostics_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_checkbox = QCheckBox("Show projector preview")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.toggled.connect(self.toggle_preview)
        self.projector_preview_label = QLabel("Waiting for projector frames...")
        self.projector_preview_label.setAlignment(Qt.AlignCenter)
        self.projector_preview_label.setMinimumHeight(220)
        preview_layout.addWidget(self.preview_checkbox)
        preview_layout.addWidget(self.projector_preview_label)
        preview_group.setLayout(preview_layout)
        self.control_layout.addWidget(preview_group)

        overlays_group = QGroupBox("Mask Overlays")
        overlays_layout = QVBoxLayout()
        self.show_mask_overlays_checkbox = QCheckBox("Show calibration/static masks")
        self.show_mask_overlays_checkbox.setChecked(True)
        self.show_mask_overlays_checkbox.toggled.connect(self.worker.set_show_mask_overlays)
        overlays_layout.addWidget(self.show_mask_overlays_checkbox)
        overlays_group.setLayout(overlays_layout)
        self.control_layout.addWidget(overlays_group)

        self.apply_preview_minimum_sizes()
        self.apply_control_sizing()
        self.control_layout.addStretch()

    def apply_preview_minimum_sizes(self):
        primary = QApplication.primaryScreen()
        if primary is None:
            return

        screen_geo = primary.availableGeometry()
        min_w = max(320, screen_geo.width() // 4)
        min_h = max(180, screen_geo.height() // 4)
        self.video_display.setMinimumSize(min_w, min_h)
        self.projector_preview_label.setMinimumSize(min_w, min_h)

    def toggle_preview(self, checked):
        self.projector_preview_label.setVisible(checked)
        self.worker.set_preview_enabled(checked)

    def update_projector_preview(self, image):
        if not self.preview_checkbox.isChecked() or not self.projector_preview_label.isVisible():
            return

        target_size = self.projector_preview_label.size()
        safe_w = max(1, target_size.width())
        safe_h = max(1, target_size.height())

        pixmap = QPixmap.fromImage(image)
        self.projector_preview_label.setPixmap(
            pixmap.scaled(
                safe_w,
                safe_h,
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def apply_control_sizing(self):
        for button in self.control_panel.findChildren(QPushButton):
            button.setMinimumHeight(42)
            button.setMinimumWidth(260)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for combo in self.control_panel.findChildren(QComboBox):
            combo.setMinimumHeight(34)
            combo.setMinimumWidth(260)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    @pyqtSlot(QImage)
    def cache_latest_frame(self, image):
        self.latest_camera_qimage = image.copy()
        self._camera_retry_count = 0  # reset auto-reconnect counter on successful frame

    def capture_still_frame_sync(self, timeout_ms=5000, label="capture"):
        loop = QEventLoop(self)
        result = {"image": None}

        def on_frame(image):
            result["image"] = image.copy()
            loop.quit()

        self.logger.debug("Requesting still frame: %s", label)
        # Disconnect any stale connection from a previous call before re-connecting.
        try:
            self.worker.still_frame_ready.disconnect(on_frame)
        except RuntimeError:
            pass
        self.worker.still_frame_ready.connect(on_frame)
        self.worker.capture_still_frame()
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec_()
        try:
            self.worker.still_frame_ready.disconnect(on_frame)
        except RuntimeError:
            pass

        if result["image"] is None:
            self.logger.warning("Still capture timed out for %s", label)
        else:
            self.logger.info("Still capture succeeded for %s (%dx%d)", label, result["image"].width(), result["image"].height())
        return result["image"]

    def capture_still_frame_warmed(self, label, warmup_ms=250, samples=2):
        QApplication.processEvents()
        time.sleep(max(0.0, warmup_ms / 1000.0))
        still = None
        for i in range(max(1, samples)):
            still = self.capture_still_frame_sync(label=f"{label}#{i+1}")
            QApplication.processEvents()
            time.sleep(0.08)
        return still

    def _qimage_to_bgr(self, image):
        if image is None:
            return None
        qimg = image.convertToFormat(QImage.Format_RGB32)
        w = qimg.width()
        h = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    @staticmethod
    def _order_quad_points(points):
        pts = np.array(points, dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(d)]
        ordered[3] = pts[np.argmax(d)]
        return ordered

    def detect_projector_bounds(self, off_image, on_image):
        """Detect projector bounds from on/off image pair.

        Returns (normalized_points, auto_accepted, confidence_msg) where:
          - normalized_points: list of 4 [x,y] floats in [0,1] range, or None on failure
          - auto_accepted: True if confidence thresholds were met (Improvement 1)
          - confidence_msg: human-readable confidence description (empty string if None)
        """
        off_frame = self._qimage_to_bgr(off_image)
        on_frame = self._qimage_to_bgr(on_image)
        if off_frame is None or on_frame is None:
            return None, False, ""

        h, w = on_frame.shape[:2]
        diff = cv2.absdiff(on_frame, off_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)

        adaptive_threshold = int(np.percentile(diff_gray, 95))
        _, otsu = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, pct = cv2.threshold(diff_gray, adaptive_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_or(otsu, pct)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.logger.warning("Projector bounds detection: no contours from off/on diff; trying bright-frame fallback")
            _, thresh = cv2.threshold(cv2.cvtColor(on_frame, cv2.COLOR_BGR2GRAY), 170, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self.logger.error("Projector bounds detection failed after fallback")
                return None, False, ""

        image_area = float(w * h)

        # Collect all valid candidates with scores (Improvement 2: multi-projector scoring)
        candidates = []  # list of (score, quad, area_ratio, rect_score_val, contour_area)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.05:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4:
                quad = approx.reshape(-1, 2)
            else:
                rect = cv2.minAreaRect(contour)
                quad = cv2.boxPoints(rect)

            quad_area = abs(cv2.contourArea(np.array(quad, dtype=np.float32)))
            x, y, bw, bh = cv2.boundingRect(np.array(quad, dtype=np.int32))
            rect_area = max(float(bw * bh), 1.0)
            rect_score_val = quad_area / rect_area  # 1.0 = perfect rectangle
            area_ratio = quad_area / image_area

            # Improvement 2: score = rect_score * sqrt(area), favouring large rectangular regions
            multi_score = rect_score_val * (quad_area ** 0.5)
            candidates.append((multi_score, quad, area_ratio, rect_score_val, quad_area))

        if not candidates:
            return None, False, ""

        # Improvement 2: sort candidates by score descending; pick top one
        candidates.sort(key=lambda x: -x[0])
        best = candidates[0]
        best_score, best_quad, best_area_ratio, best_rect_score, best_area = best

        # Improvement 2: warn if runner-up is close to winner
        if len(candidates) >= 2:
            runner_up_score = candidates[1][0]
            if runner_up_score >= best_score * 0.80:
                warn_msg = "Multiple projector regions detected — auto-selected best candidate. Please verify."
                self.logger.warning(warn_msg)
                if hasattr(self, "worker") and hasattr(self.worker, "calibration_warning"):
                    self.worker.calibration_warning.emit(warn_msg)

        ordered = self._order_quad_points(best_quad)
        normalized = []
        for x, y in ordered:
            normalized.append([
                float(max(0.0, min(1.0, x / max(w, 1)))),
                float(max(0.0, min(1.0, y / max(h, 1)))),
            ])

        # Improvement 1: auto-accept if confidence thresholds are met
        # area_ratio > 0.05 AND rect_score > 0.75 AND only 1 candidate
        auto_accepted = (
            best_area_ratio > 0.05
            and best_rect_score > 0.75
            and len(candidates) == 1
        )
        # Combined confidence metric for display (weighted area_ratio + rect_score)
        combined_confidence = best_area_ratio * 0.5 + best_rect_score * 0.5
        confidence_msg = f"Projector bounds auto-detected (confidence: {combined_confidence:.0%})"
        if auto_accepted:
            self.logger.info("Projector bounds auto-accepted: area_ratio=%.2f rect_score=%.2f",
                             best_area_ratio, best_rect_score)
        return normalized, auto_accepted, confidence_msg


    def _draw_polygon_overlay(self, image, points, color=Qt.green):
        if image is None:
            return None
        overlay = QImage(image)
        if not points:
            return overlay
        painter = QPainter(overlay)
        painter.setPen(QPen(color, 4))
        for idx, point in enumerate(points):
            painter.drawPoint(point)
            painter.drawText(point.x() + 8, point.y() - 8, str(idx + 1))
            if idx > 0:
                painter.drawLine(points[idx - 1], point)
        if len(points) > 2:
            painter.drawLine(points[-1], points[0])
        painter.end()
        return overlay

    def ensure_mask(self, name, points, mask_type="static", linked_marker_count=0):
        for i, mask in enumerate(self.masks):
            if mask.name == name:
                mask.source_points = points
                mask.type = mask_type
                mask.linked_marker_count = linked_marker_count
                if linked_marker_count <= 0:
                    mask.marker_anchor_points = []
                self.refresh_mask_views(select_index=i)
                return mask
        mask = Mask(name, points, None)
        mask.type = mask_type
        mask.linked_marker_count = linked_marker_count
        mask.marker_anchor_points = []
        self.masks.append(mask)
        self.refresh_mask_views(select_index=len(self.masks) - 1)
        return mask

    def _auto_create_test_masks(self):
        """Auto-create guitar and background masks for testing if none exist."""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        guitar_video = os.path.join(base_dir, "test_guitar.mp4")
        bg_video = os.path.join(base_dir, "test_background.mp4")

        if not os.path.exists(guitar_video) or not os.path.exists(bg_video):
            return
        if self.masks:
            return  # don't overwrite existing masks

        self.logger.info("Auto-creating test masks (waiting for marker calibration)")

        # Use actual camera resolution for mask coordinates.
        # Camera 0 is 640x480 — masks must be in camera pixel space.
        cam_w, cam_h = 640, 480

        # Background mask: full camera frame (static, always projected)
        bg_points = [
            (0, 0),
            (cam_w, 0),
            (cam_w, cam_h),
            (0, cam_h),
        ]
        bg_mask = self.ensure_mask("Background", bg_points, mask_type="static")
        bg_mask.add_cue(bg_video)

        # Store camera dimensions for guitar mask computation
        self._cam_w = cam_w
        self._cam_h = cam_h

        # Guitar mask will be created dynamically after auto-calibration
        # detects the 4 marker positions. Store the video path for later.
        self._guitar_video_path = guitar_video

        self.worker.set_active_cue_index(-1)  # render ALL masks
        self.worker.set_masks(self.masks)
        self.refresh_mask_views(select_index=0)
        self.logger.info("Test masks created: Background (%dx%d), Guitar pending calibration", cam_w, cam_h)

    def _on_markers_calibrated(self, marker_positions):
        """Called when worker auto-detects stable marker positions.
        Uses the actual guitar polygon from silhouette detection for the mask."""
        self.logger.info("Markers calibrated at: %s", marker_positions)

        # Re-enable calibration button
        if hasattr(self, 'calibrate_button'):
            self.calibrate_button.setEnabled(True)
            self.calibrate_button.setText("Calibrate Guitar + Homography")

        # Improvement: update calibration state UI
        self._calibration_timestamp = time.time()
        if hasattr(self, 'calib_state_label'):
            self.calib_state_label.setText(f"State: Calibrated ({len(marker_positions)} markers)")
            self.calib_state_label.setProperty("class", "status-ok")
            self.calib_state_label.style().unpolish(self.calib_state_label)
            self.calib_state_label.style().polish(self.calib_state_label)
        self._status_message(
            f"Calibration complete — {len(marker_positions)} markers detected.", "success", 6000
        )

        if not hasattr(self, '_guitar_video_path') or not self._guitar_video_path:
            return

        # Use actual camera resolution from worker (set once frames start)
        cam_w = self.worker.frame_width or getattr(self, '_cam_w', 640)
        cam_h = self.worker.frame_height or getattr(self, '_cam_h', 480)
        self._cam_w = cam_w
        self._cam_h = cam_h

        # Use the actual guitar polygon from silhouette detection if available.
        # This is built in worker._detect_markers_from_diff from the actual
        # contour shape, giving a much more accurate mask than the old
        # 4-marker T-shape extrapolation.
        guitar_polygon = getattr(self.worker, '_guitar_polygon', None)
        if guitar_polygon and len(guitar_polygon) >= 3:
            guitar_points = [(int(x), int(y)) for x, y in guitar_polygon]
            self.logger.info(
                "Using actual guitar polygon from silhouette (%d points): %s",
                len(guitar_points), guitar_points
            )
        else:
            # Fallback: construct from marker extremes
            pts = [(int(x), int(y)) for x, y in marker_positions]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = 10
            guitar_points = [
                (min(xs) - pad, min(ys) - pad),
                (max(xs) + pad, min(ys) - pad),
                (max(xs) + pad, max(ys) + pad),
                (min(xs) - pad, max(ys) + pad),
            ]
            self.logger.info("Fallback guitar rect from markers: %s", guitar_points)

        guitar_mask = self.ensure_mask("Guitar", guitar_points, mask_type="static")
        if not guitar_mask.cues or not any(c for c in guitar_mask.cues):
            guitar_mask.add_cue(self._guitar_video_path)

        # Update the background mask to cover full camera frame
        self._update_background_exclude_guitar(guitar_points, cam_w, cam_h)

        self.worker.set_masks(self.masks)
        self.refresh_mask_views(select_index=0)
        # MUST be after refresh_mask_views, which triggers on_mask_selection_changed
        # that would set active_cue_index to the selected row (0=Background only).
        # Setting -1 here ensures ALL masks render (background + guitar).
        self.worker.set_active_cue_index(-1)

        # Save verification image: camera frame with mask polygon overlaid
        self._save_mask_verification(guitar_points, cam_w, cam_h)

        self.logger.info(
            "Guitar mask created with %d polygon points, cam=%dx%d",
            len(guitar_points), cam_w, cam_h
        )

    def _save_mask_verification(self, guitar_points, cam_w, cam_h):
        """Save a debug image showing the camera frame with mask polygons overlaid."""
        import os as _os
        _base = _os.path.dirname(_os.path.abspath(__file__))
        try:
            # Use the illuminate reference as the background (shows the scene)
            illum_ref = self.worker._calib_illum_ref
            if illum_ref is None:
                return
            verify = cv2.cvtColor(illum_ref, cv2.COLOR_GRAY2BGR)

            # Draw guitar mask polygon in RED
            guitar_np = np.array(guitar_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(verify, [guitar_np], True, (0, 0, 255), 3)
            cv2.fillPoly(verify, [guitar_np], (0, 0, 100))  # semi-transparent red fill

            # Draw background mask polygon in BLUE
            for mask in self.masks:
                if mask.name == "Background":
                    bg_pts = np.array(mask.source_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(verify, [bg_pts], True, (255, 100, 0), 2)

            # Label
            cv2.putText(verify, "RED=Guitar mask, BLUE=Background",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(verify, f"Guitar polygon: {len(guitar_points)} points",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imwrite(_os.path.join(_base, "mask_verification.png"), verify)
            self.logger.info("Mask verification image saved to mask_verification.png")
        except Exception as e:
            self.logger.warning("Failed to save mask verification: %s", e)

    def _update_background_exclude_guitar(self, guitar_points, cam_w, cam_h):
        """Update the Background mask to cover the full camera frame.
        Guitar mask renders AFTER background and overwrites its area,
        so no explicit exclusion is needed — just ensure background
        covers the entire frame."""
        bg_points = [(0, 0), (cam_w, 0), (cam_w, cam_h), (0, cam_h)]
        for mask in self.masks:
            if mask.name == "Background":
                mask.source_points = bg_points
                self.logger.info("Background mask updated to %dx%d", cam_w, cam_h)
                break

    def _start_calibration(self):
        """Trigger guitar detection + homography calibration sequence."""
        self.logger.info("User triggered calibration")
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.setText("Calibrating...")
        self.worker.start_calibration()
        # Re-enable button after calibration completes (via markers_calibrated signal)

    def run_full_calibration_wizard_safe(self):
        try:
            self.run_full_calibration_wizard()
        except Exception:
            self.logger.exception("Calibration wizard crashed")
            QMessageBox.critical(
                self,
                "Calibration Error",
                "Calibration wizard crashed. Check terminal logs for stack trace and retry.",
            )

    def run_full_calibration_wizard(self):
        # Stage 1: room scan + projector bounds
        self.projector_window.set_pattern_mode(False)
        QApplication.processEvents()
        time.sleep(0.25)
        still_off = self.capture_still_frame_warmed("projector_off", warmup_ms=350, samples=2)
        if still_off is None:
            QMessageBox.warning(self, "Calibration", "Could not capture a camera frame for room scan (projector off frame).")
            return

        self.projector_window.set_pattern_mode(True, brightness=255)
        QApplication.processEvents()
        time.sleep(0.35)
        still_on = self.capture_still_frame_warmed("projector_on", warmup_ms=900, samples=3)
        self.projector_window.set_pattern_mode(False)

        if still_on is None:
            QMessageBox.warning(self, "Calibration", "Could not capture a camera frame for room scan (projector on frame).")
            return

        still = still_on
        bounds, auto_accepted, confidence_msg = self.detect_projector_bounds(still_off, still_on)
        initial_points = []
        if bounds:
            initial_points = [
                QPoint(int(p[0] * still.width()), int(p[1] * still.height()))
                for p in bounds
            ]
            self.logger.info("Auto projector bounds detected: %s", bounds)
        else:
            self.logger.warning("Auto projector bounds unavailable; seeding with full-frame corners")
            initial_points = [
                QPoint(0, 0),
                QPoint(still.width() - 1, 0),
                QPoint(still.width() - 1, still.height() - 1),
                QPoint(0, still.height() - 1),
            ]

        # Improvement 1: if confidence thresholds are met, skip the confirmation dialog
        if auto_accepted and bounds:
            self.logger.info("Projector bounds auto-accepted: %s", confidence_msg)
            self.statusBar().showMessage(confidence_msg, 5000)
            confirmed_points = initial_points
        else:
            bounds_dialog = PolygonMaskDialog("Confirm projector bounds", self)
            bounds_dialog.set_pixmap(QPixmap.fromImage(still))
            if initial_points:
                bounds_dialog.set_points(initial_points)

            QMessageBox.information(
                self,
                "Stage 1",
                "Review the auto-scanned projector bounds. Drag/click 4 points if needed, then confirm.",
            )
            if not bounds_dialog.exec_() or len(bounds_dialog.get_points()) != 4:
                QMessageBox.warning(self, "Stage 1", "Projector bounds were not confirmed.")
                return

            confirmed_points = bounds_dialog.get_points()
        normalized = [
            [p.x() / max(still.width(), 1), p.y() / max(still.height(), 1)]
            for p in confirmed_points
        ]
        self.projector_window.warp_points = self.projector_window.deserialize_warp_points(normalized)
        self.worker.set_warp_points(self.projector_window.get_warp_points_normalized())

        background_dialog = PolygonMaskDialog("Draw background mask (unlimited points)", self)
        background_dialog.set_pixmap(QPixmap.fromImage(still))
        background_dialog.set_points(confirmed_points)
        QMessageBox.information(
            self,
            "Stage 1",
            "Now draw the background mask. Add as many points as needed, then confirm.",
        )
        if not background_dialog.exec_() or len(background_dialog.get_points()) < 3:
            QMessageBox.warning(self, "Stage 1", "Background mask was not confirmed.")
            return

        bg_points_q = background_dialog.get_points()
        bg_points = [(p.x(), p.y()) for p in bg_points_q]
        self.ensure_mask("Background", bg_points, mask_type="static", linked_marker_count=0)
        self.logger.info("Background mask saved with %d points", len(bg_points))
        overlay = self._draw_polygon_overlay(still, bg_points_q, Qt.yellow)
        if overlay is not None:
            self.update_projector_preview(overlay)

        QMessageBox.information(
            self,
            "Stage 1 Complete",
            "Projector bounds and background mask applied. Continue to marker selection.",
        )

        # Stage 2: uploaded reference + live marker alignment + guitar mask + depth baseline
        reference_markers = self.select_reference_guitar_markers()
        if len(reference_markers) != 4:
            QMessageBox.warning(self, "Stage 2", "Upload a guitar image and select exactly 4 reference markers to continue.")
            return

        self.reference_markers = reference_markers

        QMessageBox.information(
            self,
            "Stage 2",
            "Now tune threshold until only live IR blobs are visible, then capture and select the same 4 markers in live video.",
        )
        self.open_marker_selection_dialog()
        if len(self.selected_markers) != 4:
            QMessageBox.warning(self, "Stage 2", "Select exactly 4 live IR markers on the guitar to continue.")
            return

        still2 = self.capture_still_frame_warmed("guitar_mask", warmup_ms=250, samples=2)
        if still2 is None:
            QMessageBox.warning(self, "Stage 2", "Could not capture still frame for guitar mask.")
            return

        auto_source = [(p.x(), p.y()) for p in self.selected_markers[:4]]
        auto_source = self._order_quad_points(auto_source)
        source = [(int(p[0]), int(p[1])) for p in auto_source]

        self.ensure_mask("Guitar", source, mask_type="dynamic", linked_marker_count=4)
        self.logger.info("Guitar mask saved with %d points", len(source))
        self.worker.set_marker_points(self.selected_markers[:4])
        self.worker.calibrate_depth()

        refine = QMessageBox.question(
            self,
            "Stage 2",
            "Auto-created guitar mask from IR markers. Refine manually?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if refine == QMessageBox.Yes:
            guitar_dialog = PolygonMaskDialog("Refine 4-point guitar mask", self)
            guitar_dialog.set_pixmap(QPixmap.fromImage(still2))
            guitar_dialog.set_points([QPoint(x, y) for x, y in source])
            if guitar_dialog.exec_() and len(guitar_dialog.get_points()) == 4:
                source = [(p.x(), p.y()) for p in guitar_dialog.get_points()]
                self.ensure_mask("Guitar", source, mask_type="dynamic", linked_marker_count=4)
        self.logger.info("Guitar mask saved with %d points", len(source))

        # Stage 3: amp mask
        still3 = self.capture_still_frame_warmed("amp_mask", warmup_ms=250, samples=2)
        if still3 is None:
            QMessageBox.warning(self, "Stage 3", "Could not capture still frame for amp mask.")
            return
        amp_dialog = PolygonMaskDialog("Draw 4-point amp mask", self)
        amp_dialog.set_pixmap(QPixmap.fromImage(still3))
        if amp_dialog.exec_() and len(amp_dialog.get_points()) == 4:
            amp_source = [(p.x(), p.y()) for p in amp_dialog.get_points()]
            self.ensure_mask("Amp", amp_source, mask_type="static", linked_marker_count=0)
            self.logger.info("Amp mask saved with %d points", len(amp_source))

        self.worker.set_masks(self.masks)
        self.statusBar().showMessage("Calibration wizard complete. Assign cues to masks in the Cues list.", 6000)

    def refresh_cameras(self, initial=False):
        selected_camera = None
        if self.available_cameras and self.camera_combo.currentIndex() >= 0:
            idx = self.camera_combo.currentIndex()
            if idx < len(self.available_cameras):
                selected_camera = self.available_cameras[idx]

        self.available_cameras = get_available_cameras()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        if self.available_cameras:
            self.camera_combo.setEnabled(True)
            self.camera_combo.addItems([f"Camera {i}" for i in self.available_cameras])
            if selected_camera in self.available_cameras:
                self.camera_combo.setCurrentIndex(self.available_cameras.index(selected_camera))
            else:
                self.camera_combo.setCurrentIndex(0)
            if not initial:
                self.change_camera(self.camera_combo.currentIndex())
        else:
            self.camera_combo.addItem("No camera detected")
            self.camera_combo.setEnabled(False)
            if not initial:
                self.statusBar().showMessage("No cameras found.", 3000)

        self.camera_combo.blockSignals(False)

    def retry_camera(self):
        self.worker.retry_camera()
        self.statusBar().showMessage("Retrying camera connection...", 2000)

    def update_camera_mode(self, _index):
        self.worker.set_camera_mode(self.camera_mode_combo.currentData())

    @pyqtSlot(str)
    def update_camera_info(self, message):
        self.camera_info_label.setText(message)

    def calibrate_depth(self):
        self.worker.calibrate_depth()
        self.depth_calibration_label.setText("Calibrating...")

    def update_depth_sensitivity(self, value):
        self.worker.set_depth_sensitivity(value / 100.0)

    @pyqtSlot(int)
    def show_camera_error(self, index):
        self.statusBar().showMessage(f"Camera {index} disconnected — retrying…", 5000)
        self.logger.warning("Camera error on source %s — scheduling auto-retry", index)
        if not hasattr(self, '_camera_retry_count'):
            self._camera_retry_count = 0
        self._camera_retry_count += 1
        if self._camera_retry_count <= 5:
            delay_ms = min(3000 * self._camera_retry_count, 15000)
            QTimer.singleShot(delay_ms, self._auto_retry_camera)
        else:
            self.statusBar().showMessage(
                f"Camera {index} failed after 5 retries. Check connection and click Retry.", 0)

    def _auto_retry_camera(self):
        self.logger.info("Auto-retrying camera connection (attempt %d)",
                         getattr(self, '_camera_retry_count', 1))
        self.worker.retry_camera()

    @pyqtSlot()
    def _on_worker_stopped(self):
        """Called when the worker thread exits — expected on close, unexpected otherwise."""
        if not self._is_closing:
            self.logger.error("Worker thread stopped unexpectedly")
            result = QMessageBox.critical(
                self, "Worker Stopped",
                "The video processing thread stopped unexpectedly.\n\n"
                "Check ir_tracker.log for details.\n\n"
                "Retry will attempt to restart the camera.",
                QMessageBox.Retry | QMessageBox.Close,
                QMessageBox.Retry,
            )
            if result == QMessageBox.Retry:
                # Only restart if the thread is actually finished — avoid double-start
                if not self.thread.isRunning():
                    self.worker._running = True  # re-arm the worker loop flag
                    self.worker.retry_camera()
                    self.thread.start()
                else:
                    self.logger.warning("Worker thread restart requested but thread is still running")

    @pyqtSlot(str, int, int)
    def _on_calibration_progress(self, phase, current, total):
        if not hasattr(self, '_calib_progress_dialog') or self._calib_progress_dialog is None:
            self._calib_progress_dialog = QProgressDialog(
                f"Calibrating: {phase}", "Cancel", 0, total, self)
            self._calib_progress_dialog.setWindowTitle("Calibration")
            self._calib_progress_dialog.setWindowModality(Qt.WindowModal)
            self._calib_progress_dialog.setMinimumDuration(0)
            self._calib_progress_dialog.setValue(0)
            self._calib_progress_dialog.show()
            self._calib_progress_dialog.canceled.connect(self._cancel_calibration)
            # Safety timeout: close dialog and notify operator if calibration stalls (#33)
            self._calib_timeout_timer = QTimer(self)
            self._calib_timeout_timer.setSingleShot(True)
            self._calib_timeout_timer.timeout.connect(self._on_calibration_timeout)
            self._calib_timeout_timer.start(90_000)
        _phase_instructions = {
            "DARK": "Turn OFF all stage lights — projector must be OFF",
            "ILLUMINATE": "Turn projector to FULL WHITE — keep markers in frame",
            "DETECT": "Hold still — detecting IR marker positions...",
            "PROJ_SCAN": "Do not block camera — computing projector mapping...",
            "Error — restarting": "Calibration restarting — check camera and lighting",
        }
        instruction = _phase_instructions.get(phase, f"Calibrating: {phase}")
        self._calib_progress_dialog.setLabelText(instruction)
        self._calib_progress_dialog.setMaximum(total)
        self._calib_progress_dialog.setValue(current)
        if current >= total:
            self._close_calibration_dialog()

    def _cancel_calibration(self):
        """Operator cancels calibration mid-run — return to idle without starting a new run."""
        self.worker.reset_calibration()
        self._close_calibration_dialog()
        if hasattr(self, 'calibrate_button'):
            self.calibrate_button.setEnabled(True)
            self.calibrate_button.setText("Calibrate Guitar + Homography")
        self.statusBar().showMessage("Calibration cancelled.", 3000)

    def _status_message(self, msg: str, level: str = "info", timeout_ms: int = 5000):
        """Show a status bar message with color coding by level."""
        colors = {"info": "", "warning": "color: orange", "error": "color: red", "success": "color: #00cc44"}
        self.statusBar().setStyleSheet(f"QStatusBar {{ {colors.get(level, '')} }}")
        self.statusBar().showMessage(msg, timeout_ms)
        if level == "info":
            # Reset color after timeout
            QTimer.singleShot(timeout_ms + 100, lambda: self.statusBar().setStyleSheet(""))

    def _close_calibration_dialog(self):
        """Close calibration dialog and cancel its safety timer."""
        if hasattr(self, '_calib_timeout_timer') and self._calib_timeout_timer:
            self._calib_timeout_timer.stop()
            self._calib_timeout_timer = None
        if hasattr(self, '_calib_progress_dialog') and self._calib_progress_dialog:
            self._calib_progress_dialog.close()
            self._calib_progress_dialog = None

    def _on_calibration_timeout(self):
        """Called when calibration dialog has been open for 90 s without completing."""
        if self._is_closing:
            return
        if not hasattr(self, '_calib_progress_dialog'):
            return
        self._close_calibration_dialog()
        QMessageBox.warning(
            self, "Calibration Timed Out",
            "Calibration did not complete within 90 seconds.\n\n"
            "Ensure the projector is on and IR markers are visible, then try again."
        )

    @pyqtSlot()
    def _on_screens_changed(self, _screen=None):
        """Repopulate the projector combo when displays are connected or removed."""
        current_idx = self.projector_combo.currentIndex()
        self.screens = QApplication.screens()
        self.projector_combo.blockSignals(True)
        self.projector_combo.clear()
        for i, s in enumerate(self.screens):
            self.projector_combo.addItem(f"Screen {i + 1}: {s.name()} ({s.geometry().width()}×{s.geometry().height()})")
        safe_idx = min(current_idx, self.projector_combo.count() - 1)
        if safe_idx >= 0:
            self.projector_combo.setCurrentIndex(safe_idx)
        self.projector_combo.blockSignals(False)
        self.logger.info("Screen configuration changed — %d screen(s) detected", len(self.screens))

    @pyqtSlot(float)
    def _on_performance_degraded(self, fps):
        """Show a status-bar warning when sustained frame rate drops below 20 fps (#44)."""
        self.statusBar().showMessage(
            f"Warning: frame rate degraded to {fps:.0f} fps — check CPU load or camera connection",
            10_000
        )
        self.logger.warning("Performance degraded: %.1f fps", fps)

    @pyqtSlot(float, float, float, float, float, float)
    def update_performance_label(self, fps, frame_time_ms, detect_ms, match_ms, warp_ms, render_ms):
        self.performance_label.setText(
            f"FPS: {fps:.1f} | Frame: {frame_time_ms:.1f}ms | D:{detect_ms:.1f} M:{match_ms:.1f} W:{warp_ms:.1f} R:{render_ms:.1f}"
        )

    def enter_mask_creation_mode(self):
        self.video_display.set_mask_creation_mode(True)
        self.create_mask_button.setEnabled(False)
        self.finish_mask_button.setEnabled(True)
        self.cancel_mask_button.setEnabled(True)

    def finish_mask_creation(self):
        mask_points = list(self.video_display.get_mask_points())
        self.video_display.set_mask_creation_mode(False)

        if not mask_points:
            self.logger.debug("Finish Mask clicked with no points; nothing saved.")
        else:
            if len(mask_points) < 3:
                QMessageBox.warning(
                    self, "Not Enough Points",
                    "A mask needs at least 3 points to form a valid polygon.\n"
                    "Click 'Create Mask' and add more points."
                )
                self.create_mask_button.setEnabled(True)
                self.finish_mask_button.setEnabled(False)
                self.cancel_mask_button.setEnabled(False)
                self.video_display.clear_mask_points()
                self.mask_points_list.clear()
                return
            if len(mask_points) > 64:
                result = QMessageBox.question(
                    self, "Many Points",
                    f"This mask has {len(mask_points)} points which may slow rendering.\n"
                    "Use the simplified polygon instead (recommended)?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if result == QMessageBox.Yes:
                    import cv2 as _cv2
                    import numpy as _np
                    pts_arr = _np.array([(p.x(), p.y()) for p in mask_points], dtype=_np.int32)
                    hull = _cv2.convexHull(pts_arr)
                    epsilon = _cv2.arcLength(hull, True) * 0.01
                    approx = _cv2.approxPolyDP(hull, epsilon, True)
                    from PyQt5.QtCore import QPoint as _QPoint
                    mask_points = [_QPoint(int(p[0][0]), int(p[0][1])) for p in approx]
                    self.logger.info("Polygon simplified from %d to %d points",
                                     len(mask_points), len(approx))
            row = self.mask_list_widget.currentRow()
            source_points = [(p.x(), p.y()) for p in mask_points]
            if 0 <= row < len(self.masks):
                self.masks[row].source_points = source_points
                self.logger.debug("Updated mask '%s' with %d points.", self.masks[row].name, len(source_points))
            else:
                mask_name = f"Mask {len(self.masks) + 1}"
                mask = Mask(mask_name, source_points, None)
                mask.type = "static"
                self.masks.append(mask)
                self.logger.debug("Created new mask '%s' with %d points.", mask_name, len(source_points))
            self.worker.set_masks(self.masks)
            self.refresh_mask_views(select_index=len(self.masks) - 1)

        self.create_mask_button.setEnabled(True)
        self.finish_mask_button.setEnabled(False)
        self.cancel_mask_button.setEnabled(False)
        self.video_display.clear_mask_points()
        self.mask_points_list.clear()
        self.auto_sync_marker_links()

    def cancel_mask_creation(self):
        self.video_display.set_mask_creation_mode(False)
        self.create_mask_button.setEnabled(True)
        self.finish_mask_button.setEnabled(False)
        self.cancel_mask_button.setEnabled(False)
        self.video_display.clear_mask_points()
        self.mask_points_list.clear()

    def add_mask_point_to_list(self, point):
        self.mask_points_list.addItem(f"({point.x()}, {point.y()})")

    def link_mask_to_markers(self):
        current_item = self.mask_list_widget.currentItem()
        if not current_item:
            self.statusBar().showMessage("Please select a cue to link.", 3000)
            return

        if not self.selected_markers:
            self.statusBar().showMessage("Please select IR markers first.", 3000)
            return

        row = self.mask_list_widget.row(current_item)
        if 0 <= row < len(self.masks):
            mask = self.masks[row]
            marker_points = [(int(p.x()), int(p.y())) for p in self.selected_markers]
            if len(marker_points) < 4:
                self.statusBar().showMessage(
                    "Need 4 IR markers selected to link a dynamic mask.",
                    5000,
                )
                return

            mask.linked_marker_count = len(marker_points)
            mask.marker_anchor_points = marker_points
            mask.type = "dynamic"
            self.statusBar().showMessage(
                f"Mask '{mask.name}' linked to {len(marker_points)} markers (mask points: {len(mask.source_points)}).",
                4000,
            )

    def auto_sync_marker_links(self):
        if not self.auto_sync_checkbox.isChecked() or not self.selected_markers:
            return

        marker_count = len(self.selected_markers)
        linked = 0
        marker_points = [(int(p.x()), int(p.y())) for p in self.selected_markers]
        if marker_count < 4:
            return

        for mask in self.masks:
            if mask.type != "dynamic":
                continue
            mask.linked_marker_count = marker_count
            mask.marker_anchor_points = list(marker_points)
            linked += 1

        if linked:
            self.statusBar().showMessage(
                f"Auto-sync updated {linked} cue(s) to {marker_count} markers.",
                3500,
            )

    def update_ir_threshold(self, value):
        self.worker.set_ir_threshold(value)

    def update_threshold_mode(self, index):
        mode = "auto" if index == 1 else "manual"
        self.worker.set_threshold_mode(mode)
        self.ir_threshold_slider.setEnabled(mode == "manual")

    @pyqtSlot(int)
    def update_tracker_label(self, count):
        self.ir_trackers_label.setText(f"Trackers detected: {count}")

    def toggle_warping(self, checked):
        self.projector_window.set_calibration_mode(checked)
        self.enable_warping_button.setText("Disable Warping" if checked else "Enable Warping")

    def refresh_mask_views(self, select_index=None):
        self.mask_list_widget.blockSignals(True)
        self.cue_mask_combo.blockSignals(True)
        self.mask_list_widget.clear()
        self.cue_mask_combo.clear()
        for mask in self.masks:
            # Improvement: show disabled masks greyed out with suffix
            enabled = getattr(mask, 'enabled', True)
            display_name = mask.name if enabled else f"{mask.name} [OFF]"
            item = self.mask_list_widget.addItem(display_name)
            if not enabled:
                # Grey out disabled masks in the list
                from PyQt5.QtGui import QColor
                it = self.mask_list_widget.item(self.mask_list_widget.count() - 1)
                if it:
                    it.setForeground(QColor("#666666"))
            self.cue_mask_combo.addItem(display_name)
        self.mask_list_widget.blockSignals(False)
        self.cue_mask_combo.blockSignals(False)

        if self.masks:
            idx = select_index if select_index is not None else min(self.mask_list_widget.currentRow(), len(self.masks) - 1)
            if idx < 0:
                idx = 0
            self.mask_list_widget.setCurrentRow(idx)
            self.cue_mask_combo.setCurrentIndex(idx)
        self.refresh_cues_for_selected_mask()

    def on_mask_selection_changed(self, row):
        if 0 <= row < len(self.masks):
            self.worker.set_active_cue_index(row)
            if self.cue_mask_combo.currentIndex() != row:
                self.cue_mask_combo.setCurrentIndex(row)
        self.refresh_cues_for_selected_mask()

    def refresh_cues_for_selected_mask(self):
        idx = self.cue_mask_combo.currentIndex()
        self.mask_cue_list_widget.clear()
        if idx < 0 or idx >= len(self.masks):
            return
        mask = self.masks[idx]
        for i, cue in enumerate(mask.cues):
            cc_list = [str(cc) for cc, cue_index in mask.midi_cc_map.items() if cue_index == i]
            cc_suffix = f" [CC: {', '.join(cc_list)}]" if cc_list else ""
            active = "* " if i == mask.active_cue else ""
            # Show a warning icon if the cue file is missing
            exists = Path(cue).exists() if cue else True
            missing_flag = " ⚠" if not exists else ""
            self.mask_cue_list_widget.addItem(f"{active}{Path(cue).name}{cc_suffix}{missing_flag}")
        if mask.cues:
            self.mask_cue_list_widget.setCurrentRow(mask.active_cue)
        # Sync loop mode combo
        if hasattr(self, 'cue_loop_combo'):
            loop_mode = getattr(mask, 'loop_mode', 'loop')
            combo_idx = self.cue_loop_combo.findData(loop_mode)
            if combo_idx >= 0:
                self.cue_loop_combo.blockSignals(True)
                self.cue_loop_combo.setCurrentIndex(combo_idx)
                self.cue_loop_combo.blockSignals(False)

    def on_mask_cue_selected(self, row):
        idx = self.cue_mask_combo.currentIndex()
        if idx < 0 or idx >= len(self.masks):
            return
        mask = self.masks[idx]
        if 0 <= row < len(mask.cues):
            mask.active_cue = row
            self.worker.set_masks(self.masks)

    def add_cue(self):
        idx = self.cue_mask_combo.currentIndex()
        if idx < 0 or idx >= len(self.masks):
            self.statusBar().showMessage("Create/select a mask first.", 3000)
            return
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if not video_path:
            return
        self.masks[idx].add_cue(video_path)
        self.worker.set_masks(self.masks)
        self.refresh_cues_for_selected_mask()

    def remove_cue(self):
        idx = self.cue_mask_combo.currentIndex()
        cue_idx = self.mask_cue_list_widget.currentRow()
        if idx < 0 or idx >= len(self.masks) or cue_idx < 0:
            return
        cue_name = Path(self.masks[idx].cues[cue_idx]).name
        if QMessageBox.question(
            self, "Remove Cue", f"Remove cue '{cue_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) != QMessageBox.Yes:
            return
        self.masks[idx].remove_cue(cue_idx)
        self.worker.set_masks(self.masks)
        self.refresh_cues_for_selected_mask()

    def _preview_selected_cue(self):
        """Play the selected cue in the preview panel only — projector unaffected (#35)."""
        idx = self.cue_mask_combo.currentIndex()
        cue_idx = self.mask_cue_list_widget.currentRow()
        if idx < 0 or idx >= len(self.masks):
            return
        mask = self.masks[idx]
        if cue_idx < 0 or cue_idx >= len(mask.cues):
            cue_idx = mask.active_cue
        if not mask.cues:
            self.statusBar().showMessage("No cue selected.", 3000)
            return
        cue_path = mask.cues[cue_idx]
        # Preview: temporarily enable preview mode, disable projector output
        prev_active_cue = self.worker.active_cue_index
        self.worker.set_preview_enabled(True)
        self.worker.set_active_cue_index(idx)
        self.statusBar().showMessage(f"Previewing: {Path(cue_path).name} (click Stop Preview to end)", 0)
        # Restore after 5 s if operator forgets to stop
        if not hasattr(self, '_preview_restore_timer'):
            self._preview_restore_timer = QTimer(self)
            self._preview_restore_timer.setSingleShot(True)
            self._preview_restore_timer.timeout.connect(self._stop_cue_preview)
        self._preview_restore_timer.start(5000)
        self._preview_prev_active_cue = prev_active_cue

    def _stop_cue_preview(self):
        """Stop cue preview and restore previous render state (#35)."""
        if hasattr(self, '_preview_restore_timer'):
            self._preview_restore_timer.stop()
        prev = getattr(self, '_preview_prev_active_cue', -1)
        self.worker.set_active_cue_index(prev)
        self.statusBar().showMessage("Cue preview stopped.", 3000)

    def map_cc_to_selected_cue(self):
        idx = self.cue_mask_combo.currentIndex()
        cue_idx = self.mask_cue_list_widget.currentRow()
        if idx < 0 or idx >= len(self.masks):
            return
        if cue_idx < 0:
            cue_idx = self.masks[idx].active_cue
        cc = int(self.cc_spin.value())
        self.masks[idx].midi_cc_map[cc] = cue_idx
        self.refresh_cues_for_selected_mask()
        self.statusBar().showMessage(f"Mapped CC {cc} to cue {cue_idx + 1} on '{self.masks[idx].name}'.", 3000)

    def refresh_midi_inputs(self):
        self.midi_input_combo.clear()
        if mido is None:
            self.midi_input_combo.addItem("mido not installed")
            self.midi_input_combo.setEnabled(False)
            return
        names = mido.get_input_names()
        if not names:
            self.midi_input_combo.addItem("No MIDI input ports")
            self.midi_input_combo.setEnabled(False)
            return
        self.midi_input_combo.setEnabled(True)
        self.midi_input_combo.addItems(names)

    def connect_midi_input(self):
        if mido is None or not self.midi_input_combo.isEnabled():
            self.statusBar().showMessage("MIDI backend unavailable.", 3000)
            return
        if self.midi_inport is not None:
            self.midi_inport.close()
            self.midi_inport = None
        name = self.midi_input_combo.currentText()
        self.midi_inport = None
        try:
            self.midi_inport = mido.open_input(name)
            self.statusBar().showMessage(f"Connected MIDI input: {name}", 3000)
            # Improvement: update MIDI status label
            if hasattr(self, 'midi_status_label'):
                self.midi_status_label.setText(f"MIDI: {name}")
                self.midi_status_label.setProperty("class", "status-ok")
                self.midi_status_label.style().unpolish(self.midi_status_label)
                self.midi_status_label.style().polish(self.midi_status_label)
            if hasattr(self, 'midi_activity_label'):
                self.midi_activity_label.setStyleSheet("color: #00cc44; font-size: 16px;")
        except Exception as exc:
            self.logger.exception("Failed to open MIDI input '%s'", name)
            if self.midi_inport is not None:
                try:
                    self.midi_inport.close()
                except Exception:
                    pass
                self.midi_inport = None
            if hasattr(self, 'midi_status_label'):
                self.midi_status_label.setText("MIDI: Error")
                self.midi_status_label.setProperty("class", "status-error")
                self.midi_status_label.style().unpolish(self.midi_status_label)
                self.midi_status_label.style().polish(self.midi_status_label)
            QMessageBox.critical(self, "MIDI Error", f"Could not open MIDI port '{name}'.\n\nError: {exc}")

    def poll_midi_messages(self):
        if self.midi_inport is None:
            return
        try:
            for msg in self.midi_inport.iter_pending():
                if msg.type == "control_change":
                    self.route_midi_cc(int(msg.control), int(msg.value))
        except Exception:
            self.logger.exception("MIDI polling error — closing port")
            try:
                if self.midi_inport:
                    self.midi_inport.close()
            except Exception:
                pass
            self.midi_inport = None
            self.statusBar().showMessage("MIDI port disconnected — reconnect in MIDI panel", 10_000)

    def route_midi_cc(self, cc, value):
        if value <= 0:
            return
        matched = False
        for mask in self.masks:
            if cc in mask.midi_cc_map:
                cue_idx = mask.midi_cc_map[cc]
                if 0 <= cue_idx < len(mask.cues):
                    mask.active_cue = cue_idx
                    matched = True
                    self.logger.debug("MIDI CC %d value %d → mask '%s' cue %d",
                                      cc, value, mask.name, cue_idx)
        if matched:
            self.worker.set_masks(self.masks)
            self.refresh_cues_for_selected_mask()
        # Flash MIDI activity LED regardless of match (shows MIDI is live)
        if hasattr(self, 'midi_activity_label'):
            self.midi_activity_label.setStyleSheet("color: #ffcc00; font-size: 16px;")
            QTimer.singleShot(120, lambda: self.midi_activity_label.setStyleSheet(
                "color: #00cc44; font-size: 16px;" if self.midi_inport else "color: #444; font-size: 16px;"
            ))

    def remove_mask(self):
        row = self.mask_list_widget.currentRow()
        if 0 <= row < len(self.masks):
            mask_name = self.masks[row].name
            if QMessageBox.question(
                self, "Remove Mask", f"Remove mask '{mask_name}' and all its cues?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            ) != QMessageBox.Yes:
                return
            del self.masks[row]
            self.worker.set_masks(self.masks)
            self.refresh_mask_views(select_index=max(0, row - 1))

    def change_camera(self, index):
        if self.available_cameras and 0 <= index < len(self.available_cameras):
            self.worker.set_video_source(self.available_cameras[index])

    def change_projector(self, index):
        self.screens = QApplication.screens()
        if index < 0 or index >= len(self.screens):
            return

        screen = self.screens[index]
        target_geometry = screen.geometry()

        # Ensure the window has a native handle before assigning a screen.
        self.projector_window.showNormal()
        QApplication.processEvents()

        window_handle = self.projector_window.windowHandle()
        if window_handle is not None:
            window_handle.setScreen(screen)

        self.projector_window.setGeometry(target_geometry)
        self.projector_window.move(target_geometry.topLeft())
        self.projector_window.showFullScreen()
        self.projector_window.raise_()
        self.projector_window.activateWindow()

        # Sync projector resolution to worker so warpPerspective output matches
        proj_w = max(1, target_geometry.width())
        proj_h = max(1, target_geometry.height())
        self.worker._proj_resolution = (proj_w, proj_h)
        self.logger.info(
            "Projector display changed to index=%d name=%s geometry=%s resolution=%dx%d",
            index,
            screen.name(),
            target_geometry,
            proj_w,
            proj_h,
        )

    # ─── New slot implementations ───────────────────────────────────────────

    def _toggle_blackout(self, checked: bool) -> None:
        """Improvement 76: toggle projector blackout."""
        self.worker.set_blackout(checked)
        self.blackout_button.setText("BLACKOUT (ON)" if checked else "BLACKOUT")
        self._status_message(
            "Projector BLACKED OUT" if checked else "Blackout cleared",
            "error" if checked else "success", 4000
        )

    def _toggle_lock_mode(self, checked: bool) -> None:
        """Improvement 77: lock/unlock editing controls."""
        self._lock_mode = checked
        self.lock_mode_button.setText("Unlock UI" if checked else "Lock UI")
        editable_widgets = [
            self.create_mask_button, self.remove_mask_button, self.rename_mask_button,
            self.toggle_mask_button, self.move_mask_up_button, self.move_mask_down_button,
            self.export_masks_button, self.import_masks_button,
            self.add_cue_button, self.remove_cue_button, self.map_cc_button,
            self.link_mask_button, self.auto_sync_checkbox,
            self.select_markers_button, self.clear_markers_button,
            self.reset_calibration_button, self.calibrate_button,
            self.setup_wizard_button,
        ]
        for w in editable_widgets:
            if hasattr(w, 'setEnabled'):
                w.setEnabled(not checked)
        self._status_message(
            "UI LOCKED — editing disabled" if checked else "UI unlocked",
            "warning" if checked else "success", 4000
        )

    def _save_screenshot(self) -> None:
        """Improvement 78: save current camera frame as PNG (timestamped filename)."""
        if self.latest_camera_qimage is None:
            self._status_message("No frame available yet.", "warning", 3000)
            return
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        default_name = f"screenshot_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", default_name, "PNG Images (*.png)"
        )
        if path:
            if self.latest_camera_qimage.save(path):
                self._status_message(f"Screenshot saved: {Path(path).name}", "success", 4000)
                self.logger.info("Screenshot saved: %s", path)
            else:
                self._status_message("Failed to save screenshot.", "error", 4000)

    def _run_preshow_checklist(self) -> None:
        """Improvement 79: pre-show readiness check."""
        lines = []
        ok = lambda s: f"✓  {s}"
        fail = lambda s: f"✗  {s}"
        warn = lambda s: f"⚠  {s}"

        # Camera
        if self.available_cameras:
            lines.append(ok(f"Camera detected (index {self.camera_combo.currentIndex()})"))
        else:
            lines.append(fail("No camera detected — check USB connection"))

        # Calibration
        info = self.worker.get_calibration_info()
        if info["calibrated"]:
            lines.append(ok(f"Calibrated ({info['marker_count']} markers)"))
        else:
            lines.append(fail("Not calibrated — run calibration before show"))

        # Masks
        if self.masks:
            enabled = [m for m in self.masks if getattr(m, 'enabled', True)]
            lines.append(ok(f"{len(enabled)}/{len(self.masks)} mask(s) enabled"))
        else:
            lines.append(warn("No masks defined"))

        # Cue validation
        missing_cues = []
        for mask in self.masks:
            missing_cues.extend(mask.validate_cues())
        if missing_cues:
            lines.append(fail(f"{len(missing_cues)} cue file(s) missing:"))
            for c in missing_cues[:3]:
                lines.append(f"    {Path(c).name}")
        else:
            lines.append(ok("All cue files present"))

        # MIDI
        if self.midi_inport is not None:
            lines.append(ok(f"MIDI connected: {self.midi_input_combo.currentText()}"))
        else:
            lines.append(warn("MIDI not connected (optional)"))

        # Projector
        lines.append(ok(f"Projector on: {self.projector_combo.currentText()}"))

        # Homography
        if info["has_homography"]:
            lines.append(ok("Camera→projector homography computed"))
        else:
            lines.append(warn("No homography — run full calibration wizard for best results"))

        all_ok = not any(l.startswith("✗") for l in lines)
        icon = QMessageBox.Information if all_ok else QMessageBox.Warning
        QMessageBox.information(
            self, "Pre-Show Checklist",
            "\n".join(lines)
        ) if all_ok else QMessageBox.warning(
            self, "Pre-Show Checklist — Issues Found",
            "\n".join(lines)
        )

    def _reset_calibration(self) -> None:
        """Improvement 80: reset calibration without triggering a new run."""
        if QMessageBox.question(
            self, "Reset Calibration",
            "Clear the calibration cache?\n\nTracking will continue in uncalibrated global-search mode "
            "until you run calibration again.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) != QMessageBox.Yes:
            return
        self.worker.reset_calibration()
        self._calibration_timestamp = None
        self._update_calibration_age_label()
        self.calib_state_label.setText("State: Not calibrated")
        self.calib_state_label.setProperty("class", "status-error")
        self.calib_state_label.style().unpolish(self.calib_state_label)
        self.calib_state_label.style().polish(self.calib_state_label)
        self._status_message("Calibration reset.", "warning", 4000)

    @pyqtSlot(str)
    def _on_tracking_state_changed(self, state: str) -> None:
        """Improvement 81: update UI when tracking state changes."""
        _map = {
            "tracking": ("Tracking: ACTIVE", "status-ok"),
            "lost": ("Tracking: LOST", "status-warn"),
            "calibrating": ("Calibrating…", "status-warn"),
            "idle": ("Tracking: idle", "status-idle"),
        }
        text, css_class = _map.get(state, (f"State: {state}", "status-idle"))
        self.tracking_state_diag_label.setText(text)
        self.tracking_state_diag_label.setProperty("class", css_class)
        self.tracking_state_diag_label.style().unpolish(self.tracking_state_diag_label)
        self.tracking_state_diag_label.style().polish(self.tracking_state_diag_label)
        # Improvement 82: window title reflects tracking state
        state_icons = {"tracking": "◉", "lost": "◌", "calibrating": "◎", "idle": "○"}
        icon = state_icons.get(state, "○")
        self.setWindowTitle(f"{icon} IR Guitar Tracker v{__version__}  [{state.upper()}]")

    @pyqtSlot()
    @pyqtSlot(list)
    def _on_guitar_candidates_ready(self, candidates: list) -> None:
        """Improvement 10: handle multiple guitar-shaped candidates detected during calibration.

        If only 1 candidate is present, this slot is not called (best candidate is used directly).
        If >1 candidates: log a message, update the status bar, and after 5 seconds auto-confirm
        the best candidate (which the worker has already stored as self._guitar_polygon).
        """
        n = len(candidates)
        self.logger.info("Multiple guitar candidates detected: %d — auto-selecting best in 5s", n)
        if n <= 1:
            return  # nothing to do
        msg = f"Multiple guitar shapes detected ({n}). Auto-selecting best match in 5 seconds..."
        self._status_message(msg, "info", 6000)
        # After 5 seconds, log confirmation (the worker already selected the best candidate)
        QTimer.singleShot(5000, lambda: self.logger.info(
            "Guitar candidate auto-selection confirmed (best of %d candidates).", n
        ))

    def _on_calibration_restored(self) -> None:
        """Improvement 83: update UI when calibration loads from cache."""
        self._calibration_timestamp = time.time()
        self._update_calibration_age_label()
        self.calib_state_label.setText("State: Calibrated (restored)")
        self.calib_state_label.setProperty("class", "status-ok")
        self.calib_state_label.style().unpolish(self.calib_state_label)
        self.calib_state_label.style().polish(self.calib_state_label)
        self._status_message("Calibration restored from cache.", "success", 5000)

    @pyqtSlot(dict)
    def _on_diagnostic_info(self, info: dict) -> None:
        """Improvement 84: update FPS history and blob count from worker diagnostics."""
        fps = info.get("fps", 0.0)
        self._fps_history.append(fps)
        if len(self._fps_history) > 60:
            self._fps_history.pop(0)
        if self._fps_history:
            avg = sum(self._fps_history) / len(self._fps_history)
            mn = min(self._fps_history)
            self.fps_history_label.setText(f"FPS avg/min: {avg:.1f}/{mn:.1f} (last {len(self._fps_history)}s)")
        detected = info.get("detected", 0)
        tracked = info.get("tracked", 0)
        blobs = info.get("blob_history", 0)
        self.blob_count_label.setText(f"Blobs hist: {blobs}  Det: {detected}  Trk: {tracked}")

    def _update_calibration_age_label(self) -> None:
        """Improvement 85: show how long ago the last calibration ran."""
        if not hasattr(self, 'calib_state_label'):
            return
        if self._calibration_timestamp is None:
            return
        age_s = time.time() - self._calibration_timestamp
        if age_s < 60:
            age_str = f"{int(age_s)}s ago"
        elif age_s < 3600:
            age_str = f"{int(age_s/60)}m ago"
        else:
            age_str = f"{age_s/3600:.1f}h ago"
        self.calib_state_label.setText(f"State: Calibrated ({age_str})")

    def _auto_save(self) -> None:
        """Improvement 86: periodic auto-save so show config survives a crash."""
        self.save_settings()
        self.logger.info("Auto-saved settings")

    def _check_midi_hotplug(self) -> None:
        """Poll for newly connected or removed MIDI input ports every 2 seconds.
        Updates the combo box silently; does not disconnect the active port."""
        if mido is None:
            return
        try:
            current_names = mido.get_input_names()
        except Exception:
            return
        if current_names != self._midi_port_names:
            self.logger.info(
                "MIDI ports changed: was %s, now %s",
                self._midi_port_names, current_names
            )
            self._midi_port_names = current_names
            # Repopulate combo without disconnecting the active port
            selected = self.midi_input_combo.currentText()
            self.midi_input_combo.blockSignals(True)
            self.midi_input_combo.clear()
            if current_names:
                self.midi_input_combo.setEnabled(True)
                self.midi_input_combo.addItems(current_names)
                # Restore previous selection if still available
                idx = self.midi_input_combo.findText(selected)
                if idx >= 0:
                    self.midi_input_combo.setCurrentIndex(idx)
            else:
                self.midi_input_combo.addItem("No MIDI input ports")
                self.midi_input_combo.setEnabled(False)
            self.midi_input_combo.blockSignals(False)
            # Warn if the currently connected port disappeared
            if self.midi_inport is not None and selected not in current_names:
                self.logger.warning("Active MIDI port '%s' disappeared", selected)
                self.statusBar().showMessage(
                    f"MIDI port '{selected}' disconnected — reconnect in MIDI panel", 8000
                )
                try:
                    self.midi_inport.close()
                except Exception:
                    pass
                self.midi_inport = None
                if hasattr(self, 'midi_status_label'):
                    self.midi_status_label.setText("MIDI: Disconnected")
                if hasattr(self, 'midi_activity_label'):
                    self.midi_activity_label.setStyleSheet("color: #444; font-size: 16px;")

    def _check_memory(self) -> None:
        """Log memory usage snapshot every 5 minutes when IRTK_PROFILE_MEMORY is set."""
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                return
            snapshot = tracemalloc.take_snapshot()
            top = snapshot.statistics('lineno')[:5]
            self.logger.info(
                "Memory snapshot (top 5): %s",
                [(str(s.traceback), s.size // 1024) for s in top]
            )
            # Also log process RSS if psutil is available
            try:
                import psutil, os as _os
                proc = psutil.Process(_os.getpid())
                rss_mb = proc.memory_info().rss / 1024 / 1024
                self.logger.info("Process RSS: %.1f MB", rss_mb)
            except ImportError:
                pass
        except Exception as exc:
            self.logger.debug("Memory check error: %s", exc)

    def _notes_changed(self) -> None:
        """Improvement 87: track operator notes changes for saving."""
        self._operator_notes = self.operator_notes_edit.toPlainText()
        self.request_save_settings()

    def _rename_mask_inline(self, _item=None) -> None:
        """Improvement 88: rename selected mask via an input dialog."""
        row = self.mask_list_widget.currentRow()
        if not (0 <= row < len(self.masks)):
            return
        mask = self.masks[row]
        from PyQt5.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "Rename Mask", "New name:", text=mask.name
        )
        if ok and new_name.strip():
            mask.name = new_name.strip()
            self.worker.set_masks(self.masks)
            self.refresh_mask_views(select_index=row)
            self.request_save_settings()

    def _toggle_mask_enabled(self) -> None:
        """Improvement 89: toggle mask enabled/disabled."""
        row = self.mask_list_widget.currentRow()
        if not (0 <= row < len(self.masks)):
            return
        mask = self.masks[row]
        mask.enabled = not getattr(mask, 'enabled', True)
        self.worker.set_masks(self.masks)
        self.refresh_mask_views(select_index=row)
        self._status_message(
            f"Mask '{mask.name}' {'enabled' if mask.enabled else 'disabled'}.",
            "info", 3000
        )
        self.request_save_settings()

    def _move_mask_up(self) -> None:
        """Improvement 90: move mask up in render order."""
        row = self.mask_list_widget.currentRow()
        if row <= 0 or row >= len(self.masks):
            return
        self.masks[row - 1], self.masks[row] = self.masks[row], self.masks[row - 1]
        self.worker.set_masks(self.masks)
        self.refresh_mask_views(select_index=row - 1)
        self.request_save_settings()

    def _move_mask_down(self) -> None:
        """Improvement 91: move mask down in render order."""
        row = self.mask_list_widget.currentRow()
        if row < 0 or row >= len(self.masks) - 1:
            return
        self.masks[row], self.masks[row + 1] = self.masks[row + 1], self.masks[row]
        self.worker.set_masks(self.masks)
        self.refresh_mask_views(select_index=row + 1)
        self.request_save_settings()

    def _export_masks(self) -> None:
        """Improvement 92: export all masks to a JSON file."""
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Masks", "masks_export.json", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            data = {
                "version": _SETTINGS_VERSION,
                "exported_from": f"IR Guitar Tracker v{__version__}",
                "masks": [m.to_dict() for m in self.masks],
            }
            Path(path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8'
            )
            self._status_message(f"Exported {len(self.masks)} mask(s) to {Path(path).name}", "success", 4000)
        except Exception as exc:
            self.logger.exception("Export masks failed")
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _import_masks(self) -> None:
        """Improvement 93: import masks from a JSON file."""
        from PyQt5.QtWidgets import QFileDialog
        from mask import Mask as _Mask
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Masks", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            raw = Path(path).read_text(encoding='utf-8')
            data = json.loads(raw)
            mask_dicts = data.get("masks", [])
            if not mask_dicts:
                QMessageBox.warning(self, "Import Masks", "No masks found in the file.")
                return
            # Migrate imported masks to latest schema
            for md in mask_dicts:
                md.setdefault("enabled", True)
                md.setdefault("opacity", 1.0)
                md.setdefault("blend_mode", "normal")
                md.setdefault("loop_mode", "loop")
                md.setdefault("fade_in", 0.0)
                md.setdefault("fade_out", 0.0)
            imported = [_Mask.from_dict(d) for d in mask_dicts]
            if QMessageBox.question(
                self, "Import Masks",
                f"Replace all {len(self.masks)} current mask(s) with {len(imported)} imported mask(s)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            ) == QMessageBox.Yes:
                self.masks = imported
                self.worker.set_masks(self.masks)
                self.refresh_mask_views()
                self.request_save_settings()
                self._status_message(f"Imported {len(imported)} mask(s).", "success", 4000)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            QMessageBox.critical(self, "Import Failed",
                                 f"Could not parse file: {exc}")
        except Exception as exc:
            self.logger.exception("Import masks failed")
            QMessageBox.critical(self, "Import Failed", str(exc))

    def _advance_cue(self) -> None:
        """Improvement 94: manually advance to next cue on selected mask."""
        idx = self.cue_mask_combo.currentIndex()
        if not (0 <= idx < len(self.masks)):
            return
        mask = self.masks[idx]
        if not mask.cues:
            return
        mask.advance_cue()
        self.worker.set_masks(self.masks)
        self.refresh_cues_for_selected_mask()
        self._status_message(
            f"Mask '{mask.name}' → cue {mask.active_cue + 1}/{len(mask.cues)}",
            "info", 2000
        )

    def _update_cue_loop_mode(self, _index: int) -> None:
        """Improvement 95: sync loop mode combo to selected mask."""
        idx = self.cue_mask_combo.currentIndex()
        if not (0 <= idx < len(self.masks)):
            return
        mode = self.cue_loop_combo.currentData()
        self.masks[idx].loop_mode = mode
        self.worker.set_masks(self.masks)
        self.request_save_settings()

    def _reconnect_projector(self) -> None:
        """Improvement 96: re-detect screens and reconnect projector window."""
        self._on_screens_changed()
        self.change_projector(self.projector_combo.currentIndex())
        self._status_message("Projector reconnected.", "success", 3000)

    def _toggle_projector_visibility(self, hidden: bool) -> None:
        """Improvement 97: show or hide the projector output window."""
        if hidden:
            self.projector_window.hide()
            self.toggle_projector_button.setText("Show Projector")
        else:
            self.projector_window.showFullScreen()
            self.toggle_projector_button.setText("Hide Projector")

    def _show_about_dialog(self) -> None:
        """Improvement 98: show About dialog with version and dependency info."""
        import cv2 as _cv2
        from PyQt5 import QtCore as _QtCore
        msg = (
            f"<b>IR Guitar Tracker</b> v{__version__}<br><br>"
            f"Real-time IR marker tracking and projection mapping for live stage use.<br><br>"
            f"<b>Runtime versions:</b><br>"
            f"OpenCV: {_cv2.__version__}<br>"
            f"PyQt5: {_QtCore.PYQT_VERSION_STR}<br>"
            f"NumPy: {np.__version__}<br>"
            f"Python: {sys.version.split()[0]}<br><br>"
            f"Settings file: {SETTINGS_PATH.resolve()}<br>"
            f"Log file: ir_tracker.log"
        )
        QMessageBox.about(self, "About IR Guitar Tracker", msg)

    def _setup_shortcuts(self):
        from PyQt5.QtGui import QKeySequence
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_settings)
        QShortcut(QKeySequence("Ctrl+C"), self,
                  activated=lambda: self.calibrate_button.click() if hasattr(self, 'calibrate_button') else None)
        QShortcut(QKeySequence("Delete"), self, activated=self.remove_mask)
        QShortcut(QKeySequence("Space"), self,
                  activated=lambda: self.preview_checkbox.toggle() if hasattr(self, 'preview_checkbox') else None)
        # Esc for instant blackout toggle
        QShortcut(QKeySequence("Escape"), self,
                  activated=lambda: self.blackout_button.toggle() if hasattr(self, 'blackout_button') else None)
        # F1 for help / keyboard shortcuts cheatsheet
        QShortcut(QKeySequence("F1"), self, activated=self._show_shortcuts_help)
        # F2 to rename selected mask
        QShortcut(QKeySequence("F2"), self, activated=self._rename_mask_inline)
        # Ctrl+Right: advance cue on selected mask
        QShortcut(QKeySequence("Ctrl+Right"), self, activated=self._advance_cue)
        # Ctrl+A: render all masks
        QShortcut(QKeySequence("Ctrl+A"), self,
                  activated=lambda: self.worker.set_active_cue_index(-1))
        # Ctrl+Z: undo last mask point during mask creation
        QShortcut(QKeySequence("Ctrl+Z"), self,
                  activated=self._undo_last_mask_point)
        # Ctrl+B: toggle blackout (alternative to Esc for hardware controllers)
        QShortcut(QKeySequence("Ctrl+B"), self,
                  activated=lambda: self.blackout_button.toggle() if hasattr(self, 'blackout_button') else None)
        # Ctrl+L: toggle lock mode
        QShortcut(QKeySequence("Ctrl+L"), self,
                  activated=lambda: self.lock_mode_button.toggle() if hasattr(self, 'lock_mode_button') else None)

    def _undo_last_mask_point(self) -> None:
        """Remove the last point added during active mask creation."""
        if self.video_display.mask_creation_mode and self.video_display.mask_points:
            self.video_display.mask_points.pop()
            if self.mask_points_list.count() > 0:
                self.mask_points_list.takeItem(self.mask_points_list.count() - 1)
            self.video_display.update()

    def _show_shortcuts_help(self) -> None:
        """Show keyboard shortcut reference."""
        shortcuts = (
            "<b>Keyboard Shortcuts</b><br><br>"
            "<b>Ctrl+S</b> — Save settings<br>"
            "<b>Ctrl+C</b> — Trigger calibration<br>"
            "<b>Esc / Ctrl+B</b> — Toggle projector BLACKOUT<br>"
            "<b>Space</b> — Toggle preview panel<br>"
            "<b>F1</b> — Show this help<br>"
            "<b>F2</b> — Rename selected mask<br>"
            "<b>Delete</b> — Remove selected mask<br>"
            "<b>Ctrl+Right</b> — Advance cue on selected mask<br>"
            "<b>Ctrl+A</b> — Render all masks<br>"
            "<b>Ctrl+Z</b> — Undo last mask point (during mask creation)<br>"
            "<b>Ctrl+L</b> — Toggle UI lock mode<br>"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts)

    def closeEvent(self, event):
        self._is_closing = True
        self.save_settings()
        if self.midi_inport is not None:
            try:
                self.midi_inport.close()
            except Exception as exc:
                self.logger.warning("MIDI port close error: %s", exc)
        # Stop all timers first to prevent callbacks on destroyed objects
        for attr in ('midi_poll_timer', '_save_settings_timer', '_calib_timeout_timer',
                     '_midi_hotplug_timer', '_auto_save_timer', '_calibration_age_timer'):
            timer = getattr(self, attr, None)
            if timer:
                timer.stop()
                if attr == '_calib_timeout_timer':
                    setattr(self, attr, None)
        # Disconnect all worker signals before stopping thread
        _signals_to_disconnect = [
            self.worker.frame_ready,
            self.worker.projector_frame_ready,
            self.worker.trackers_detected,
            self.worker.camera_error,
            self.worker.performance_updated,
            self.worker.camera_info_updated,
            self.worker.markers_calibrated,
            self.worker.still_frame_ready,
            self.worker.worker_stopped,
            self.worker.calibration_progress,
            self.worker.performance_degraded,
            self.worker.tracking_state_changed,
            self.worker.calibration_restored,
            self.worker.diagnostic_info,
        ]
        for sig in _signals_to_disconnect:
            try:
                sig.disconnect()
            except RuntimeError:
                pass
        # Save calibration state before exit
        if self.worker._calibrated:
            try:
                self.worker._save_calibration()
            except Exception:
                pass
        self.worker.stop()
        self.thread.quit()
        # Wait up to 5 seconds for the worker thread to finish — don't hang forever
        if not self.thread.wait(5000):
            self.logger.warning("Worker thread did not stop within 5 s — forcing termination")
            self.thread.terminate()
            self.thread.wait(1000)
        event.accept()


if __name__ == '__main__':
    # Per-monitor DPI awareness must be set before QApplication is created.
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            pass

    configure_opencv_logging()

    # Graceful Ctrl+C (SIGINT) handler: stop the Qt event loop cleanly so
    # closeEvent fires and settings/calibration are saved before exit.
    import signal as _signal

    def _sigint_handler(signum, frame):
        logging.getLogger(__name__).info("SIGINT received — requesting clean shutdown")
        _app_instance = QApplication.instance()
        if _app_instance:
            _app_instance.quit()

    _signal.signal(_signal.SIGINT, _sigint_handler)

    app = QApplication(sys.argv)

    _asset_dir = Path(__file__).resolve().parent
    _icon_path = _asset_dir / 'icon.ico'
    if _icon_path.exists():
        from PyQt5.QtGui import QIcon
        app.setWindowIcon(QIcon(str(_icon_path)))

    splash = SplashScreen()
    splash.show()
    app.processEvents()

    _qss_path = _asset_dir / 'style.qss'
    try:
        app.setStyleSheet(_qss_path.read_text(encoding='utf-8'))
    except OSError:
        logging.getLogger(__name__).warning("Stylesheet not found: %s", _qss_path)

    try:
        main_win = ProjectionMappingApp()
        main_win.showFullScreen()
        splash.finish(main_win)
        sys.exit(app.exec_())
    except Exception:
        logging.getLogger(__name__).exception("Unhandled exception in main loop")
        QMessageBox.critical(
            None, "Fatal Error",
            "IR Guitar Tracker encountered an unrecoverable error.\n\n"
            "Check ir_tracker.log for details.\n\n"
            f"{sys.exc_info()[1]}"
        )
        sys.exit(1)
