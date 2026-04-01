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
from PyQt5.QtCore import QEventLoop, QPoint, QThread, Qt, QTimer
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
    QPushButton,
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


def configure_opencv_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except AttributeError:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass  # OpenCV build does not expose logging API; silently skip


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


class ProjectionMappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projection Mapping Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.masks = []
        self.selected_markers = []
        self.reference_markers = []
        self.latest_camera_qimage = None
        self.settings = self.load_settings()
        self.logger = logging.getLogger("ProjectionMappingApp")

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

        self.worker.frame_ready.connect(self.video_display.set_image)
        self.worker.frame_ready.connect(self.cache_latest_frame)
        self.worker.projector_frame_ready.connect(self.projector_window.set_image)
        self.worker.projector_frame_ready.connect(self.update_projector_preview)
        self.projector_window.warp_points_changed.connect(self.worker.set_warp_points)
        self.worker.trackers_detected.connect(self.update_tracker_label)
        self.worker.camera_error.connect(self.show_camera_error)
        self.worker.performance_updated.connect(self.update_performance_label)
        self.worker.camera_info_updated.connect(self.update_camera_info)
        self.worker.markers_calibrated.connect(self._on_markers_calibrated)

        self.marker_selection_dialog = MarkerSelectionDialog(self)
        self.marker_selection_dialog.take_picture_button.clicked.connect(
            self.start_marker_capture_countdown
        )
        self.worker.still_frame_ready.connect(self.set_marker_selection_image)

        self.apply_loaded_settings()
        self.refresh_mask_views()
        self.change_projector(self.projector_combo.currentIndex())
        self.maybe_show_startup_wizard()
        self._auto_create_test_masks()

        self.thread.started.connect(self.worker.process_video)
        self.thread.start()

    def load_settings(self):
        if SETTINGS_PATH.exists():
            try:
                return json.loads(SETTINGS_PATH.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logging.getLogger("MainWindow").warning("Could not load settings: %s", exc)
                return {}
        return {}

    def save_settings(self):
        settings = {
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
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        except OSError as exc:
            self.logger.error("Could not save settings: %s", exc)

    def apply_loaded_settings(self):
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

    def _run_marker_selection_dialog(self, *, use_live_capture=True, reference_pixmap=None, title="Select IR Markers", ir_assist=True):
        self.marker_selection_dialog.setWindowTitle(title)
        self.marker_selection_dialog.clear_selection()
        self.marker_selection_dialog.set_ir_assist_enabled(ir_assist)
        self.marker_selection_dialog.take_picture_button.setVisible(use_live_capture)
        self.marker_selection_dialog.take_picture_button.setEnabled(use_live_capture)
        self.marker_selection_dialog.take_picture_button.setText("Take Picture")

        if reference_pixmap is not None:
            self.marker_selection_dialog.set_pixmap(reference_pixmap)

        if self.marker_selection_dialog.exec_():
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
                self.camera_combo.setCurrentIndex(dialog.camera_combo.currentIndex())
                self.change_camera(dialog.camera_combo.currentIndex())
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
        self.camera_mode_combo = QComboBox()
        self.camera_mode_combo.addItem("Native / Driver Default", "native")
        self.camera_mode_combo.addItem("Performance (960x540 @ 30)", "performance")
        self.camera_mode_combo.addItem("High Detail (1280x720 @ 30)", "hd")
        self.camera_mode_combo.currentIndexChanged.connect(self.update_camera_mode)
        self.refresh_camera_button = QPushButton("Refresh Cameras")
        self.refresh_camera_button.clicked.connect(self.refresh_cameras)
        self.retry_camera_button = QPushButton("Retry Camera")
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
        self.screens = QApplication.screens()
        self.projector_combo.addItems(
            [screen.name() or f"Screen {i + 1}" for i, screen in enumerate(self.screens)]
        )

        self.projector_combo.currentIndexChanged.connect(self.change_projector)
        self.logger.info("Detected displays: %s", [f"{i}:{s.name()} {s.geometry()}" for i, s in enumerate(self.screens)])
        projector_layout.addWidget(self.projector_combo)
        projector_group.setLayout(projector_layout)
        self.control_layout.addWidget(projector_group)

        self.setup_wizard_button = QPushButton("Run Full Calibration Wizard")
        self.setup_wizard_button.clicked.connect(self.run_full_calibration_wizard_safe)
        self.control_layout.addWidget(self.setup_wizard_button)

        self.calibrate_button = QPushButton("Calibrate Guitar + Homography")
        self.calibrate_button.setStyleSheet("QPushButton { background-color: #2a5; color: white; font-weight: bold; padding: 8px; }")
        self.calibrate_button.clicked.connect(self._start_calibration)
        self.control_layout.addWidget(self.calibrate_button)

        self.mapping_tabs = QTabWidget()

        # Page 1: Masks
        masks_page = QWidget()
        masks_layout = QVBoxLayout(masks_page)

        mask_list_group = QGroupBox("Masks")
        mask_list_layout = QVBoxLayout()
        self.mask_list_widget = QListWidget()
        self.mask_list_widget.currentRowChanged.connect(self.on_mask_selection_changed)
        self.remove_mask_button = QPushButton("Remove Selected Mask")
        self.remove_mask_button.clicked.connect(self.remove_mask)
        mask_list_layout.addWidget(self.mask_list_widget)
        mask_list_layout.addWidget(self.remove_mask_button)
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

        cue_layout.addWidget(QLabel("Mask"))
        cue_layout.addWidget(self.cue_mask_combo)
        cue_layout.addWidget(self.mask_cue_list_widget)
        cue_layout.addWidget(self.add_cue_button)
        cue_layout.addWidget(self.remove_cue_button)
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
        midi_layout.addWidget(self.midi_input_combo)
        midi_layout.addWidget(self.refresh_midi_button)
        midi_layout.addWidget(self.connect_midi_button)
        midi_group.setLayout(midi_layout)
        cues_layout.addWidget(midi_group)

        self.mapping_tabs.addTab(masks_page, "Masks")
        self.mapping_tabs.addTab(cues_page, "Cues")
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

        ir_layout.addWidget(QLabel("IR Threshold:"))
        ir_layout.addWidget(self.ir_threshold_slider)
        ir_layout.addWidget(QLabel("Threshold Mode:"))
        ir_layout.addWidget(self.threshold_mode_combo)
        ir_layout.addWidget(self.ir_trackers_label)

        self.select_markers_button = QPushButton("Select Guitar Markers")
        self.select_markers_button.clicked.connect(self.open_marker_selection_dialog)
        self.clear_markers_button = QPushButton("Clear Marker Selection")
        self.clear_markers_button.clicked.connect(self.clear_marker_selection)
        ir_layout.addWidget(self.select_markers_button)
        ir_layout.addWidget(self.clear_markers_button)
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
        depth_layout.addWidget(QLabel("Sensitivity:"))
        depth_layout.addWidget(self.depth_sensitivity_slider)
        depth_layout.addWidget(self.depth_calibration_label)
        depth_group.setLayout(depth_layout)
        self.control_layout.addWidget(depth_group)

        diagnostics_group = QGroupBox("Diagnostics")
        diagnostics_layout = QVBoxLayout()
        self.performance_label = QLabel("FPS: -- | Frame: -- | D: -- M: -- W: -- R: --")
        self.camera_info_label = QLabel("Camera: waiting for stream...")
        diagnostics_layout.addWidget(self.performance_label)
        diagnostics_layout.addWidget(self.camera_info_label)
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

    def cache_latest_frame(self, image):
        self.latest_camera_qimage = image.copy()

    def capture_still_frame_sync(self, timeout_ms=5000, label="capture"):
        loop = QEventLoop(self)
        result = {"image": None}

        def on_frame(image):
            result["image"] = image.copy()
            loop.quit()

        self.logger.debug("Requesting still frame: %s", label)
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
        off_frame = self._qimage_to_bgr(off_image)
        on_frame = self._qimage_to_bgr(on_image)
        if off_frame is None or on_frame is None:
            return None

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
                return None

        best_quad = None
        best_score = -1.0
        image_area = float(w * h)

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
            rectangularity = quad_area / rect_area
            score = (quad_area / image_area) * 0.8 + rectangularity * 0.2

            if score > best_score:
                best_score = score
                best_quad = quad

        if best_quad is None:
            return None

        ordered = self._order_quad_points(best_quad)
        normalized = []
        for x, y in ordered:
            normalized.append([
                float(max(0.0, min(1.0, x / max(w, 1)))),
                float(max(0.0, min(1.0, y / max(h, 1)))),
            ])
        return normalized


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
        bounds = self.detect_projector_bounds(still_off, still_on)
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

    def update_camera_info(self, message):
        self.camera_info_label.setText(message)

    def calibrate_depth(self):
        self.worker.calibrate_depth()
        self.depth_calibration_label.setText("Calibrating...")

    def update_depth_sensitivity(self, value):
        self.worker.set_depth_sensitivity(value / 100.0)

    def show_camera_error(self, index):
        self.statusBar().showMessage(f"Error: Could not open Camera {index}", 5000)

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
            self.mask_list_widget.addItem(mask.name)
            self.cue_mask_combo.addItem(mask.name)
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
            self.mask_cue_list_widget.addItem(f"{active}{Path(cue).name}{cc_suffix}")
        if mask.cues:
            self.mask_cue_list_widget.setCurrentRow(mask.active_cue)

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
        self.masks[idx].remove_cue(cue_idx)
        self.worker.set_masks(self.masks)
        self.refresh_cues_for_selected_mask()

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
        try:
            self.midi_inport = mido.open_input(name)
            self.statusBar().showMessage(f"Connected MIDI input: {name}", 3000)
        except Exception as exc:
            self.logger.exception("Failed to open MIDI input")
            self.statusBar().showMessage(f"MIDI connect failed: {exc}", 5000)

    def poll_midi_messages(self):
        if self.midi_inport is None:
            return
        try:
            for msg in self.midi_inport.iter_pending():
                if msg.type == "control_change":
                    self.route_midi_cc(int(msg.control), int(msg.value))
        except Exception:
            self.logger.exception("MIDI polling error")

    def route_midi_cc(self, cc, value):
        if value <= 0:
            return
        for mask in self.masks:
            if cc in mask.midi_cc_map:
                cue_idx = mask.midi_cc_map[cc]
                if 0 <= cue_idx < len(mask.cues):
                    mask.active_cue = cue_idx
        self.worker.set_masks(self.masks)
        self.refresh_cues_for_selected_mask()

    def remove_mask(self):
        row = self.mask_list_widget.currentRow()
        if 0 <= row < len(self.masks):
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

        self.logger.info(
            "Projector display changed to index=%d name=%s geometry=%s",
            index,
            screen.name(),
            target_geometry,
        )

    def closeEvent(self, event):
        self.save_settings()
        if self.midi_inport is not None:
            try:
                self.midi_inport.close()
            except Exception as exc:
                self.logger.warning("MIDI port close error: %s", exc)
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()


if __name__ == '__main__':
    configure_opencv_logging()
    app = QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()
    app.processEvents()

    try:
        with open('style.qss', 'r') as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print("Stylesheet not found. Using default style.")

    main_win = ProjectionMappingApp()
    main_win.showFullScreen()
    splash.finish(main_win)
    sys.exit(app.exec_())
