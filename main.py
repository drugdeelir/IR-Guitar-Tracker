import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
from PyQt5.QtCore import QThread, Qt, QTimer
from PyQt5.QtGui import QPixmap
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
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mask import Mask
from splash import SplashScreen
from widgets import MarkerSelectionDialog, ProjectorWindow, VideoDisplay
from worker import Worker

SETTINGS_PATH = Path("settings.json")


def configure_opencv_logging():
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except AttributeError:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass


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

def _get_camera_backends():
    is_windows = platform.system().lower() == "windows"
    if not is_windows:
        return [cv2.CAP_ANY]

    preferred = ["CAP_DSHOW", "CAP_MSMF", "CAP_ANY"]
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


def get_available_cameras(max_probe=10):
    arr = []
    backends = _get_camera_backends()
    misses_after_first = 0

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
    return arr


class ProjectionMappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projection Mapping Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.masks = []
        self.selected_markers = []
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

        self.layout.addWidget(self.video_display)
        self.video_display.mask_point_added.connect(self.add_mask_point_to_list)
        self.projector_window.show()

        self.worker.frame_ready.connect(self.video_display.set_image)
        self.worker.projector_frame_ready.connect(self.projector_window.set_image)
        self.worker.projector_frame_ready.connect(self.update_projector_preview)
        self.projector_window.warp_points_changed.connect(self.worker.set_warp_points)
        self.worker.trackers_detected.connect(self.update_tracker_label)
        self.worker.camera_error.connect(self.show_camera_error)
        self.worker.performance_updated.connect(self.update_performance_label)

        self.marker_selection_dialog = MarkerSelectionDialog(self)
        self.marker_selection_dialog.take_picture_button.clicked.connect(
            self.start_marker_capture_countdown
        )
        self.worker.still_frame_ready.connect(self.set_marker_selection_image)

        self.apply_loaded_settings()
        self.maybe_show_startup_wizard()

        self.thread.started.connect(self.worker.process_video)
        self.thread.start()

    def load_settings(self):
        if SETTINGS_PATH.exists():
            try:
                return json.loads(SETTINGS_PATH.read_text())
            except Exception:
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
            "wizard_completed": True,
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        except Exception:
            pass

    def apply_loaded_settings(self):
        ir_threshold = self.settings.get("ir_threshold", 200)
        self.ir_threshold_slider.setValue(ir_threshold)
        threshold_mode = self.settings.get("threshold_mode", 0)
        self.threshold_mode_combo.setCurrentIndex(threshold_mode)
        self.depth_sensitivity_slider.setValue(self.settings.get("depth_sensitivity", 100))
        self.auto_sync_checkbox.setChecked(self.settings.get("auto_sync_enabled", True))
        self.preview_checkbox.setChecked(self.settings.get("show_preview_enabled", True))
        self.projector_preview_label.setVisible(self.preview_checkbox.isChecked())

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

    def open_marker_selection_dialog(self):
        self.marker_selection_dialog.clear_selection()
        self.marker_selection_dialog.take_picture_button.setText("Take Picture")
        self.marker_selection_dialog.take_picture_button.setEnabled(True)

        if self.marker_selection_dialog.exec_():
            self.selected_markers = self.marker_selection_dialog.get_selected_points()
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
        self.layout.addWidget(self.control_panel)

        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout()
        self.camera_combo = QComboBox()
        self.refresh_camera_button = QPushButton("Refresh Cameras")
        self.refresh_camera_button.clicked.connect(self.refresh_cameras)
        self.retry_camera_button = QPushButton("Retry Camera")
        self.retry_camera_button.clicked.connect(self.retry_camera)

        self.available_cameras = []
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        self.refresh_cameras(initial=True)

        camera_layout.addWidget(self.camera_combo)
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
        projector_layout.addWidget(self.projector_combo)
        projector_group.setLayout(projector_layout)
        self.control_layout.addWidget(projector_group)

        cue_group = QGroupBox("Cues")
        cue_layout = QVBoxLayout()
        self.cue_list_widget = QListWidget()
        self.cue_list_widget.currentRowChanged.connect(self.worker.set_active_cue_index)
        self.add_cue_button = QPushButton("Add Video Cue")
        self.add_cue_button.clicked.connect(self.add_cue)
        self.remove_cue_button = QPushButton("Remove Cue")
        self.remove_cue_button.clicked.connect(self.remove_cue)
        self.render_all_cues_button = QPushButton("Render All Cues")
        self.render_all_cues_button.clicked.connect(lambda: self.worker.set_active_cue_index(-1))
        cue_layout.addWidget(self.cue_list_widget)
        cue_layout.addWidget(self.add_cue_button)
        cue_layout.addWidget(self.remove_cue_button)
        cue_layout.addWidget(self.render_all_cues_button)
        cue_group.setLayout(cue_layout)
        self.control_layout.addWidget(cue_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_checkbox = QCheckBox("Show projector preview")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.toggled.connect(self.toggle_preview)
        self.projector_preview_label = QLabel("Waiting for projector frames...")
        self.projector_preview_label.setAlignment(Qt.AlignCenter)
        self.projector_preview_label.setMinimumHeight(140)
        preview_layout.addWidget(self.preview_checkbox)
        preview_layout.addWidget(self.projector_preview_label)
        preview_group.setLayout(preview_layout)
        self.control_layout.addWidget(preview_group)

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
        self.control_layout.addWidget(mask_group)

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
        diagnostics_layout.addWidget(self.performance_label)
        diagnostics_group.setLayout(diagnostics_layout)
        self.control_layout.addWidget(diagnostics_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_checkbox = QCheckBox("Show projector preview")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.toggled.connect(self.toggle_preview)
        self.projector_preview_label = QLabel("Waiting for projector frames...")
        self.projector_preview_label.setAlignment(Qt.AlignCenter)
        self.projector_preview_label.setMinimumHeight(140)
        preview_layout.addWidget(self.preview_checkbox)
        preview_layout.addWidget(self.projector_preview_label)
        preview_group.setLayout(preview_layout)
        self.control_layout.addWidget(preview_group)

        self.apply_preview_minimum_sizes()
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
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

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
        self.video_display.set_mask_creation_mode(False)
        mask_points = self.video_display.get_mask_points()

        current_item = self.cue_list_widget.currentItem()
        if current_item and mask_points:
            row = self.cue_list_widget.row(current_item)
            if 0 <= row < len(self.masks):
                self.masks[row].source_points = [(p.x(), p.y()) for p in mask_points]
                self.worker.set_masks(self.masks)

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
        current_item = self.cue_list_widget.currentItem()
        if not current_item:
            self.statusBar().showMessage("Please select a cue to link.", 3000)
            return

        if not self.selected_markers:
            self.statusBar().showMessage("Please select IR markers first.", 3000)
            return

        row = self.cue_list_widget.row(current_item)
        if 0 <= row < len(self.masks):
            mask = self.masks[row]
            if len(mask.source_points) != len(self.selected_markers):
                self.statusBar().showMessage(
                    f"Error: Mask has {len(mask.source_points)} points, but {len(self.selected_markers)} markers are selected.",
                    5000,
                )
            else:
                mask.linked_marker_count = len(self.selected_markers)
                self.statusBar().showMessage(
                    f"Mask '{mask.name}' linked to {len(self.selected_markers)} markers.", 3000
                )

    def auto_sync_marker_links(self):
        if not self.auto_sync_checkbox.isChecked() or not self.selected_markers:
            return

        marker_count = len(self.selected_markers)
        linked = 0
        for mask in self.masks:
            if len(mask.source_points) == marker_count:
                mask.linked_marker_count = marker_count
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

    def add_cue(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if video_path:
            mask_name = f"Cue {len(self.masks) + 1}: {video_path.split('/')[-1]}"
            new_mask = Mask(mask_name, [], video_path)
            self.masks.append(new_mask)
            self.cue_list_widget.addItem(mask_name)
            self.worker.set_masks(self.masks)

    def change_camera(self, index):
        if self.available_cameras and 0 <= index < len(self.available_cameras):
            self.worker.set_video_source(self.available_cameras[index])

    def change_projector(self, index):
        if index >= len(self.screens):
            return

        screen = self.screens[index]
        window_handle = self.projector_window.windowHandle()
        if window_handle is not None:
            window_handle.setScreen(screen)
        else:
            self.projector_window.setGeometry(screen.geometry())

        self.projector_window.showFullScreen()

    def remove_cue(self):
        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            self.cue_list_widget.takeItem(row)
            del self.masks[row]
            self.worker.set_masks(self.masks)

    def closeEvent(self, event):
        self.save_settings()
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
    main_win.show()
    splash.finish(main_win)
    sys.exit(app.exec_())
