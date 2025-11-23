
import sys
import time
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QGroupBox, QComboBox, QFileDialog, QLineEdit, QSlider, QListWidget, QStatusBar
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from widgets import VideoDisplay, ProjectorWindow
from worker import Worker
from mask import Mask
from splash import SplashScreen

def get_available_cameras():
    """Returns a list of available camera indices."""
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        arr.append(index)
        cap.release()
        index += 1
    return arr

class ProjectionMappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projection Mapping Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.masks = []

        self.setStatusBar(QStatusBar(self))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.video_display = VideoDisplay()
        self.projector_window = ProjectorWindow()

        self.create_control_panel()
        
        self.layout.addWidget(self.video_display)
        self.video_display.mask_point_added.connect(self.add_mask_point_to_list)
        self.projector_window.show()

        self.worker = Worker(parent=None)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.frame_ready.connect(self.video_display.set_image)
        self.worker.projector_frame_ready.connect(self.projector_window.set_image)
        self.projector_window.warp_points_changed.connect(self.worker.set_warp_points)
        self.worker.trackers_detected.connect(self.update_tracker_label)
        self.worker.camera_error.connect(self.show_camera_error)

        self.thread.started.connect(self.worker.process_video)
        self.thread.start()

    def create_control_panel(self):
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.layout.addWidget(self.control_panel)

        # Camera selection
        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout()
        self.camera_combo = QComboBox()
        self.available_cameras = get_available_cameras()
        self.camera_combo.addItems([f"Camera {i}" for i in self.available_cameras])
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        camera_layout.addWidget(self.camera_combo)
        camera_group.setLayout(camera_layout)
        self.control_layout.addWidget(camera_group)

        # Projector selection
        projector_group = QGroupBox("Projector Display")
        projector_layout = QVBoxLayout()
        self.projector_combo = QComboBox()
        self.screens = QApplication.screens()
        self.projector_combo.addItems([screen.name() or f"Screen {i+1}" for i, screen in enumerate(self.screens)])
        self.projector_combo.currentIndexChanged.connect(self.change_projector)
        projector_layout.addWidget(self.projector_combo)
        projector_group.setLayout(projector_layout)
        self.control_layout.addWidget(projector_group)

        # Cue system
        cue_group = QGroupBox("Cues")
        cue_layout = QVBoxLayout()
        self.cue_list_widget = QListWidget()
        self.add_cue_button = QPushButton("Add Video Cue")
        self.add_cue_button.clicked.connect(self.add_cue)
        self.remove_cue_button = QPushButton("Remove Cue")
        self.remove_cue_button.clicked.connect(self.remove_cue)
        cue_layout.addWidget(self.cue_list_widget)
        cue_layout.addWidget(self.add_cue_button)
        cue_layout.addWidget(self.remove_cue_button)
        cue_group.setLayout(cue_layout)
        self.control_layout.addWidget(cue_group)

        # Warping controls
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
        
        # IR Tracking controls
        ir_group = QGroupBox("IR Tracking")
        ir_layout = QVBoxLayout()
        self.ir_threshold_slider = QSlider(Qt.Horizontal)
        self.ir_threshold_slider.setRange(0, 255)
        self.ir_threshold_slider.setValue(200)
        self.ir_threshold_slider.valueChanged.connect(self.update_ir_threshold)
        self.ir_trackers_label = QLabel("Trackers detected: 0")
        ir_layout.addWidget(QLabel("IR Threshold:"))
        ir_layout.addWidget(self.ir_threshold_slider)
        ir_layout.addWidget(self.ir_trackers_label)
        ir_group.setLayout(ir_layout)
        self.control_layout.addWidget(ir_group)

        # Mask creation
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
        
        tracker_link_layout = QHBoxLayout()
        self.tracker_link_input = QLineEdit("0,1,2,3")
        self.link_trackers_button = QPushButton("Link Trackers")
        self.link_trackers_button.clicked.connect(self.link_trackers_to_mask)
        tracker_link_layout.addWidget(QLabel("Trackers:"))
        tracker_link_layout.addWidget(self.tracker_link_input)
        tracker_link_layout.addWidget(self.link_trackers_button)

        mask_layout.addWidget(self.create_mask_button)
        mask_layout.addWidget(self.finish_mask_button)
        mask_layout.addWidget(self.cancel_mask_button)
        mask_layout.addWidget(self.mask_points_list)
        mask_layout.addLayout(tracker_link_layout)
        mask_group.setLayout(mask_layout)
        self.control_layout.addWidget(mask_group)

        # Depth Estimation
        depth_group = QGroupBox("Depth Estimation")
        depth_layout = QVBoxLayout()
        self.calibrate_depth_button = QPushButton("Calibrate Depth")
        self.calibrate_depth_button.clicked.connect(self.calibrate_depth)
        self.depth_sensitivity_slider = QSlider(Qt.Horizontal)
        self.depth_sensitivity_slider.setRange(0, 200) # 0-200%
        self.depth_sensitivity_slider.setValue(100)
        self.depth_sensitivity_slider.valueChanged.connect(self.update_depth_sensitivity)
        self.depth_calibration_label = QLabel("Not calibrated")
        depth_layout.addWidget(self.calibrate_depth_button)
        depth_layout.addWidget(QLabel("Sensitivity:"))
        depth_layout.addWidget(self.depth_sensitivity_slider)
        depth_layout.addWidget(self.depth_calibration_label)
        depth_group.setLayout(depth_layout)
        self.control_layout.addWidget(depth_group)

        self.control_layout.addStretch()

    def calibrate_depth(self):
        self.worker.calibrate_depth()
        self.depth_calibration_label.setText("Calibrated!")

    def update_depth_sensitivity(self, value):
        self.worker.set_depth_sensitivity(value / 100.0)

    def show_camera_error(self, index):
        self.statusBar().showMessage(f"Error: Could not open Camera {index}", 5000) # 5 seconds

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
                self.masks[row].source_points = [ (p.x(), p.y()) for p in mask_points]
                self.worker.set_masks(self.masks)
                print(f"Mask created for {self.masks[row].name} with {len(mask_points)} points.")

        self.create_mask_button.setEnabled(True)
        self.finish_mask_button.setEnabled(False)
        self.cancel_mask_button.setEnabled(False)
        self.video_display.clear_mask_points()
        self.mask_points_list.clear()

    def cancel_mask_creation(self):
        self.video_display.set_mask_creation_mode(False)
        self.create_mask_button.setEnabled(True)
        self.finish_mask_button.setEnabled(False)
        self.cancel_mask_button.setEnabled(False)
        self.video_display.clear_mask_points()
        self.mask_points_list.clear()

    def add_mask_point_to_list(self, point):
        self.mask_points_list.addItem(f"({point.x()}, {point.y()})")

    def link_trackers_to_mask(self):
        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            if 0 <= row < len(self.masks):
                try:
                    tracker_ids_str = self.tracker_link_input.text().split(',')
                    tracker_ids = [int(i.strip()) for i in tracker_ids_str]
                    if len(tracker_ids) == 4:
                        self.masks[row].tracker_ids = tracker_ids
                        self.statusBar().showMessage(f"Trackers {tracker_ids} linked to {self.masks[row].name}", 3000)
                    else:
                        self.statusBar().showMessage("Error: Please enter 4 tracker indices.", 3000)
                except ValueError:
                    self.statusBar().showMessage("Error: Invalid tracker indices.", 3000)

    def update_ir_threshold(self, value):
        self.worker.set_ir_threshold(value)

    def update_tracker_label(self, count):
        self.ir_trackers_label.setText(f"Trackers detected: {count}")

    def toggle_warping(self, checked):
        self.projector_window.set_calibration_mode(checked)
        if checked:
            self.enable_warping_button.setText("Disable Warping")
        else:
            self.enable_warping_button.setText("Enable Warping")

    def add_cue(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if video_path:
            mask_name = f"Cue {len(self.masks) + 1}: {video_path.split('/')[-1]}"
            # Placeholder for mask points
            new_mask = Mask(mask_name, [], video_path)
            self.masks.append(new_mask)
            self.cue_list_widget.addItem(mask_name)
            self.worker.set_masks(self.masks)

    def change_camera(self, index):
        if self.available_cameras:
            new_camera_index = self.available_cameras[index]
            self.worker.set_video_source(new_camera_index)
            # You might need to restart the worker thread for the change to take effect
            # self.thread.quit()
            # self.thread.wait()
            # self.thread.start()

    def change_projector(self, index):
        if index < len(self.screens):
            screen = self.screens[index]
            self.projector_window.setScreen(screen)
            self.projector_window.showFullScreen()

    def remove_cue(self):
        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            self.cue_list_widget.takeItem(row)
            del self.masks[row]
            self.worker.set_masks(self.masks)

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    splash = SplashScreen()
    splash.show()
    
    # Process events to show splash screen
    app.processEvents()

    # Load and apply stylesheet
    try:
        with open('style.qss', 'r') as f:
            style = f.read()
        app.setStyleSheet(style)
    except FileNotFoundError:
        print("Stylesheet not found. Using default style.")

    main_win = ProjectionMappingApp()
    
    main_win.show()
    splash.finish(main_win)
    sys.exit(app.exec_())
