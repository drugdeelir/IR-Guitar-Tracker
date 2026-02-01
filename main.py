
import sys
import time
import cv2
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QGroupBox, QComboBox, QFileDialog,
                             QLineEdit, QSlider, QListWidget, QStatusBar, QCheckBox,
                             QDialog, QFormLayout)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPoint
from widgets import VideoDisplay, ProjectorWindow, MarkerSelectionDialog
from worker import Worker
from mask import Mask
from splash import SplashScreen
from midi_handler import MIDIHandler, get_midi_ports

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

class MIDIMappingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MIDI Mapping")
        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()
        self.layout.addLayout(self.form)

        self.mappings = parent.midi_mappings
        self.learn_buttons = {}

        # Define actions
        self.actions = [
            ('Toggle Amp Visibility', 'toggle_amp'),
            ('Toggle Background Visibility', 'toggle_bg'),
            ('Amp Strobe', 'fx_amp_strobe'),
            ('Amp Blur', 'fx_amp_blur'),
            ('Amp Invert', 'fx_amp_invert'),
            ('Amp Edges', 'fx_amp_edges'),
            ('Amp Tint', 'fx_amp_tint'),
            ('Amp Hue Cycle', 'fx_amp_hue_cycle'),
            ('Amp Feedback', 'fx_amp_feedback'),
            ('Amp RGB Shift', 'fx_amp_rgb_shift'),
            ('Amp Glitch', 'fx_amp_glitch'),
            ('Amp Trails', 'fx_amp_trails'),
            ('Amp Kaleidoscope', 'fx_amp_kaleidoscope'),
            ('Amp Mirror H', 'fx_amp_mirror_h'),
            ('Amp Mirror V', 'fx_amp_mirror_v'),
            ('Amp Design: Spiral', 'design_amp_spiral'),
            ('Amp Design: Moon', 'design_amp_moon'),
            ('Amp Design: Mushroom', 'design_amp_mushroom'),
            ('Amp Design: Star', 'design_amp_star'),
            ('Amp Design: Hexagon', 'design_amp_hexagon'),
            ('Amp Design: Heart', 'design_amp_heart'),
            ('Amp Design: None', 'design_amp_none'),
            ('Amp LFO Toggle', 'lfo_amp_toggle'),
            ('Amp LFO Speed', 'lfo_amp_speed'),
            ('Amp LFO Cycle Target', 'lfo_amp_cycle'),
            ('BG Strobe', 'fx_bg_strobe'),
            ('BG Blur', 'fx_bg_blur'),
            ('BG Invert', 'fx_bg_invert'),
            ('BG Edges', 'fx_bg_edges'),
            ('BG Tint', 'fx_bg_tint'),
            ('BG Hue Cycle', 'fx_bg_hue_cycle'),
            ('BG Feedback', 'fx_bg_feedback'),
            ('BG RGB Shift', 'fx_bg_rgb_shift'),
            ('BG Glitch', 'fx_bg_glitch'),
            ('BG Trails', 'fx_bg_trails'),
            ('BG Kaleidoscope', 'fx_bg_kaleidoscope'),
            ('BG Mirror H', 'fx_bg_mirror_h'),
            ('BG Mirror V', 'fx_bg_mirror_v'),
            ('BG Design: Spiral', 'design_bg_spiral'),
            ('BG Design: Moon', 'design_bg_moon'),
            ('BG Design: Mushroom', 'design_bg_mushroom'),
            ('BG Design: Star', 'design_bg_star'),
            ('BG Design: Hexagon', 'design_bg_hexagon'),
            ('BG Design: Heart', 'design_bg_heart'),
            ('BG Design: None', 'design_bg_none'),
            ('BG LFO Toggle', 'lfo_bg_toggle'),
            ('BG LFO Speed', 'lfo_bg_speed'),
            ('BG LFO Cycle Target', 'lfo_bg_cycle'),
        ]

        for i in range(8):
            self.actions.append((f'Switch Amp to Cue {i+1}', f'cue_amp_{i}'))
            self.actions.append((f'Switch BG to Cue {i+1}', f'cue_bg_{i}'))

        for label, key in self.actions:
            btn = QPushButton(self.get_mapping_text(key))
            btn.clicked.connect(lambda checked, k=key: self.start_learning(k))
            self.form.addRow(label, btn)
            self.learn_buttons[key] = btn

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.close_btn)

    def get_mapping_text(self, key):
        m = self.mappings.get(key)
        if not m: return "None (Click to Learn)"
        return f"{m[0].upper()} {m[1]}"

    def start_learning(self, key):
        self.parent().start_midi_learn(key)
        self.learn_buttons[key].setText("Listening...")

    def update_mappings(self):
        for label, key in self.actions:
            self.learn_buttons[key].setText(self.get_mapping_text(key))

class ProjectionMappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projection Mapping Tool")
        self.setGeometry(100, 100, 1200, 900)
        self.masks = []
        self.selected_markers = []

        # Default MIDI Mappings
        self.midi_mappings = {
            'cue_amp_0': ('note', 60), 'cue_amp_1': ('note', 61), 'cue_amp_2': ('note', 62), 'cue_amp_3': ('note', 63),
            'cue_bg_0': ('note', 72), 'cue_bg_1': ('note', 73), 'cue_bg_2': ('note', 74), 'cue_bg_3': ('note', 75),
            'toggle_amp': ('note', 48), 'toggle_bg': ('note', 50),
            'fx_amp_strobe': ('cc', 20), 'fx_amp_blur': ('cc', 21), 'fx_amp_invert': ('cc', 22), 'fx_amp_edges': ('cc', 23), 'fx_amp_tint': ('cc', 24),
            'fx_bg_strobe': ('cc', 30), 'fx_bg_blur': ('cc', 31), 'fx_bg_invert': ('cc', 32), 'fx_bg_edges': ('cc', 33), 'fx_bg_tint': ('cc', 34),
        }
        self.learning_key = None

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

        self.worker = Worker()
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.frame_ready.connect(self.video_display.set_image)
        self.worker.projector_frame_ready.connect(self.projector_window.set_image)
        self.projector_window.warp_points_changed.connect(self.worker.set_warp_points)
        self.worker.trackers_detected.connect(self.update_tracker_label)
        self.worker.camera_error.connect(self.show_camera_error)
        self.marker_selection_dialog = MarkerSelectionDialog(self)
        self.worker.still_frame_ready.connect(self.set_marker_selection_image)

        self.thread.started.connect(self.worker.process_video)
        self.thread.start()

    def start_midi_learn(self, key):
        self.learning_key = key
        if hasattr(self, 'midi_handler'):
            self.midi_handler.learning_mode = True

    def handle_learned_message(self, msg_type, number):
        if self.learning_key:
            self.midi_mappings[self.learning_key] = (msg_type, number)
            self.learning_key = None
            if hasattr(self, 'midi_handler'):
                self.midi_handler.learning_mode = False
            if hasattr(self, 'mapping_dialog') and self.mapping_dialog.isVisible():
                self.mapping_dialog.update_mappings()
            self.statusBar().showMessage(f"Mapped {msg_type} {number}", 3000)

    def open_marker_selection_dialog(self):
        self.marker_selection_dialog.clear_selection()
        try:
            self.marker_selection_dialog.take_picture_button.clicked.disconnect()
        except TypeError:
            pass
        self.marker_selection_dialog.take_picture_button.clicked.connect(self.start_marker_capture_countdown)

        if self.marker_selection_dialog.exec_():
            self.selected_markers = self.marker_selection_dialog.get_selected_points()
            print(f"Selected {len(self.selected_markers)} markers.")
            self.worker.set_marker_points(self.selected_markers)

    def start_marker_capture_countdown(self):
        self.marker_selection_dialog.take_picture_button.setEnabled(False)
        self.countdown_timer = QTimer(self)
        self.countdown_seconds = 7
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

    def update_countdown(self):
        if self.countdown_seconds > 0:
            self.marker_selection_dialog.take_picture_button.setText(f"{self.countdown_seconds}...")
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

    def create_control_panel(self):
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.layout.addWidget(self.control_panel)

        # Project management
        project_group = QGroupBox("Project")
        project_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Project")
        self.save_button.clicked.connect(self.save_project)
        self.load_button = QPushButton("Load Project")
        self.load_button.clicked.connect(self.load_project)
        project_layout.addWidget(self.save_button)
        project_layout.addWidget(self.load_button)
        project_group.setLayout(project_layout)
        self.control_layout.addWidget(project_group)

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

        # MIDI Settings
        midi_group = QGroupBox("MIDI Settings")
        midi_layout = QVBoxLayout()
        self.midi_combo = QComboBox()
        self.midi_ports = get_midi_ports()
        self.midi_combo.addItems(["None"] + self.midi_ports)
        self.midi_combo.currentIndexChanged.connect(self.change_midi_port)
        midi_layout.addWidget(QLabel("MIDI Input Port:"))
        midi_layout.addWidget(self.midi_combo)

        self.midi_map_btn = QPushButton("Configure MIDI Mappings")
        self.midi_map_btn.clicked.connect(self.open_midi_mapping)
        midi_layout.addWidget(self.midi_map_btn)

        self.bpm_label = QLabel("BPM: 120.0")
        midi_layout.addWidget(self.bpm_label)
        midi_group.setLayout(midi_layout)
        self.control_layout.addWidget(midi_group)

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

        self.auto_ir_check = QCheckBox("Auto Threshold")
        self.auto_ir_check.toggled.connect(self.toggle_auto_ir)
        ir_layout.addWidget(self.auto_ir_check)

        self.ir_threshold_slider = QSlider(Qt.Horizontal)
        self.ir_threshold_slider.setRange(0, 255)
        self.ir_threshold_slider.setValue(200)
        self.ir_threshold_slider.valueChanged.connect(self.update_ir_threshold)
        self.ir_trackers_label = QLabel("Trackers detected: 0")
        ir_layout.addWidget(QLabel("IR Threshold:"))
        ir_layout.addWidget(self.ir_threshold_slider)
        ir_layout.addWidget(self.ir_trackers_label)

        self.select_markers_button = QPushButton("Select Guitar Markers")
        self.select_markers_button.clicked.connect(self.open_marker_selection_dialog)

        self.clear_markers_button = QPushButton("Clear Marker Selection")
        self.clear_markers_button.clicked.connect(self.clear_marker_selection)

        ir_layout.addWidget(self.select_markers_button)
        ir_layout.addWidget(self.clear_markers_button)

        ir_group.setLayout(ir_layout)
        self.control_layout.addWidget(ir_group)

        # Mask creation
        mask_group = QGroupBox("Mask Creation")
        mask_layout = QVBoxLayout()
        self.create_mask_button = QPushButton("Create Mask")
        self.create_mask_button.clicked.connect(self.enter_mask_creation_mode)

        self.mask_tag_combo = QComboBox()
        self.mask_tag_combo.addItems(["none", "amp", "background"])
        mask_layout.addWidget(QLabel("Mask Tag:"))
        mask_layout.addWidget(self.mask_tag_combo)

        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItems(["dynamic", "static"])
        mask_layout.addWidget(QLabel("Mask Type:"))
        mask_layout.addWidget(self.mask_type_combo)

        self.mask_design_combo = QComboBox()
        self.mask_design_combo.addItems(["none", "spiral", "moon", "mushroom", "star", "hexagon", "heart"])
        mask_layout.addWidget(QLabel("Design Overlay:"))
        mask_layout.addWidget(self.mask_design_combo)

        self.lfo_target_combo = QComboBox()
        self.lfo_target_combo.addItems(["none", "blur", "tint", "rgb_shift", "hue"])
        mask_layout.addWidget(QLabel("LFO Target:"))
        mask_layout.addWidget(self.lfo_target_combo)

        self.mask_blend_combo = QComboBox()
        self.mask_blend_combo.addItems(["normal", "add", "screen", "multiply"])
        mask_layout.addWidget(QLabel("Blend Mode:"))
        mask_layout.addWidget(self.mask_blend_combo)

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
        mask_layout.addWidget(self.link_mask_button)

        mask_group.setLayout(mask_layout)
        self.control_layout.addWidget(mask_group)

        # Depth Estimation
        depth_group = QGroupBox("Depth Estimation")
        depth_layout = QVBoxLayout()
        self.calibrate_depth_button = QPushButton("Calibrate Depth")
        self.calibrate_depth_button.clicked.connect(self.calibrate_depth)

        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(50)
        self.smoothing_slider.valueChanged.connect(self.update_smoothing)
        depth_layout.addWidget(QLabel("Smoothing:"))
        depth_layout.addWidget(self.smoothing_slider)

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

        self.control_layout.addStretch()

    def open_midi_mapping(self):
        self.mapping_dialog = MIDIMappingDialog(self)
        self.mapping_dialog.show()

    def toggle_auto_ir(self, checked):
        self.worker.set_auto_threshold(checked)
        self.ir_threshold_slider.setEnabled(not checked)

    def update_smoothing(self, value):
        self.worker.set_smoothing(value / 100.0)

    def save_project(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Project Files (*.json)")
        if filename:
            project_data = {
                'masks': [mask.to_dict() for mask in self.masks],
                'warp_points': self.projector_window.warp_points,
                'ir_threshold': self.ir_threshold_slider.value(),
                'auto_ir': self.auto_ir_check.isChecked(),
                'depth_sensitivity': self.depth_sensitivity_slider.value(),
                'smoothing': self.smoothing_slider.value(),
                'midi_port': self.midi_combo.currentText(),
                'midi_mappings': self.midi_mappings,
                'marker_config': self.worker.marker_config,
                'baseline_distance': self.worker.baseline_distance
            }
            with open(filename, 'w') as f:
                json.dump(project_data, f, indent=4)
            self.statusBar().showMessage(f"Project saved to {filename}", 3000)

    def load_project(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Project Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                self.masks = [Mask.from_dict(d) for d in data.get('masks', [])]
                self.cue_list_widget.clear()
                self.cue_list_widget.addItems([mask.name for mask in self.masks])
                self.worker.set_masks(self.masks)

                warp_points = data.get('warp_points')
                if warp_points:
                    self.projector_window.warp_points = warp_points
                    self.worker.set_warp_points(warp_points)

                self.auto_ir_check.setChecked(data.get('auto_ir', False))
                self.ir_threshold_slider.setValue(data.get('ir_threshold', 200))
                self.depth_sensitivity_slider.setValue(data.get('depth_sensitivity', 100))
                self.smoothing_slider.setValue(data.get('smoothing', 50))

                midi_port = data.get('midi_port')
                if midi_port and midi_port != "None":
                    index = self.midi_combo.findText(midi_port)
                    if index >= 0:
                        self.midi_combo.setCurrentIndex(index)

                self.midi_mappings = data.get('midi_mappings', self.midi_mappings)

                marker_config = data.get('marker_config')
                if marker_config:
                    config_pts = [tuple(p) for p in marker_config]
                    self.worker.set_marker_points([QPoint(p[0], p[1]) for p in config_pts])

                self.worker.baseline_distance = data.get('baseline_distance', 0)

                self.statusBar().showMessage(f"Project loaded from {filename}", 3000)
            except Exception as e:
                self.statusBar().showMessage(f"Error loading project: {e}", 5000)

    def change_midi_port(self, index):
        if index == 0:
            if hasattr(self, 'midi_handler'):
                self.midi_handler.stop()
        else:
            port_name = self.midi_ports[index - 1]
            if hasattr(self, 'midi_handler'):
                self.midi_handler.stop()
                if hasattr(self, 'midi_thread'):
                    self.midi_thread.quit()
                    self.midi_thread.wait()

            self.midi_handler = MIDIHandler(port_name)
            self.midi_thread = QThread()
            self.midi_handler.moveToThread(self.midi_thread)
            self.midi_handler.note_on.connect(self.handle_midi_note)
            self.midi_handler.control_change.connect(self.handle_midi_cc)
            self.midi_handler.beat.connect(self.handle_bpm)
            self.midi_handler.learned_message.connect(self.handle_learned_message)
            self.midi_thread.started.connect(self.midi_handler.start_listening)
            self.midi_thread.start()

    def handle_midi_note(self, note, velocity):
        # Use Dynamic Mappings
        for key, mapping in self.midi_mappings.items():
            if mapping[0] == 'note' and mapping[1] == note:
                self.execute_midi_action(key, velocity)

    def handle_midi_cc(self, cc, value):
        for key, mapping in self.midi_mappings.items():
            if mapping[0] == 'cc' and mapping[1] == cc:
                self.execute_midi_action(key, value)

    def execute_midi_action(self, key, value):
        if key.startswith('cue_amp_'):
            idx = int(key.split('_')[-1])
            if idx < len(self.masks):
                self.worker.switch_video('amp', self.masks[idx].video_path)
        elif key.startswith('cue_bg_'):
            idx = int(key.split('_')[-1])
            if idx < len(self.masks):
                self.worker.switch_video('background', self.masks[idx].video_path)
        elif key == 'toggle_amp':
            self.worker.toggle_mask('amp', value > 0)
        elif key == 'toggle_bg':
            self.worker.toggle_mask('background', value > 0)
        elif key.startswith('fx_'):
            parts = key.split('_')
            tag = parts[1]
            fx_name = "_".join(parts[2:])
            self.worker.set_fx(tag, fx_name, value > 64)
        elif key.startswith('design_'):
            parts = key.split('_')
            tag = parts[1]
            design_name = parts[2]
            if value > 64:
                for mask in self.masks:
                    if mask.tag == tag:
                        mask.design_overlay = design_name
        elif key.startswith('lfo_'):
            parts = key.split('_')
            tag = parts[1]
            param = parts[2]
            for mask in self.masks:
                if mask.tag == tag:
                    if param == 'toggle':
                        mask.fx_params['lfo_enabled'] = (value > 64)
                    elif param == 'speed':
                        mask.fx_params['lfo_speed'] = (value / 64.0)
                    elif param == 'cycle' and value > 64:
                        targets = ["none", "blur", "tint", "rgb_shift"]
                        curr = mask.fx_params.get('lfo_target', 'none')
                        idx = (targets.index(curr) + 1) % len(targets)
                        mask.fx_params['lfo_target'] = targets[idx]

    def handle_bpm(self, bpm):
        self.bpm_label.setText(f"BPM: {bpm:.1f}")
        self.worker.set_bpm(bpm)

    def calibrate_depth(self):
        self.worker.calibrate_depth()
        self.depth_calibration_label.setText("Calibrated!")

    def update_depth_sensitivity(self, value):
        self.worker.set_depth_sensitivity(value / 100.0)

    def show_camera_error(self, index):
        self.statusBar().showMessage(f"Error: Could not open Camera {index}", 5000)

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
                self.masks[row].tag = self.mask_tag_combo.currentText()
                self.masks[row].type = self.mask_type_combo.currentText()
                self.masks[row].design_overlay = self.mask_design_combo.currentText()
                self.masks[row].fx_params['lfo_target'] = self.lfo_target_combo.currentText()
                self.masks[row].blend_mode = self.mask_blend_combo.currentText()
                self.worker.set_masks(self.masks)
                print(f"Mask created for {self.masks[row].name} with tag {self.masks[row].tag}")

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
                self.statusBar().showMessage(f"Error: Mask has {len(mask.source_points)} points, but {len(self.selected_markers)} markers are selected.", 5000)
            else:
                mask.linked_marker_count = len(self.selected_markers)
                self.statusBar().showMessage(f"Mask '{mask.name}' linked to {len(self.selected_markers)} markers.", 3000)

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
        if not video_path:
            # Allow creating a generative cue
            video_path = "generative"

        if video_path:
            mask_name = f"Cue {len(self.masks) + 1}: {video_path.split('/')[-1]}"
            new_mask = Mask(mask_name, [], video_path)
            self.masks.append(new_mask)
            self.cue_list_widget.addItem(mask_name)
            self.worker.set_masks(self.masks)

    def change_camera(self, index):
        if self.available_cameras:
            new_camera_index = self.available_cameras[index]
            self.worker.set_video_source(new_camera_index)

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
        if hasattr(self, 'midi_handler'):
            self.midi_handler.stop()
            if hasattr(self, 'midi_thread'):
                self.midi_thread.quit()
                self.midi_thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    splash = SplashScreen()
    splash.show()
    app.processEvents()

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
