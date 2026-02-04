
import sys
import time
import cv2
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QGroupBox, QComboBox, QFileDialog,
                             QLineEdit, QSlider, QListWidget, QStatusBar, QCheckBox,
                             QDialog, QFormLayout, QInputDialog, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt5.QtGui import QPixmap, QDesktopServices
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPoint, QPointF, QUrl
from widgets import VideoDisplay, ProjectorWindow, MarkerSelectionDialog, AudioMonitor
from worker import Worker
from mask import Mask
from splash import SplashScreen
from midi_handler import MIDIHandler, get_midi_ports
from osc_handler import OSCHandler
from utils import resource_path

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
        self.actions = []
        for tag_label, tag in [('Amp', 'amp'), ('BG', 'background'), ('Master', 'master')]:
            self.actions += [
                (f'Toggle {tag_label} Visibility', f'toggle_{tag}'),
                (f'{tag_label} Strobe', f'fx_{tag}_strobe'),
                (f'{tag_label} Blur', f'fx_{tag}_blur'),
                (f'{tag_label} Invert', f'fx_{tag}_invert'),
                (f'{tag_label} Edges', f'fx_{tag}_edges'),
                (f'{tag_label} Tint', f'fx_{tag}_tint'),
                (f'{tag_label} Duotone', f'fx_{tag}_duotone'),
                (f'{tag_label} Chromakey', f'fx_{tag}_chromakey'),
                (f'{tag_label} Hue Cycle', f'fx_{tag}_hue_cycle'),
                (f'{tag_label} Feedback', f'fx_{tag}_feedback'),
                (f'{tag_label} RGB Shift', f'fx_{tag}_rgb_shift'),
                (f'{tag_label} Glitch', f'fx_{tag}_glitch'),
                (f'{tag_label} Trails', f'fx_{tag}_trails'),
                (f'{tag_label} Kaleidoscope', f'fx_{tag}_kaleidoscope'),
                (f'{tag_label} Mirror H', f'fx_{tag}_mirror_h'),
                (f'{tag_label} Mirror V', f'fx_{tag}_mirror_v'),
                (f'{tag_label} Pixelate', f'fx_{tag}_pixelate'),
                (f'{tag_label} Chroma Ab', f'fx_{tag}_chroma_aberration'),
                (f'{tag_label} Ooze', f'fx_{tag}_ooze'),
                (f'{tag_label} Matrix', f'fx_{tag}_matrix'),
                (f'{tag_label} VHS', f'fx_{tag}_vhs'),
                (f'{tag_label} Scanline', f'fx_{tag}_scanline'),
                (f'{tag_label} Design: Spiral', f'design_{tag}_spiral'),
                (f'{tag_label} Design: Moon', f'design_{tag}_moon'),
                (f'{tag_label} Design: Mushroom', f'design_{tag}_mushroom'),
                (f'{tag_label} Design: Star', f'design_{tag}_star'),
                (f'{tag_label} Design: Hexagon', f'design_{tag}_hexagon'),
                (f'{tag_label} Design: Heart', f'design_{tag}_heart'),
                (f'{tag_label} Design: None', f'design_{tag}_none'),
                (f'{tag_label} LFO Toggle', f'lfo_{tag}_toggle'),
                (f'{tag_label} LFO Speed', f'lfo_{tag}_speed'),
                (f'{tag_label} LFO Shape Cycle', f'lfo_{tag}_shape'),
                (f'{tag_label} LFO Cycle Target', f'lfo_{tag}_cycle'),
                (f'{tag_label} Bezier Toggle', f'bezier_{tag}_toggle'),
            ]

        self.actions += [
            ('Auto-Pilot Toggle', 'auto_pilot_toggle'),
            ('Toggle HUD', 'hud_toggle'),
            ('Toggle Safety Mode', 'safety_toggle'),
            ('Style: Acid', 'style_acid'),
            ('Style: Noir', 'style_noir'),
            ('Style: Retro', 'style_retro'),
            ('Style: Clear', 'style_none'),
            ('Particles: Dust', 'part_dust'),
            ('Particles: Rain', 'part_rain'),
            ('Particles: Off', 'part_none'),
            ('Blackout (Panic)', 'blackout_toggle'),
        ]

        for i in range(8):
            self.actions.append((f'Save Snapshot {i+1}', f'snap_save_{i}'))
            self.actions.append((f'Load Snapshot {i+1}', f'snap_load_{i}'))

        for i in range(8):
            self.actions.append((f'Switch Amp to Cue {i+1}', f'cue_amp_{i}'))
            self.actions.append((f'Switch BG to Cue {i+1}', f'cue_background_{i}'))

        self.actions.append(('Toggle Projector Splash', 'toggle_projector_splash'))

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

class PlaylistEditorDialog(QDialog):
    def __init__(self, mask, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Playlist: {mask.name}")
        self.setMinimumSize(500, 400)
        self.mask = mask
        self.layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        for path in mask.playlist:
            self.list_widget.addItem(path.split('/')[-1])
        self.layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.up_btn = QPushButton("Move Up")
        self.up_btn.clicked.connect(self.move_up)
        self.down_btn = QPushButton("Move Down")
        self.down_btn.clicked.connect(self.move_down)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_item)
        btn_layout.addWidget(self.up_btn)
        btn_layout.addWidget(self.down_btn)
        btn_layout.addWidget(self.remove_btn)
        self.layout.addLayout(btn_layout)

        self.confirm_btn = QPushButton("Save Playlist Order")
        self.confirm_btn.clicked.connect(self.accept)
        self.confirm_btn.setStyleSheet("background-color: #4a148c; color: white; height: 40px;")
        self.layout.addWidget(self.confirm_btn)

    def move_up(self):
        curr = self.list_widget.currentRow()
        if curr > 0:
            item = self.list_widget.takeItem(curr)
            self.list_widget.insertItem(curr - 1, item)
            self.list_widget.setCurrentRow(curr - 1)
            # Update actual playlist
            self.mask.playlist.insert(curr - 1, self.mask.playlist.pop(curr))

    def move_down(self):
        curr = self.list_widget.currentRow()
        if curr >= 0 and curr < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(curr)
            self.list_widget.insertItem(curr + 1, item)
            self.list_widget.setCurrentRow(curr + 1)
            # Update actual playlist
            self.mask.playlist.insert(curr + 1, self.mask.playlist.pop(curr))

    def remove_item(self):
        curr = self.list_widget.currentRow()
        if curr >= 0:
            self.list_widget.takeItem(curr)
            self.mask.playlist.pop(curr)

class ProjectionMappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projection Mapping Tool")
        self.setGeometry(100, 100, 1400, 950)
        self.masks = []
        self.media_library = []
        self.selected_markers = []
        self.snapshots = [None] * 8
        self.moods = {
            "Cyberpunk": ((255, 0, 255), (0, 255, 255)), # Magenta, Cyan
            "Ocean": ((255, 100, 0), (200, 50, 0)),     # Blue, Deep Blue
            "Inferno": ((0, 0, 255), (0, 100, 255)),    # Red, Orange
            "Toxic": ((0, 255, 0), (0, 255, 150)),      # Green, Lime
            "Noir": ((50, 50, 50), (150, 150, 150)),    # Gray, White
            "None": ((255, 255, 255), (255, 255, 255))
        }
        self.setup_step = 0 # 0: Camera, 1: BG, 2: Guitar, 3: Done
        self.current_project_path = None

        # Default MIDI Mappings
        self.midi_mappings = {
            'cue_amp_0': ('note', 60), 'cue_amp_1': ('note', 61), 'cue_amp_2': ('note', 62), 'cue_amp_3': ('note', 63),
            'cue_background_0': ('note', 72), 'cue_background_1': ('note', 73), 'cue_background_2': ('note', 74), 'cue_background_3': ('note', 75),
            'toggle_amp': ('note', 48), 'toggle_background': ('note', 50),
            'fx_amp_strobe': ('cc', 20), 'fx_amp_blur': ('cc', 21), 'fx_amp_invert': ('cc', 22), 'fx_amp_edges': ('cc', 23), 'fx_amp_tint': ('cc', 24),
            'fx_background_strobe': ('cc', 30), 'fx_background_blur': ('cc', 31), 'fx_background_invert': ('cc', 32), 'fx_background_edges': ('cc', 33), 'fx_background_tint': ('cc', 34),
        }
        self.learning_key = None

        self.setStatusBar(QStatusBar(self))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(10, 10, 10, 10)

        self.video_display = VideoDisplay()
        self.video_display.setMinimumWidth(800)
        self.projector_window = ProjectorWindow()

        self.available_cameras = get_available_cameras()
        self.screens = QApplication.screens()

        self.worker = Worker()

        self.tabs = QTabWidget()
        self.create_setup_tab() # New Guided Setup Tab
        self.create_workspace_tab()
        self.create_media_tab()
        self.create_calibration_tab()
        self.create_boundary_tab()
        self.create_system_tab()
        self.create_diagnostics_tab()

        self.start_osc_server()

        self.layout.addWidget(self.video_display, stretch=2)
        self.layout.addWidget(self.tabs, stretch=1)
        self.video_display.mask_point_added.connect(self.add_mask_point_to_list)

        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.frame_ready.connect(self.video_display.set_image)
        self.worker.projector_frame_ready.connect(self.projector_window.set_image)
        self.projector_window.warp_points_changed.connect(lambda pts, res: self.worker.set_warp_points(pts, res))
        self.worker.trackers_detected.connect(self.update_tracker_label)
        self.worker.trackers_ready.connect(self.video_display.set_detected_markers)
        self.worker.camera_error.connect(self.show_camera_error)
        self.worker.system_warning.connect(lambda msg: self.statusBar().showMessage(msg, 10000))
        self.worker.calibration_complete.connect(self.handle_calibration_complete)
        self.worker.boundary_detected.connect(self.handle_boundary_detected)
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

    def add_media_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Media Files", "", "All Supported (*.mp4 *.mkv *.avi *.mov *.jpg *.png *.jpeg *.bmp *.webp);;Videos (*.mp4 *.mkv *.avi *.mov);;Images (*.jpg *.png *.jpeg *.bmp *.webp)")
        for f in files:
            if f not in self.media_library:
                self.media_library.append(f)
                self.media_list.addItem(f.split('/')[-1])

    def add_media_folder(self):
        import os
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            for f in os.listdir(path):
                if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.jpg', '.png', '.jpeg', '.bmp', '.webp')):
                    full_path = os.path.join(path, f).replace('\\', '/')
                    if full_path not in self.media_library:
                        self.media_library.append(full_path)
                        self.media_list.addItem(f)

    def update_mask_combos(self):
        mask_names = [mask.name for mask in self.masks]

        for combo_name in ['setup_link_mask_combo', 'workspace_link_mask_combo']:
            combo = getattr(self, combo_name, None)
            if combo:
                try:
                    curr = combo.currentText()
                    combo.blockSignals(True)
                    combo.clear()
                    combo.addItems(mask_names)
                    if curr in mask_names:
                        combo.setCurrentText(curr)
                    combo.blockSignals(False)
                except RuntimeError: pass

        self.refresh_link_status_labels()

    def refresh_link_status_labels(self):
        try:
            if getattr(self, 'setup_link_status_label', None) and getattr(self, 'setup_link_mask_combo', None):
                selected = self.setup_link_mask_combo.currentText()
                linked = any(m.name == selected and m.is_linked for m in self.masks)
                if linked:
                    self.setup_link_status_label.setText("Status: LINKED")
                    self.setup_link_status_label.setStyleSheet("font-weight: bold; color: #00c853;")
                else:
                    self.setup_link_status_label.setText("Status: Not Linked")
                    self.setup_link_status_label.setStyleSheet("font-weight: bold; color: #ff5252;")
        except (RuntimeError, AttributeError):
            pass

        try:
            if getattr(self, 'workspace_link_status_label', None) and getattr(self, 'workspace_link_mask_combo', None):
                selected = self.workspace_link_mask_combo.currentText()
                linked = any(m.name == selected and m.is_linked for m in self.masks)
                if linked:
                    self.workspace_link_status_label.setText("Status: LINKED")
                    self.workspace_link_status_label.setStyleSheet("font-weight: bold; color: #00c853;")
                else:
                    self.workspace_link_status_label.setText("Status: Not Linked")
                    self.workspace_link_status_label.setStyleSheet("font-weight: bold; color: #ff5252;")
        except (RuntimeError, AttributeError):
            pass

    def assign_media_to_mask(self):
        selected_media_indices = [idx.row() for idx in self.media_list.selectedIndexes()]
        selected_mask_row = self.cue_table.currentRow()

        if not selected_media_indices:
            self.statusBar().showMessage("Select media from library first.", 3000)
            return
        if selected_mask_row < 0:
            self.statusBar().showMessage("Select a mask from the table first.", 3000)
            return

        mask = self.masks[selected_mask_row]
        mask.playlist = [self.media_library[i] for i in selected_media_indices]
        mask.playlist_index = 0
        if mask.playlist:
            mask.video_path = mask.playlist[0]

        self.update_cue_table()
        self.worker.set_masks(self.masks)
        self.statusBar().showMessage(f"Assigned {len(mask.playlist)} videos to {mask.name}", 3000)
        self.maybe_auto_save()

    def update_play_button_ui(self):
        row = self.cue_table.currentRow()
        if row >= 0 and row < len(self.masks):
            mask = self.masks[row]
            if mask.visible:
                self.play_cue_btn.setText("■ STOP CUE")
                self.play_cue_btn.setStyleSheet("height: 50px; font-weight: bold; background-color: #ff5252; color: white; margin-top: 10px;")
            else:
                self.play_cue_btn.setText("▶ START CUE")
                self.play_cue_btn.setStyleSheet("height: 50px; font-weight: bold; background-color: #00c853; color: black; margin-top: 10px;")
        else:
            self.play_cue_btn.setText("▶ START CUE")
            self.play_cue_btn.setStyleSheet("height: 50px; font-weight: bold; background-color: #00c853; color: black; margin-top: 10px;")

    def update_cue_table(self):
        self.cue_table.setRowCount(len(self.masks))
        for i, mask in enumerate(self.masks):
            link_prefix = "🔗 " if mask.is_linked else ""
            name_item = QTableWidgetItem(f"{link_prefix}{mask.name} ({mask.tag})")
            if mask.is_linked:
                name_item.setForeground(Qt.green)

            curr_video = mask.video_path.split('/')[-1] if mask.video_path else "None"
            video_item = QTableWidgetItem(curr_video)
            count_item = QTableWidgetItem(str(len(mask.playlist)))

            self.cue_table.setItem(i, 0, name_item)
            self.cue_table.setItem(i, 1, video_item)
            self.cue_table.setItem(i, 2, count_item)
        self.update_cue_list_widget()
        self.update_play_button_ui()

    def update_cue_list_widget(self):
        curr_row = self.cue_list_widget.currentRow()
        self.cue_list_widget.blockSignals(True)
        self.cue_list_widget.clear()
        for mask in self.masks:
            display_name = f"🔗 {mask.name}" if mask.is_linked else mask.name
            self.cue_list_widget.addItem(display_name)
        if 0 <= curr_row < self.cue_list_widget.count():
            self.cue_list_widget.setCurrentRow(curr_row)
        self.cue_list_widget.blockSignals(False)

    def on_mask_selected(self, row):
        if 0 <= row < len(self.masks):
            mask = self.masks[row]
            self.mask_tag_combo.setCurrentText(mask.tag or "none")
            self.mask_type_combo.setCurrentText(mask.type)
            self.mask_blend_combo.setCurrentText(mask.blend_mode)
            self.bezier_check.setChecked(mask.bezier_enabled)
            self.mask_feather_slider.setValue(mask.feather)
            self.mask_opacity_slider.setValue(int(mask.opacity * 100))

            # Update Link Status Labels
            status_text = "Status: LINKED" if mask.is_linked else "Status: Not Linked"
            status_color = "#00c853" if mask.is_linked else "#ff5252"

            if getattr(self, 'setup_link_status_label', None):
                try:
                    self.setup_link_status_label.setText(status_text)
                    self.setup_link_status_label.setStyleSheet(f"font-weight: bold; color: {status_color};")
                except RuntimeError: pass
            if getattr(self, 'workspace_link_status_label', None):
                try:
                    self.workspace_link_status_label.setText(status_text)
                    self.workspace_link_status_label.setStyleSheet(f"font-weight: bold; color: {status_color};")
                except RuntimeError: pass

            # Update combo selection to match
            if getattr(self, 'setup_link_mask_combo', None):
                self.setup_link_mask_combo.setCurrentText(mask.name)
            if hasattr(self, 'workspace_link_mask_combo'):
                self.workspace_link_mask_combo.setCurrentText(mask.name)

            # Update display color
            color = Qt.magenta
            if mask.tag == 'background': color = Qt.blue
            elif mask.tag == 'amp': color = Qt.green
            self.video_display.set_mask_color(color)

            # Update LFO UI
            self.lfo_enable_check.setChecked(mask.fx_params.get('lfo_enabled', False))
            self.lfo_target_combo.setCurrentText(mask.fx_params.get('lfo_target', 'none'))
            self.lfo_shape_combo.setCurrentText(mask.fx_params.get('lfo_shape', 'sine'))
            self.lfo_speed_slider.setValue(int(mask.fx_params.get('lfo_speed', 1.0) * 10))

    def update_mask_mod(self):
        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            if 0 <= row < len(self.masks):
                mask = self.masks[row]
                mask.fx_params['lfo_enabled'] = self.lfo_enable_check.isChecked()
                mask.fx_params['lfo_target'] = self.lfo_target_combo.currentText()
                mask.fx_params['lfo_shape'] = self.lfo_shape_combo.currentText()
                mask.fx_params['lfo_speed'] = self.lfo_speed_slider.value() / 10.0
                self.worker.set_masks(self.masks)
                self.maybe_auto_save()

    def update_mask_opacity(self, value):
        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            if 0 <= row < len(self.masks):
                self.masks[row].opacity = value / 100.0
                self.worker.set_masks(self.masks)
                self.maybe_auto_save()

    def open_marker_selection_dialog(self):
        self.marker_selection_dialog.clear_selection()
        try:
            self.marker_selection_dialog.take_picture_button.clicked.disconnect()
        except TypeError:
            pass
        self.marker_selection_dialog.take_picture_button.clicked.connect(self.start_marker_capture_countdown)

        res = self.marker_selection_dialog.exec_()
        if res:
            import numpy as np
            new_markers = self.marker_selection_dialog.get_selected_points()

            # If we already have a configuration, try to transition masks
            if self.worker.marker_config and len(new_markers) == len(self.worker.marker_config) and len(new_markers) >= 4:
                # markers in dialog are in still-frame pixels
                w_still = self.marker_selection_dialog.image_label.pix.width()
                h_still = self.marker_selection_dialog.image_label.pix.height()

                # Both old and new pts are now normalized for resolution-independence
                old_pts = np.float32(self.worker.marker_config).reshape(-1, 1, 2)
                new_pts = np.float32([(p.x() / w_still, p.y() / h_still) for p in new_markers]).reshape(-1, 1, 2)
                matrix, _ = cv2.findHomography(old_pts, new_pts)

                if matrix is not None:
                    for mask in self.masks:
                        # Only transform static masks; linked ones are already relative to markers
                        if not mask.is_linked and mask.source_points:
                            # mask.source_points are normalized
                            pts = np.float32(mask.source_points).reshape(-1, 1, 2)
                            trans_pts = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
                            mask.source_points = [(float(p[0]), float(p[1])) for p in trans_pts]

                    self.statusBar().showMessage("Adjusted static masks to new camera perspective.", 3000)

            self.selected_markers = new_markers
            print(f"Selected {len(self.selected_markers)} markers.")
            self.worker.set_marker_points(self.selected_markers)
            self.maybe_auto_save()
        return res

    def start_marker_capture_countdown(self):
        self.marker_selection_dialog.take_picture_button.setEnabled(False)
        self.countdown_seconds = 7
        if not hasattr(self, 'countdown_timer'):
            self.countdown_timer = QTimer(self)
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

    def set_marker_selection_image(self, image, points):
        guide_pts = self.worker.marker_config if hasattr(self.worker, 'marker_config') else None
        self.marker_selection_dialog.set_pixmap(QPixmap.fromImage(image), points, guide_pts)

    def clear_marker_selection(self):
        self.selected_markers = []
        self.worker.clear_marker_config()
        self.statusBar().showMessage("Marker selection cleared.", 3000)

    def create_setup_tab(self):
        self.setup_tab = QWidget()
        self.setup_layout = QVBoxLayout(self.setup_tab)
        self.setup_layout.setSpacing(12)
        self.setup_layout.setContentsMargins(15, 15, 15, 15)

        self.setup_title = QLabel("<h2>Guided Setup</h2>")
        self.setup_desc = QLabel("Welcome! Let's get your projection mapped. Follow the steps below.")
        self.setup_desc.setWordWrap(True)

        self.setup_layout.addWidget(self.setup_title)
        self.setup_layout.addWidget(self.setup_desc)

        self.setup_group = QGroupBox("Step 1: Devices & Boundary Detection")
        self.setup_group_layout = QVBoxLayout()

        self.setup_instruction = QLabel("Select your camera and projector, then perform a ONE-CLICK SYNC to align everything.")
        self.setup_instruction.setWordWrap(True)
        self.setup_group_layout.addWidget(self.setup_instruction)

        # Camera selection
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Camera:"))
        self.setup_cam_combo = QComboBox()
        self.setup_cam_combo.addItems([f"Camera {i}" for i in self.available_cameras])
        self.setup_cam_combo.currentIndexChanged.connect(self.change_camera)
        cam_layout.addWidget(self.setup_cam_combo)
        self.setup_group_layout.addLayout(cam_layout)

        # Projector selection
        proj_layout = QHBoxLayout()
        proj_layout.addWidget(QLabel("Projector Display:"))
        self.setup_proj_combo = QComboBox()
        self.setup_proj_combo.addItems([screen.name() or f"Screen {i+1}" for i, screen in enumerate(self.screens)])
        self.setup_proj_combo.currentIndexChanged.connect(self.change_projector)
        proj_layout.addWidget(self.setup_proj_combo)
        self.setup_group_layout.addLayout(proj_layout)

        # One-Click Sync
        self.sync_btn = QPushButton("ONE-CLICK SYNC (ALIGN & BOUNDS)")
        self.sync_btn.setMinimumHeight(70)
        self.sync_btn.setStyleSheet("background-color: #d500f9; color: white; font-weight: bold; font-size: 16px; margin-top: 10px;")
        self.sync_btn.clicked.connect(self.start_one_click_sync)
        self.setup_group_layout.addWidget(self.sync_btn)

        # Boundary Detection (Legacy/Fallback)
        self.detect_bounds_btn = QPushButton("Only Detect Projector Bounds")
        self.detect_bounds_btn.clicked.connect(self.worker.run_boundary_detection)
        self.setup_group_layout.addWidget(self.detect_bounds_btn)

        self.setup_status_label = QLabel("Mapping: Not Aligned")
        self.setup_status_label.setAlignment(Qt.AlignCenter)
        self.setup_status_label.setStyleSheet("color: #ff5252; font-weight: bold;")
        self.setup_group_layout.addWidget(self.setup_status_label)

        self.load_template_btn = QPushButton("OR: Load Existing Template / Project")
        self.load_template_btn.clicked.connect(self.load_project)
        self.load_template_btn.setStyleSheet("background-color: #311b92; color: white; margin-top: 20px;")
        self.setup_group_layout.addWidget(self.load_template_btn)

        self.setup_next_btn = QPushButton("Next Step")
        self.setup_next_btn.setMinimumHeight(50)
        self.setup_next_btn.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")
        self.setup_next_btn.clicked.connect(self.next_setup_step)

        self.setup_group.setLayout(self.setup_group_layout)
        self.setup_layout.addWidget(self.setup_group)
        self.setup_layout.addWidget(self.setup_next_btn)

        self.setup_layout.addStretch()
        self.tabs.insertTab(0, self.setup_tab, "Setup Wizard")

    def start_auto_calibration(self):
        self.statusBar().showMessage("Displaying calibration pattern...", 2000)
        self.worker.show_calibration_pattern = True
        # Wait a moment for the projector to actually show it before capturing
        QTimer.singleShot(1000, self.worker.run_auto_calibration)

    def start_room_scan(self):
        self.statusBar().showMessage("Scanning room... This will take about 10 seconds. Please keep the area still.", 0)
        self.worker.run_room_scan()

    def start_one_click_sync(self):
        if hasattr(self, 'sync_btn') and self.sync_btn:
            self.sync_btn.setText("SYNCING... PLEASE WAIT")
            self.sync_btn.setEnabled(False)
        if hasattr(self, 'tab_sync_btn') and self.tab_sync_btn:
            self.tab_sync_btn.setText("SYNCING...")
            self.tab_sync_btn.setEnabled(False)

        self.statusBar().showMessage("Starting ONE-CLICK SYNC. Analyzing environment and aligning...", 0)
        self.worker.run_one_click_sync()

    def start_manual_calibration(self):
        self.statusBar().showMessage("Manual Alignment: Click the 4 corners of the projector's output.", 5000)
        self.worker.show_calibration_pattern = True

        # We reuse MarkerSelectionDialog but for 4 corners
        self.marker_selection_dialog.setWindowTitle("Manual Alignment - Select 4 Projector Corners")
        self.worker.capture_still_frame()

        if self.marker_selection_dialog.exec_():
            points = self.marker_selection_dialog.get_selected_points()
            if len(points) == 4:
                import numpy as np
                cam_pts = np.array([(p.x(), p.y()) for p in points], dtype=np.float32)

                # Sort them
                s = cam_pts.sum(axis=1)
                tl = cam_pts[np.argmin(s)]
                br = cam_pts[np.argmax(s)]
                diff = np.diff(cam_pts, axis=1)
                tr = cam_pts[np.argmin(diff)]
                bl = cam_pts[np.argmax(diff)]
                cam_ordered = np.array([tl, tr, br, bl], dtype=np.float32)

                w = self.worker.projector_width
                h = self.worker.projector_height
                proj_ordered = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

                h_matrix, _ = cv2.findHomography(cam_ordered, proj_ordered)
                self.worker.set_h_c2p(h_matrix.tolist(), cam_res=self.worker._current_camera_res)
                self.handle_calibration_complete(True)
            else:
                self.statusBar().showMessage("Manual Alignment failed: Please select exactly 4 corners.", 5000)

        self.worker.show_calibration_pattern = False
        self.marker_selection_dialog.setWindowTitle("Select Markers")

    def handle_calibration_complete(self, success):
        # Reset sync buttons if they were used
        if hasattr(self, 'sync_btn') and self.sync_btn:
            self.sync_btn.setText("ONE-CLICK SYNC (ALIGN & BOUNDS)")
            self.sync_btn.setEnabled(True)
        if hasattr(self, 'tab_sync_btn') and self.tab_sync_btn:
            self.tab_sync_btn.setText("ONE-CLICK SYNC (AUTO-ALIGN)")
            self.tab_sync_btn.setEnabled(True)

        self.worker.show_calibration_pattern = False
        # Safe check for widget existence to avoid RuntimeError if deleted during step transition
        try:
            if getattr(self, 'test_pattern_check', None):
                self.test_pattern_check.blockSignals(True)
                self.test_pattern_check.setChecked(False)
                self.test_pattern_check.blockSignals(False)
        except RuntimeError:
            self.test_pattern_check = None
        if success:
            self.statusBar().showMessage("Auto-Alignment Successful!", 5000)
            if hasattr(self, 'setup_status_label') and self.setup_status_label:
                self.setup_status_label.setText("Mapping: ALIGNED ✓")
                self.setup_status_label.setStyleSheet("color: #00c853; font-weight: bold;")
            self.maybe_auto_save()
        else:
            self.statusBar().showMessage("Auto-Alignment Failed. Ensure camera sees projector.", 5000)
            if hasattr(self, 'setup_status_label') and self.setup_status_label:
                self.setup_status_label.setText("Mapping: Failed (Retry)")
                self.setup_status_label.setStyleSheet("color: #ff5252; font-weight: bold;")

    def clear_setup_layout(self):
        # Reset specific references to avoid RuntimeErrors on deleted widgets
        self.setup_link_status_label = None
        self.setup_link_mask_combo = None
        self.setup_status_label = None
        self.test_pattern_check = None
        self.verify_align_btn = None
        self.scan_room_btn = None
        self.manual_align_btn = None
        self.align_btn = None
        self.detect_bounds_btn = None
        self.sync_btn = None

        while self.setup_group_layout.count():
            item = self.setup_group_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

    def handle_boundary_detected(self, points):
        if not points:
            self.statusBar().showMessage("Boundary Detection Failed! Ensure camera sees projector.", 5000)
            return

        self.statusBar().showMessage(f"Detected Projector Boundary with {len(points)} points.", 3000)

        # Create/Update Background Mask
        bg_mask = None
        for m in self.masks:
            if m.tag == 'background':
                bg_mask = m
                break

        if not bg_mask:
            bg_mask = Mask("Background", points, None, tag="background", mask_type="static")
            self.masks.append(bg_mask)
        else:
            bg_mask.source_points = points
            bg_mask.visible = True

        self.update_cue_table()
        self.update_mask_combos()
        self.worker.set_masks(self.masks)

        # Advance Wizard or update UI
        if self.setup_step == 0:
            try:
                if self.setup_status_label:
                    self.setup_status_label.setText("Boundary: DETECTED ✓")
                    self.setup_status_label.setStyleSheet("color: #00c853; font-weight: bold;")
            except RuntimeError: pass

    def next_setup_step(self):
        # Ensure any active calibration is stopped before moving to next step
        self.worker.stop_calibration()

        self.setup_step += 1
        self.clear_setup_layout()

        # Add common instruction label back
        self.setup_instruction = QLabel()
        self.setup_instruction.setWordWrap(True)
        self.setup_group_layout.addWidget(self.setup_instruction)

        if self.setup_step == 1: # Alignment
            self.setup_group.setTitle("Step 2: Verification")
            self.setup_instruction.setText("Verify that the alignment and background bounds are correct. If the background boundary is incorrect, you can manually adjust it in the 'Boundary' tab.")

            self.verify_align_btn = QPushButton("VERIFY ALIGNMENT (GRID)")
            self.verify_align_btn.setCheckable(True)
            self.verify_align_btn.toggled.connect(self.toggle_verify_alignment)
            self.setup_group_layout.addWidget(self.verify_align_btn)

            self.test_pattern_check = QCheckBox("Show Test Pattern (Manual Alignment Check)")
            self.test_pattern_check.toggled.connect(self.toggle_test_pattern)
            self.setup_group_layout.addWidget(self.test_pattern_check)
        elif self.setup_step == 2: # Guitar Markers
            self.setup_group.setTitle("Step 3: Guitar Markers")
            self.setup_instruction.setText("Identify the IR markers on your guitar. This allows the system to track its movement.")

            # IR Threshold Slider
            ir_label = QLabel("IR Threshold (Adjust until only dots are visible):")
            self.setup_group_layout.addWidget(ir_label)
            wizard_ir_slider = QSlider(Qt.Horizontal)
            wizard_ir_slider.setRange(0, 255)
            wizard_ir_slider.setValue(self.ir_threshold_slider.value())
            wizard_ir_slider.valueChanged.connect(self.update_ir_threshold)
            self.setup_group_layout.addWidget(wizard_ir_slider)

            marker_btn = QPushButton("CALIBRATE MARKERS")
            marker_btn.clicked.connect(self.open_marker_selection_dialog)
            marker_btn.setMinimumHeight(60)
            marker_btn.setStyleSheet("background-color: #00c853; color: black; font-weight: bold;")
            self.setup_group_layout.addWidget(marker_btn)

        elif self.setup_step == 3: # Guitar Mask
            self.setup_group.setTitle("Step 4: Guitar Mask (DYNAMIC)")
            self.setup_instruction.setText("Draw the shape of your guitar. This mask will AUTOMATICALLY FOLLOW the markers you calibrated.")

            mask_btn = QPushButton("DRAW TRACKED GUITAR MASK")
            mask_btn.clicked.connect(self.start_setup_guitar_mask)
            mask_btn.setMinimumHeight(60)
            mask_btn.setStyleSheet("background-color: #00c853; color: black; font-weight: bold;")
            self.setup_group_layout.addWidget(mask_btn)

            self.setup_link_status_label = QLabel("Status: Not Linked")
            self.setup_link_status_label.setStyleSheet("font-weight: bold; color: #ff5252;")
            self.setup_group_layout.addWidget(self.setup_link_status_label)
            self.update_mask_combos()

        elif self.setup_step == 4: # Amp Mask
            self.setup_group.setTitle("Step 5: Amp Mask (STATIC)")
            self.setup_instruction.setText("Draw the shape of your stationary amp or cabinet. This mask will stay in place.")

            mask_btn = QPushButton("DRAW STATIONARY AMP MASK")
            mask_btn.clicked.connect(self.start_setup_amp_mask)
            mask_btn.setMinimumHeight(60)
            mask_btn.setStyleSheet("background-color: #00c853; color: black; font-weight: bold;")
            self.setup_group_layout.addWidget(mask_btn)

        elif self.setup_step == 5: # Done
            self.setup_group.setTitle("Setup Complete")
            self.setup_instruction.setText("Setup finished! You can save this configuration as a preset below.")

            save_preset_btn = QPushButton("Save Project / Preset")
            save_preset_btn.clicked.connect(lambda: self.save_project())
            save_preset_btn.setStyleSheet("background-color: #4a148c; color: white; height: 40px; margin-bottom: 10px;")
            self.setup_group_layout.addWidget(save_preset_btn)

            self.setup_next_btn.setText("Enter Performance Mode")
            self.setup_next_btn.clicked.disconnect()
            self.setup_next_btn.clicked.connect(self.enter_performance_mode)

    def add_wizard_finish_button(self):
        # Check if button already exists in the layout
        for i in range(self.setup_group_layout.count()):
            widget = self.setup_group_layout.itemAt(i).widget()
            if widget and isinstance(widget, QPushButton) and widget.text() in ["Finish & Save Mask", "Mask Saved!"]:
                widget.setEnabled(True)
                widget.setText("Finish & Save Mask")
                return

        finish_btn = QPushButton("Finish & Save Mask")
        finish_btn.setStyleSheet("background-color: #00c853; color: white; font-weight: bold; height: 40px; margin-top: 10px;")
        finish_btn.clicked.connect(self.finish_mask_creation)
        # Also disable the button once clicked to avoid confusion
        finish_btn.clicked.connect(lambda: finish_btn.setEnabled(False))
        finish_btn.clicked.connect(lambda: finish_btn.setText("Mask Saved!"))
        self.setup_group_layout.addWidget(finish_btn)

    def start_setup_bg_mask(self):
        self.video_display.clear_mask_points()
        self.video_display.set_snap_to_markers(False)
        self.video_display.set_mask_color(Qt.blue)
        # Ensure a background cue exists
        bg_mask = None
        for m in self.masks:
            if m.tag == 'background':
                bg_mask = m
                break

        if not bg_mask:
            bg_mask = Mask("Background", [], None, tag="background", mask_type="static")
            self.masks.append(bg_mask)
            self.update_cue_table()
            self.update_mask_combos()

        idx = self.masks.index(bg_mask)
        self.cue_list_widget.setCurrentRow(idx)
        # Sync workspace UI so saving works correctly
        self.mask_tag_combo.setCurrentText("background")
        self.mask_type_combo.setCurrentText("static")
        self.add_wizard_finish_button()
        self.enter_mask_creation_mode()

    def start_guided_guitar_setup(self):
        if self.open_marker_selection_dialog():
            self.start_setup_amp_mask()

    def start_setup_guitar_mask(self):
        self.video_display.clear_mask_points()
        self.video_display.set_snap_to_markers(True)
        self.video_display.set_mask_color(Qt.green)
        mask = None
        for m in self.masks:
            if m.name == 'Guitar':
                mask = m
                break

        if not mask:
            mask = Mask("Guitar", [], None, tag="amp", mask_type="dynamic")
            self.masks.append(mask)
            self.update_cue_table()
            self.update_mask_combos()

        idx = self.masks.index(mask)
        self.cue_list_widget.setCurrentRow(idx)
        self.mask_tag_combo.setCurrentText("amp")
        self.mask_type_combo.setCurrentText("dynamic")
        self.add_wizard_finish_button()
        self.enter_mask_creation_mode()

    def start_setup_amp_mask(self):
        self.video_display.clear_mask_points()
        self.video_display.set_snap_to_markers(False)
        self.video_display.set_mask_color(Qt.cyan)
        mask = None
        for m in self.masks:
            if m.name == 'Amp':
                mask = m
                break

        if not mask:
            mask = Mask("Amp", [], None, tag="background", mask_type="static")
            self.masks.append(mask)
            self.update_cue_table()
            self.update_mask_combos()

        idx = self.masks.index(mask)
        self.cue_list_widget.setCurrentRow(idx)
        self.mask_tag_combo.setCurrentText("background")
        self.mask_type_combo.setCurrentText("static")
        self.add_wizard_finish_button()
        self.enter_mask_creation_mode()

    def enter_performance_mode(self):
        # Switch to the Stage tab (which has basic performance controls)
        # or we could hide the tabs entirely and just show the video.
        # Let's try hiding the sidebar (tabs)
        self.tabs.hide()
        self.video_display.setMinimumWidth(1200) # Take more space

        # Add a floating button or status bar button to exit
        self.exit_perf_btn = QPushButton("EXIT PERFORMANCE MODE")
        self.exit_perf_btn.setStyleSheet("background-color: #aa00ff; color: white; font-weight: bold; height: 40px;")
        self.exit_perf_btn.clicked.connect(self.exit_performance_mode)
        self.layout.addWidget(self.exit_perf_btn)

        self.statusBar().showMessage("PERFORMANCE MODE ACTIVE - Listening for MIDI/OSC", 0)

    def exit_performance_mode(self):
        self.tabs.show()
        if hasattr(self, 'exit_perf_btn'):
            self.layout.removeWidget(self.exit_perf_btn)
            self.exit_perf_btn.deleteLater()
            del self.exit_perf_btn
        self.video_display.setMinimumWidth(800)
        self.statusBar().showMessage("Configuration Mode", 3000)

    def create_workspace_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Project controls
        proj_group = QGroupBox("Project")
        proj_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_project)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_project)
        btn_layout.addWidget(self.save_button)
        btn_layout.addWidget(self.load_button)
        proj_layout.addLayout(btn_layout)

        self.auto_save_check = QCheckBox("Auto-Save on Changes")
        self.auto_save_check.setChecked(True)
        proj_layout.addWidget(self.auto_save_check)

        self.show_ref_btn = QPushButton("Show Setup Reference Frame")
        self.show_ref_btn.clicked.connect(self.show_setup_reference)
        proj_layout.addWidget(self.show_ref_btn)

        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)

        # Mask Creation
        mask_group = QGroupBox("Mask Editor")
        mask_layout = QVBoxLayout()

        self.cue_list_widget = QListWidget()
        self.cue_list_widget.currentRowChanged.connect(self.on_mask_selected)
        mask_layout.addWidget(self.cue_list_widget)

        btn_layout = QHBoxLayout()
        self.create_mask_button = QPushButton("Draw/Edit Points")
        self.create_mask_button.clicked.connect(self.enter_mask_creation_mode)
        self.add_mask_btn = QPushButton("Add New Mask")
        self.add_mask_btn.clicked.connect(self.add_new_mask_prompt)
        self.add_mask_btn.setStyleSheet("background-color: #1b5e20; color: white;")
        self.remove_cue_button = QPushButton("Delete")
        self.remove_cue_button.clicked.connect(self.remove_cue)
        btn_layout.addWidget(self.add_mask_btn)
        btn_layout.addWidget(self.create_mask_button)
        btn_layout.addWidget(self.remove_cue_button)
        mask_layout.addLayout(btn_layout)

        form = QFormLayout()
        self.mask_tag_combo = QComboBox()
        self.mask_tag_combo.addItems(["none", "amp", "background", "instrument"])
        form.addRow("Tag:", self.mask_tag_combo)

        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItems(["dynamic", "static"])
        form.addRow("Type:", self.mask_type_combo)

        self.mask_blend_combo = QComboBox()
        self.mask_blend_combo.addItems(["normal", "add", "screen", "multiply"])
        form.addRow("Blend:", self.mask_blend_combo)

        self.bezier_check = QCheckBox("Bezier Curves")
        form.addRow(self.bezier_check)

        self.mask_feather_slider = QSlider(Qt.Horizontal)
        self.mask_feather_slider.setRange(0, 100)
        form.addRow("Feather:", self.mask_feather_slider)

        self.mask_opacity_slider = QSlider(Qt.Horizontal)
        self.mask_opacity_slider.setRange(0, 100)
        self.mask_opacity_slider.setValue(100)
        self.mask_opacity_slider.valueChanged.connect(self.update_mask_opacity)
        form.addRow("Opacity:", self.mask_opacity_slider)

        mask_layout.addLayout(form)

        # Tracking & Link in Workspace
        self.tracking_section = QGroupBox("Tracking && Link")
        link_layout = QVBoxLayout()

        self.workspace_link_mask_combo = QComboBox()
        self.workspace_link_mask_combo.setMinimumHeight(30)
        self.workspace_link_mask_combo.currentIndexChanged.connect(self.refresh_link_status_labels)
        link_layout.addWidget(QLabel("Select Mask to Link:"))
        link_layout.addWidget(self.workspace_link_mask_combo)

        self.workspace_link_btn = QPushButton("Link Selected Mask to Tracking")
        self.workspace_link_btn.setStyleSheet("background-color: #311b92; color: white; min-height: 35px;")
        self.workspace_link_btn.clicked.connect(self.link_mask_to_markers)
        link_layout.addWidget(self.workspace_link_btn)

        self.workspace_link_status_label = QLabel("Status: Not Linked")
        self.workspace_link_status_label.setStyleSheet("font-weight: bold; color: #ff5252;")
        link_layout.addWidget(self.workspace_link_status_label)

        self.tracking_section.setLayout(link_layout)
        mask_layout.addWidget(self.tracking_section)

        edit_btns = QHBoxLayout()
        self.finish_mask_button = QPushButton("SAVE POINTS")
        self.finish_mask_button.setStyleSheet("background-color: #4a148c; color: white; font-weight: bold;")
        self.finish_mask_button.clicked.connect(self.finish_mask_creation)
        self.finish_mask_button.setEnabled(False)
        self.cancel_mask_button = QPushButton("Cancel")
        self.cancel_mask_button.clicked.connect(self.cancel_mask_creation)
        self.cancel_mask_button.setEnabled(False)
        edit_btns.addWidget(self.finish_mask_button)
        edit_btns.addWidget(self.cancel_mask_button)
        mask_layout.addLayout(edit_btns)

        layer_layout = QHBoxLayout()
        self.front_btn = QPushButton("Bring to Front")
        self.front_btn.clicked.connect(self.mask_to_front)
        self.back_btn = QPushButton("Send to Back")
        self.back_btn.clicked.connect(self.mask_to_back)
        layer_layout.addWidget(self.front_btn)
        layer_layout.addWidget(self.back_btn)
        mask_layout.addLayout(layer_layout)

        # Modulation (LFO) Group
        mod_group = QGroupBox("Modulation (LFO)")
        mod_layout = QFormLayout()

        self.lfo_enable_check = QCheckBox("Enable LFO")
        self.lfo_enable_check.toggled.connect(self.update_mask_mod)
        mod_layout.addRow(self.lfo_enable_check)

        self.lfo_target_combo = QComboBox()
        self.lfo_target_combo.addItems(["none", "blur", "tint", "rgb_shift", "hue", "opacity"])
        self.lfo_target_combo.currentIndexChanged.connect(self.update_mask_mod)
        mod_layout.addRow("Target:", self.lfo_target_combo)

        self.lfo_shape_combo = QComboBox()
        self.lfo_shape_combo.addItems(["sine", "square", "triangle", "sawtooth"])
        self.lfo_shape_combo.currentIndexChanged.connect(self.update_mask_mod)
        mod_layout.addRow("Shape:", self.lfo_shape_combo)

        self.lfo_speed_slider = QSlider(Qt.Horizontal)
        self.lfo_speed_slider.setRange(1, 100) # 0.1x to 10x
        self.lfo_speed_slider.setValue(10)
        self.lfo_speed_slider.valueChanged.connect(self.update_mask_mod)
        mod_layout.addRow("Speed Multiplier:", self.lfo_speed_slider)

        mod_group.setLayout(mod_layout)
        mask_layout.addWidget(mod_group)

        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)

        # FX Controls
        fx_group = QGroupBox("Performance")
        fx_layout = QVBoxLayout()
        self.splash_check = QPushButton("START SHOW (Disable Splash)")
        self.splash_check.setCheckable(True)
        self.splash_check.toggled.connect(self.toggle_splash_mode)
        fx_layout.addWidget(self.splash_check)

        self.style_combo = QComboBox()
        self.style_combo.addItems(["none", "acid", "noir", "retro"])
        self.style_combo.currentIndexChanged.connect(lambda i: self.worker.set_style(self.style_combo.currentText()))
        fx_layout.addWidget(QLabel("Global Style:"))
        fx_layout.addWidget(self.style_combo)

        self.mood_combo = QComboBox()
        self.mood_combo.addItems(list(self.moods.keys()))
        self.mood_combo.currentIndexChanged.connect(self.apply_mood)
        fx_layout.addWidget(QLabel("Global Mood (Color Palette):"))
        fx_layout.addWidget(self.mood_combo)

        self.pnp_check = QCheckBox("Enable 3D Perspective (Tilt/Rotation)")
        self.pnp_check.toggled.connect(self.worker.set_pnp_enabled)
        fx_layout.addWidget(self.pnp_check)

        self.occlusion_check = QCheckBox("Enable Performer Occlusion (Shadow Removal)")
        self.occlusion_check.toggled.connect(self.worker.set_occlusion_enabled)
        fx_layout.addWidget(self.occlusion_check)

        self.blackout_btn = QPushButton("BLACKOUT (PANIC)")
        self.blackout_btn.setStyleSheet("background-color: #aa00ff; color: white; font-weight: bold; height: 40px;")
        self.blackout_btn.clicked.connect(self.toggle_blackout)
        fx_layout.addWidget(self.blackout_btn)

        fx_group.setLayout(fx_layout)
        layout.addWidget(fx_group)

        # Master Visuals
        master_group = QGroupBox("Master Visuals")
        master_layout = QFormLayout()

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(lambda v: setattr(self.worker, 'master_brightness', v))
        master_layout.addRow("Brightness:", self.brightness_slider)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(lambda v: setattr(self.worker, 'master_contrast', v))
        master_layout.addRow("Contrast:", self.contrast_slider)

        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        self.saturation_slider.valueChanged.connect(lambda v: setattr(self.worker, 'master_saturation', v))
        master_layout.addRow("Saturation:", self.saturation_slider)

        self.grain_slider = QSlider(Qt.Horizontal)
        self.grain_slider.setRange(0, 100)
        self.grain_slider.setValue(0)
        self.grain_slider.valueChanged.connect(lambda v: setattr(self.worker, 'master_grain', v))
        master_layout.addRow("Visual Grain:", self.grain_slider)

        self.bloom_slider = QSlider(Qt.Horizontal)
        self.bloom_slider.setRange(0, 100)
        self.bloom_slider.setValue(0)
        self.bloom_slider.valueChanged.connect(lambda v: setattr(self.worker, 'master_bloom', v))
        master_layout.addRow("Bloom Intensity:", self.bloom_slider)

        self.master_fader_slider = QSlider(Qt.Horizontal)
        self.master_fader_slider.setRange(0, 100)
        self.master_fader_slider.setValue(100)
        self.master_fader_slider.valueChanged.connect(lambda v: setattr(self.worker, 'master_fader', v / 100.0))
        master_layout.addRow("GRAND MASTER FADER:", self.master_fader_slider)

        master_group.setLayout(master_layout)
        layout.addWidget(master_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Stage")

    def create_media_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Left: Media Library
        lib_group = QGroupBox("Media Library")
        lib_layout = QVBoxLayout()
        self.media_list = QListWidget()
        self.media_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lib_layout.addWidget(self.media_list)

        lib_btns = QHBoxLayout()
        add_btn = QPushButton("Add Files")
        add_btn.clicked.connect(self.add_media_files)
        add_dir_btn = QPushButton("Add Folder")
        add_dir_btn.clicked.connect(self.add_media_folder)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.media_list.clear)
        gen_btn = QPushButton("Add Generator")
        gen_btn.clicked.connect(self.add_vj_generator_to_lib)
        lib_btns.addWidget(add_btn)
        lib_btns.addWidget(add_dir_btn)
        lib_btns.addWidget(gen_btn)
        lib_btns.addWidget(clear_btn)
        lib_layout.addLayout(lib_btns)
        lib_group.setLayout(lib_layout)
        layout.addWidget(lib_group, 1)

        # Right: Cue Playlists
        cue_group = QGroupBox("Cue Management")
        cue_layout = QVBoxLayout()

        self.cue_table = QTableWidget(0, 3)
        self.cue_table.setHorizontalHeaderLabels(["Mask / Tag", "Current Video", "Playlist Count"])
        self.cue_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cue_table.itemSelectionChanged.connect(self.update_play_button_ui)
        cue_layout.addWidget(self.cue_table)

        assign_btn = QPushButton("Assign Selected Media to Mask Playlist")
        assign_btn.setStyleSheet("height: 40px; font-weight: bold;")
        assign_btn.clicked.connect(self.assign_media_to_mask)
        cue_layout.addWidget(assign_btn)

        self.edit_playlist_btn = QPushButton("Edit Playlist Order / Manage Cue")
        self.edit_playlist_btn.setStyleSheet("height: 40px; font-weight: bold; background-color: #311b92;")
        self.edit_playlist_btn.clicked.connect(self.open_playlist_editor)
        cue_layout.addWidget(self.edit_playlist_btn)

        self.play_cue_btn = QPushButton("▶ START CUE")
        self.play_cue_btn.setStyleSheet("height: 50px; font-weight: bold; background-color: #00c853; color: black; margin-top: 10px;")
        self.play_cue_btn.clicked.connect(self.toggle_selected_cue)
        cue_layout.addWidget(self.play_cue_btn)

        cue_group.setLayout(cue_layout)
        layout.addWidget(cue_group, 2)

        self.tabs.addTab(tab, "Media & Cues")

    def create_boundary_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("Projector Usable Area (Camera View)")
        glayout = QVBoxLayout()

        glayout.addWidget(QLabel("This defines where the projector light is visible to the camera."))

        self.edit_bounds_btn = QPushButton("MANUALLY EDIT BOUNDARY")
        self.edit_bounds_btn.setCheckable(True)
        self.edit_bounds_btn.toggled.connect(self.toggle_boundary_edit)
        glayout.addWidget(self.edit_bounds_btn)

        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Point Count:"))
        self.bounds_points_spin = QComboBox()
        self.bounds_points_spin.addItems(["4", "6", "8", "10", "12"])
        self.bounds_points_spin.setCurrentText("8")
        num_layout.addWidget(self.bounds_points_spin)
        self.bounds_points_spin.currentTextChanged.connect(self.reset_boundary_to_count)
        glayout.addLayout(num_layout)

        self.apply_bounds_btn = QPushButton("SAVE BOUNDARY AS BACKGROUND")
        self.apply_bounds_btn.clicked.connect(self.save_boundary_as_bg)
        glayout.addWidget(self.apply_bounds_btn)

        self.clear_bounds_btn = QPushButton("CLEAR ALL POINTS")
        self.clear_bounds_btn.clicked.connect(lambda: self.video_display.clear_mask_points())
        glayout.addWidget(self.clear_bounds_btn)

        group.setLayout(glayout)
        layout.addWidget(group)
        layout.addStretch()
        self.tabs.addTab(tab, "Boundary")

    def reset_boundary_to_count(self, count_str):
        if not self.edit_bounds_btn.isChecked(): return

        count = int(count_str)
        # Generate a simple centered polygon with N points
        w_cam, h_cam = 640, 480 # Standard camera resolution used in worker
        cx, cy = w_cam / 2, h_cam / 2
        rx, ry = w_cam * 0.4, h_cam * 0.4

        new_pts = []
        import numpy as np
        for i in range(count):
            angle = 2 * np.pi * i / count
            px = cx + rx * np.cos(angle)
            py = cy + ry * np.sin(angle)
            new_pts.append(QPointF(px, py))

        self.video_display.set_mask_points(new_pts)
        self.worker.projector_boundary = [(p.x(), p.y()) for p in new_pts]

    def toggle_boundary_edit(self, checked):
        if checked:
            self.video_display.clear_mask_points()
            if self.worker.projector_boundary:
                # Load current boundary points into display
                pts = [QPointF(p[0], p[1]) for p in self.worker.projector_boundary]
                self.video_display.set_mask_points(pts)
            self.video_display.set_mask_creation_mode(True, Qt.yellow)
            self.edit_bounds_btn.setText("FINISH EDITING")
        else:
            self.video_display.set_mask_creation_mode(False)
            self.edit_bounds_btn.setText("MANUALLY EDIT BOUNDARY")
            # Don't save yet, wait for apply button or just auto-update?
            # User said "drag corners", so we should update worker immediately or on finish.
            pts = list(self.video_display.get_mask_points())
            if pts:
                self.worker.projector_boundary = [(p.x(), p.y()) for p in pts]

    def save_boundary_as_bg(self):
        pts = self.worker.projector_boundary
        if not pts:
            self.statusBar().showMessage("No boundary defined!", 3000)
            return

        self.handle_boundary_detected(pts)
        self.statusBar().showMessage("Boundary saved as Background mask.", 3000)

    def create_calibration_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.warp_group = QGroupBox("Projector Warping (3x3 Grid)")
        warp_layout = QVBoxLayout()

        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Grid Resolution:"))
        self.warp_res_combo = QComboBox()
        self.warp_res_combo.addItems(["3x3", "4x4", "5x5", "6x6", "8x8"])
        self.warp_res_combo.currentIndexChanged.connect(self.change_warp_res)
        res_layout.addWidget(self.warp_res_combo)
        warp_layout.addLayout(res_layout)

        self.enable_warping_button = QPushButton("ENABLE ALIGNMENT MODE")
        self.enable_warping_button.setCheckable(True)
        self.enable_warping_button.toggled.connect(self.toggle_warping)
        self.reset_warping_button = QPushButton("Reset Grid")
        self.reset_warping_button.clicked.connect(self.projector_window.reset_warp_points)
        self.tab_sync_btn = QPushButton("ONE-CLICK SYNC (AUTO-ALIGN)")
        self.tab_sync_btn.clicked.connect(self.start_one_click_sync)
        self.tab_sync_btn.setStyleSheet("background-color: #d500f9; color: white; font-weight: bold;")

        warp_layout.addWidget(self.enable_warping_button)
        warp_layout.addWidget(self.reset_warping_button)
        warp_layout.addWidget(self.tab_sync_btn)
        self.warp_group.setLayout(warp_layout)
        layout.addWidget(self.warp_group)
        
        ir_group = QGroupBox("IR Tracking")
        ir_layout = QVBoxLayout()
        self.auto_ir_check = QCheckBox("Auto Threshold")
        self.auto_ir_check.toggled.connect(self.toggle_auto_ir)
        ir_layout.addWidget(self.auto_ir_check)

        self.ir_threshold_slider = QSlider(Qt.Horizontal)
        self.ir_threshold_slider.setRange(0, 255)
        self.ir_threshold_slider.setValue(200)
        self.ir_threshold_slider.valueChanged.connect(self.update_ir_threshold)
        ir_layout.addWidget(QLabel("IR Threshold:"))
        ir_layout.addWidget(self.ir_threshold_slider)

        self.ir_trackers_label = QLabel("Trackers detected: 0")
        ir_layout.addWidget(self.ir_trackers_label)

        self.select_markers_button = QPushButton("Calibrate Guitar Markers")
        self.select_markers_button.clicked.connect(self.open_marker_selection_dialog)
        ir_layout.addWidget(self.select_markers_button)

        ir_group.setLayout(ir_layout)
        layout.addWidget(ir_group)

        depth_group = QGroupBox("Depth & Smoothing")
        depth_layout = QFormLayout()
        self.calibrate_depth_button = QPushButton("Calibrate Depth")
        self.calibrate_depth_button.clicked.connect(self.calibrate_depth)
        self.depth_calibration_label = QLabel("Not Calibrated")
        depth_layout.addRow(self.calibrate_depth_button, self.depth_calibration_label)

        self.depth_sensitivity_slider = QSlider(Qt.Horizontal)
        self.depth_sensitivity_slider.setRange(0, 200)
        self.depth_sensitivity_slider.setValue(100)
        self.depth_sensitivity_slider.valueChanged.connect(self.update_depth_sensitivity)
        depth_layout.addRow("Depth Sensitivity:", self.depth_sensitivity_slider)

        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(50)
        self.smoothing_slider.valueChanged.connect(self.update_smoothing)
        depth_layout.addRow("Smoothing:", self.smoothing_slider)
        depth_group.setLayout(depth_layout)
        layout.addWidget(depth_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Calibration")

    def create_system_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Camera
        cam_group = QGroupBox("Camera")
        cam_layout = QVBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Camera {i}" for i in self.available_cameras])
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        cam_layout.addWidget(self.camera_combo)
        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)

        # Display
        disp_group = QGroupBox("Projector Display")
        disp_layout = QVBoxLayout()
        self.projector_combo = QComboBox()
        self.projector_combo.addItems([screen.name() or f"Screen {i+1}" for i, screen in enumerate(self.screens)])
        self.projector_combo.currentIndexChanged.connect(self.change_projector)
        disp_layout.addWidget(self.projector_combo)
        disp_group.setLayout(disp_layout)
        layout.addWidget(disp_group)

        # MIDI
        midi_group = QGroupBox("MIDI")
        midi_layout = QVBoxLayout()
        self.midi_combo = QComboBox()
        self.midi_ports = get_midi_ports()
        self.midi_combo.addItems(["None"] + self.midi_ports)
        self.midi_combo.currentIndexChanged.connect(self.change_midi_port)
        midi_layout.addWidget(self.midi_combo)
        self.midi_map_btn = QPushButton("Edit MIDI Mappings")
        self.midi_map_btn.clicked.connect(self.open_midi_mapping)
        midi_layout.addWidget(self.midi_map_btn)
        self.bpm_label = QLabel("BPM: 120.0")
        midi_layout.addWidget(self.bpm_label)

        self.help_button = QPushButton("Open MainStage Ethernet Setup Guide")
        self.help_button.clicked.connect(self.open_help)
        midi_layout.addWidget(self.help_button)

        midi_group.setLayout(midi_layout)
        layout.addWidget(midi_group)

        # Audio
        audio_group = QGroupBox("Audio Input")
        audio_layout = QVBoxLayout()
        self.audio_combo = QComboBox()
        from audio_handler import get_audio_devices
        devices = get_audio_devices()
        self.audio_devices = ["None"] + [d['name'] for d in devices if d['max_input_channels'] > 0]
        self.audio_combo.addItems(self.audio_devices)
        self.audio_combo.currentIndexChanged.connect(self.change_audio_device)
        audio_layout.addWidget(self.audio_combo)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        layout.addStretch()
        self.tabs.addTab(tab, "System")

    def create_diagnostics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.audio_monitor = AudioMonitor()
        layout.addWidget(QLabel("Audio Reactive Monitor:"))
        layout.addWidget(self.audio_monitor)

        self.diag_log = QListWidget()
        layout.addWidget(QLabel("MIDI / OSC Message Log:"))
        layout.addWidget(self.diag_log)

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.diag_log.clear)
        layout.addWidget(clear_btn)

        self.tabs.addTab(tab, "Connectivity")

    def log_message(self, msg):
        try:
            if not self.diag_log: return
            item = QTableWidgetItem(f"[{time.strftime('%H:%M:%S')}] {msg}")
            self.diag_log.insertItem(0, f"[{time.strftime('%H:%M:%S')}] {msg}")
            if self.diag_log.count() > 100:
                self.diag_log.takeItem(self.diag_log.count() - 1)
        except RuntimeError: pass

    def start_osc_server(self):
        self.osc_handler = OSCHandler()
        self.osc_handler.message_received.connect(self.handle_osc_message)
        self.osc_handler.start()

    def handle_osc_message(self, address, args):
        self.log_message(f"OSC: {address} {args}")
        parts = address.strip('/').split('/')
        if not parts: return

        if parts[0] == 'mask' and len(parts) >= 3:
            tag = parts[1]
            if tag == 'bg': tag = 'background' # Standardize
            action = parts[2]
            val = args[0] if args else 0

            if action == 'visible':
                self.worker.toggle_mask(tag, val > 0.5)
            elif action == 'fx' and len(parts) >= 4:
                fx_name = parts[3]
                self.worker.set_fx(tag, fx_name, val > 0.5)
            elif action == 'video' and args:
                self.worker.switch_video(tag, str(args[0]))
            elif action == 'cue' and args:
                self.worker.switch_cue(tag, int(args[0]))
        elif parts[0] == 'style' and args:
            self.worker.set_style(str(args[0]))
        elif parts[0] == 'snapshot' and args:
            self.load_snapshot(int(args[0]))

    def change_audio_device(self, index):
        if hasattr(self, 'audio_handler'):
            self.audio_handler.stop()

        # Update Audio devices list if needed
        from audio_handler import get_audio_devices
        devices = get_audio_devices()
        self.audio_devices = ["None"] + [d['name'] for d in devices if d['max_input_channels'] > 0]

        if index > 0:
            from audio_handler import AudioHandler, get_audio_devices
            # Subtract 1 because "None" is at index 0
            device_name = self.audio_devices[index]
            # Find the actual index for the device name
            devices = get_audio_devices()
            actual_index = None
            for i, d in enumerate(devices):
                if d['name'] == device_name and d['max_input_channels'] > 0:
                    actual_index = i
                    break

            self.audio_handler = AudioHandler(device_index=actual_index)
            self.audio_handler.bands_updated.connect(self.worker.set_audio_bands)
            self.audio_handler.bands_updated.connect(self.audio_monitor.set_levels)
            self.audio_handler.start()

    def open_midi_mapping(self):
        self.mapping_dialog = MIDIMappingDialog(self)
        self.mapping_dialog.show()

    def toggle_auto_ir(self, checked):
        self.worker.set_auto_threshold(checked)
        self.ir_threshold_slider.setEnabled(not checked)

    def update_smoothing(self, value):
        self.worker.set_smoothing(value / 100.0)

    def toggle_selected_cue(self):
        row = self.cue_table.currentRow()
        if row >= 0 and row < len(self.masks):
            mask = self.masks[row]
            if not mask.video_path:
                self.statusBar().showMessage("No video assigned to this mask.", 3000)
                return

            # Toggle visibility as a way to start/stop
            mask.visible = not mask.visible
            self.worker.set_masks(self.masks)

            if mask.visible:
                self.worker.restart_mask_video(mask.video_path)
                self.play_cue_btn.setText("■ STOP CUE")
                self.play_cue_btn.setStyleSheet("height: 50px; font-weight: bold; background-color: #ff5252; color: white; margin-top: 10px;")
            else:
                self.play_cue_btn.setText("▶ START CUE")
                self.play_cue_btn.setStyleSheet("height: 50px; font-weight: bold; background-color: #00c853; color: black; margin-top: 10px;")

            self.update_cue_table()
        else:
            self.statusBar().showMessage("Select a mask in the table first.", 3000)

    def change_warp_res(self, index):
        res_str = self.warp_res_combo.currentText()
        res = int(res_str.split('x')[0])
        self.projector_window.reset_warp_points(res)
        self.warp_group.setTitle(f"Projector Warping ({res_str} Grid)")

    def save_project(self, filename=None):
        if not filename:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Project", self.current_project_path or "", "Project Files (*.json)")

        if filename:
            import base64
            self.current_project_path = filename

            ref_frame_b64 = None
            if self.worker.latest_main_frame is not None:
                _, buffer = cv2.imencode('.jpg', self.worker.latest_main_frame)
                ref_frame_b64 = base64.b64encode(buffer).decode('utf-8')

            project_data = {
                'masks': [mask.to_dict() for mask in self.masks],
                'media_library': self.media_library,
                'warp_points': self.projector_window.get_warp_points_normalized(),
                'warp_grid_res': self.projector_window.grid_res,
                'ir_threshold': self.ir_threshold_slider.value(),
                'auto_ir': self.auto_ir_check.isChecked(),
                'depth_sensitivity': self.depth_sensitivity_slider.value() if hasattr(self, 'depth_sensitivity_slider') else 100,
                'smoothing': self.smoothing_slider.value(),
                'midi_port': self.midi_combo.currentText(),
                'midi_mappings': self.midi_mappings,
                'marker_config': self.worker.marker_config,
                'h_c2p': self.worker.h_c2p.tolist() if self.worker.h_c2p is not None else None,
                'calib_points': getattr(self.worker, 'calib_points', None),
                'calibration_camera_res': self.worker.calibration_camera_res,
                'baseline_distance': self.worker.baseline_distance,
                'projector_boundary': self.worker.projector_boundary,
                'pnp_enabled': self.pnp_check.isChecked(),
                'occlusion_enabled': self.occlusion_check.isChecked(),
                'setup_reference': ref_frame_b64,
                'master_visuals': {
                    'brightness': self.brightness_slider.value(),
                    'contrast': self.contrast_slider.value(),
                    'saturation': self.saturation_slider.value(),
                    'grain': self.grain_slider.value(),
                    'bloom': self.bloom_slider.value(),
                    'fader': self.master_fader_slider.value()
                }
            }
            with open(filename, 'w') as f:
                json.dump(project_data, f, indent=4)
            self.statusBar().showMessage(f"Project saved to {filename}", 3000)

    def maybe_auto_save(self):
        if hasattr(self, 'auto_save_check') and self.auto_save_check.isChecked() and self.current_project_path:
            self.save_project(self.current_project_path)

    def open_help(self):
        file_path = resource_path('HELP_MAINSTAGE.md')
        QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

    def show_setup_reference(self):
        if hasattr(self, 'setup_reference_b64') and self.setup_reference_b64:
            import base64
            import numpy as np
            img_data = base64.b64decode(self.setup_reference_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888).copy()

                # Show in a simple dialog
                dialog = QDialog(self)
                dialog.setWindowTitle("Setup Reference Frame")
                l = QVBoxLayout(dialog)
                lbl = QLabel()
                lbl.setPixmap(QPixmap.fromImage(qimg))
                l.addWidget(lbl)
                dialog.exec_()
        else:
            self.statusBar().showMessage("No setup reference frame found in this project.", 3000)

    def load_project(self, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Project Files (*.json)")

        if filename:
            self.current_project_path = filename
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                self.setup_reference_b64 = data.get('setup_reference')
                self.masks = [Mask.from_dict(d) for d in data.get('masks', [])]
                self.media_library = data.get('media_library', [])
                self.media_list.clear()
                for f in self.media_library:
                    self.media_list.addItem(f.split('/')[-1])

                self.update_cue_table()
                self.update_mask_combos()
                self.worker.set_masks(self.masks)

                warp_points = data.get('warp_points')
                warp_grid_res = data.get('warp_grid_res', 3)
                if warp_points:
                    self.warp_res_combo.setCurrentText(f"{warp_grid_res}x{warp_grid_res}")
                    self.projector_window.grid_res = warp_grid_res
                    self.projector_window.warp_points = [QPointF(p[0], p[1]) for p in warp_points]
                    self.worker.set_warp_points(warp_points, warp_grid_res)
                    self.warp_group.setTitle(f"Projector Warping ({warp_grid_res}x{warp_grid_res} Grid)")

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
                self.worker.projector_boundary = data.get('projector_boundary')
                self.pnp_check.setChecked(data.get('pnp_enabled', False))
                self.occlusion_check.setChecked(data.get('occlusion_enabled', False))

                h_c2p = data.get('h_c2p')
                calib_points = data.get('calib_points')
                calib_res = data.get('calibration_camera_res')
                if calib_points:
                    self.worker.set_h_c2p(calib_points, cam_res=calib_res)
                elif h_c2p:
                    self.worker.set_h_c2p(h_c2p, cam_res=calib_res)

                if calib_points or h_c2p:
                    if hasattr(self, 'setup_status_label') and self.setup_status_label:
                        self.setup_status_label.setText("Mapping: ALIGNED (Loaded) ✓")
                        self.setup_status_label.setStyleSheet("color: #00c853; font-weight: bold;")

                mv = data.get('master_visuals', {})
                self.brightness_slider.setValue(mv.get('brightness', 0))
                self.contrast_slider.setValue(mv.get('contrast', 0))
                self.saturation_slider.setValue(mv.get('saturation', 100))
                self.grain_slider.setValue(mv.get('grain', 0))
                self.bloom_slider.setValue(mv.get('bloom', 0))
                self.master_fader_slider.setValue(mv.get('fader', 100))

                self.statusBar().showMessage(f"Project loaded from {filename}", 3000)

                # If we are in the setup wizard, advance to the calibration step
                if self.setup_step == 0 and self.tabs.currentIndex() == 0:
                    self.next_setup_step()
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
            self.midi_handler.beat_pulse.connect(self.handle_beat_pulse)
            self.midi_handler.learned_message.connect(self.handle_learned_message)
            self.midi_handler.message_received.connect(self.log_message)
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

    def save_snapshot(self, index):
        state = {
            'masks': [mask.to_dict() for mask in self.masks],
            'style': self.style_combo.currentText(),
            'particles': self.particle_combo.currentText(),
            'audio_target': self.audio_target_combo.currentText()
        }
        self.snapshots[index] = state
        self.statusBar().showMessage(f"Snapshot {index+1} saved.", 3000)

    def load_snapshot(self, index):
        state = self.snapshots[index]
        if not state: return

        # Restore masks
        self.masks = [Mask.from_dict(d) for d in state['masks']]
        self.worker.set_masks(self.masks)

        # Restore UI combos
        self.style_combo.setCurrentText(state.get('style', 'none'))
        self.particle_combo.setCurrentText(state.get('particles', 'none'))
        self.audio_target_combo.setCurrentText(state.get('audio_target', 'none'))

        self.statusBar().showMessage(f"Snapshot {index+1} loaded.", 3000)
        self.send_midi_feedback_for_group('snap_load_', index, 8)

    def send_midi_feedback(self, key, value):
        if hasattr(self, 'midi_handler') and self.midi_handler:
            mapping = self.midi_mappings.get(key)
            if mapping:
                self.midi_handler.send_feedback(mapping[0], mapping[1], value)

    def send_midi_feedback_for_group(self, prefix, active_idx, count):
        for i in range(count):
            val = 127 if i == active_idx else 0
            self.send_midi_feedback(f"{prefix}{i}", val)

    def execute_midi_action(self, key, value):
        if key.startswith('cue_amp_'):
            idx = int(key.split('_')[-1])
            self.worker.switch_cue('amp', idx)
            self.send_midi_feedback_for_group('cue_amp_', idx, 8)
        elif key.startswith('cue_background_'):
            idx = int(key.split('_')[-1])
            self.worker.switch_cue('background', idx)
            self.send_midi_feedback_for_group('cue_background_', idx, 8)
        elif key == 'toggle_amp':
            self.worker.toggle_mask('amp', value > 0)
            self.send_midi_feedback('toggle_amp', 127 if value > 0 else 0)
        elif key == 'toggle_background':
            self.worker.toggle_mask('background', value > 0)
            self.send_midi_feedback('toggle_background', 127 if value > 0 else 0)
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
                        targets = ["none", "blur", "tint", "rgb_shift", "hue"]
                        curr = mask.fx_params.get('lfo_target', 'none')
                        idx = (targets.index(curr) + 1) % len(targets)
                        mask.fx_params['lfo_target'] = targets[idx]
                    elif param == 'shape' and value > 64:
                        shapes = ["sine", "square", "triangle", "sawtooth"]
                        curr = mask.fx_params.get('lfo_shape', 'sine')
                        idx = (shapes.index(curr) + 1) % len(shapes)
                        mask.fx_params['lfo_shape'] = shapes[idx]
        elif key.startswith('bezier_'):
            tag = key.split('_')[1]
            for mask in self.masks:
                if mask.tag == tag:
                    mask.bezier_enabled = (value > 64)
        elif key.startswith('style_'):
            style_name = key.split('_')[1]
            if value > 64:
                self.worker.set_style(style_name)
        elif key.startswith('part_'):
            part_name = key.split('_')[1]
            if value > 64:
                self.worker.set_particle_preset(part_name)
        elif key == 'auto_pilot_toggle':
            if value > 64:
                self.auto_pilot_check.setChecked(not self.auto_pilot_check.isChecked())
        elif key == 'hud_toggle':
            if value > 64:
                self.hud_check.setChecked(not self.hud_check.isChecked())
        elif key == 'safety_toggle':
            if value > 64:
                self.safety_check.setChecked(not self.safety_check.isChecked())
        elif key == 'toggle_projector_splash':
            if value > 64:
                self.splash_check.setChecked(not self.splash_check.isChecked())
        elif key == 'blackout_toggle':
            if value > 64:
                self.toggle_blackout()
        elif key.startswith('snap_save_'):
            idx = int(key.split('_')[-1])
            if value > 64: self.save_snapshot(idx)
        elif key.startswith('snap_load_'):
            idx = int(key.split('_')[-1])
            if value > 64: self.load_snapshot(idx)

    def handle_bpm(self, bpm):
        self.bpm_label.setText(f"BPM: {bpm:.1f}")
        self.worker.set_bpm(bpm)

    def handle_beat_pulse(self):
        self.worker.trigger_beat()

    def apply_mood(self):
        mood_name = self.mood_combo.currentText()
        colors = self.moods[mood_name]
        if not colors: return

        primary, secondary = colors
        for i, mask in enumerate(self.masks):
            # Alternate colors or just use primary for amp, secondary for bg
            if mask.tag == 'amp':
                mask.tint_color = primary
            else:
                mask.tint_color = secondary

        # Also set master tint
        self.worker.master_tint_color = primary
        self.statusBar().showMessage(f"Applied {mood_name} mood.", 3000)

    def toggle_blackout(self):
        # Instant blackout via worker flag
        self.worker.blackout_active = not self.worker.blackout_active
        if self.worker.blackout_active:
            self.blackout_btn.setText("DISABLE BLACKOUT")
            self.blackout_btn.setStyleSheet("background-color: #ff5252; color: white; font-weight: bold; height: 40px;")
            self.statusBar().showMessage("BLACKOUT ENABLED (PANIC)", 0)
        else:
            self.blackout_btn.setText("BLACKOUT (PANIC)")
            self.blackout_btn.setStyleSheet("background-color: #aa00ff; color: white; font-weight: bold; height: 40px;")
            self.statusBar().showMessage("BLACKOUT DISABLED", 3000)

    def toggle_splash_mode(self, checked):
        self.worker.show_splash = checked
        if checked:
            self.splash_check.setText("Stop Splash / START SHOW")
            self.splash_check.setStyleSheet("background-color: #d500f9; color: black; font-weight: bold;")
        else:
            self.splash_check.setText("Show Splash on Projector")
            self.splash_check.setStyleSheet("")

    def calibrate_depth(self):
        self.worker.calibrate_depth()
        self.depth_calibration_label.setText("Calibrated!")

    def update_depth_sensitivity(self, value):
        self.worker.set_depth_sensitivity(value / 100.0)

    def show_camera_error(self, index):
        self.statusBar().showMessage(f"Error: Could not open Camera {index}", 5000)

    def add_new_mask_prompt(self):
        mask_name, ok = QInputDialog.getText(self, "New Mask", "Enter Name for new mask:")
        if ok and mask_name:
            new_mask = Mask(mask_name, [], None)
            self.masks.append(new_mask)
            self.update_cue_table()
            self.cue_list_widget.setCurrentRow(len(self.masks)-1)
            self.update_mask_combos()
            self.enter_mask_creation_mode()

    def enter_mask_creation_mode(self):
        if self.cue_list_widget.currentRow() < 0:
            self.add_new_mask_prompt()
            return

        color = Qt.magenta
        row = self.cue_list_widget.currentRow()
        if 0 <= row < len(self.masks):
            mask = self.masks[row]
            if mask.tag == 'background': color = Qt.blue
            elif mask.tag == 'amp': color = Qt.green

        self.video_display.set_mask_creation_mode(True, color)
        self.create_mask_button.setEnabled(False)
        self.finish_mask_button.setEnabled(True)
        self.cancel_mask_button.setEnabled(True)

    def finish_mask_creation(self):
        mask_points = list(self.video_display.get_mask_points())
        self.video_display.set_mask_creation_mode(False)
        
        if not mask_points:
            self.cancel_mask_creation()
            return

        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            mask = self.masks[row]
        else:
            # Creation mode was entered without a selection, or selection was cleared
            new_name = f"Mask {len(self.masks) + 1}"
            mask = Mask(new_name, [], None)
            self.masks.append(mask)

        mask.source_points = [(p.x(), p.y()) for p in mask_points]

        # Update metadata from Workspace combos if they exist
        if hasattr(self, 'mask_tag_combo'):
            mask.tag = self.mask_tag_combo.currentText()
        if hasattr(self, 'mask_type_combo'):
            mask.type = self.mask_type_combo.currentText()
        if hasattr(self, 'mask_blend_combo'):
            mask.blend_mode = self.mask_blend_combo.currentText()
        if hasattr(self, 'bezier_check'):
            mask.bezier_enabled = self.bezier_check.isChecked()
        if hasattr(self, 'mask_feather_slider'):
            mask.feather = self.mask_feather_slider.value()

        # Specific handling for Setup Wizard tags
        if self.setup_step == 1: # Background step
            mask.tag = 'background'
            mask.type = 'static'
            mask.name = 'Background'
        elif self.setup_step == 3: # Guitar Mask step
            mask.tag = 'amp'
            mask.type = 'dynamic'
            mask.name = 'Guitar'
            # Automatically link to markers if they exist
            if self.selected_markers and not mask.is_linked:
                 self.link_mask_to_markers()
        elif self.setup_step == 4: # Amp Mask step
            mask.tag = 'background'
            mask.type = 'static'
            mask.name = 'Amp'

        self.update_cue_table()
        self.update_mask_combos()
        self.worker.set_masks(self.masks)
        self.statusBar().showMessage(f"Mask '{mask.name}' saved with {len(mask_points)} points.", 3000)
        self.maybe_auto_save()

    def play_selected_cue(self):
        row = self.cue_table.currentRow()
        if row >= 0 and row < len(self.masks):
            mask = self.masks[row]
            if mask.video_path:
                self.worker.restart_mask_video(mask.video_path)
                self.statusBar().showMessage(f"Restarted playback for '{mask.name}'", 2000)
            else:
                self.statusBar().showMessage("No video assigned to this mask.", 3000)
        else:
            self.statusBar().showMessage("Select a mask in the table first.", 3000)

    def open_playlist_editor(self):
        selected_mask_row = self.cue_table.currentRow()
        if selected_mask_row < 0:
            self.statusBar().showMessage("Select a mask from the table first.", 3000)
            return

        mask = self.masks[selected_mask_row]
        dialog = PlaylistEditorDialog(mask, self)
        if dialog.exec_():
            # If the current video path was removed or playlist changed
            if mask.playlist:
                if mask.playlist_index >= len(mask.playlist):
                    mask.playlist_index = 0
                mask.video_path = mask.playlist[mask.playlist_index]
            else:
                mask.video_path = None

            self.update_cue_table()
            self.worker.set_masks(self.masks)
            self.maybe_auto_save()
            self.statusBar().showMessage(f"Updated playlist for {mask.name}", 3000)
        else:
            self.statusBar().showMessage("Mask not saved: No points or no cue selected.", 5000)

        self.create_mask_button.setEnabled(True)
        self.finish_mask_button.setEnabled(False)
        self.cancel_mask_button.setEnabled(False)

    def cancel_mask_creation(self):
        self.video_display.set_mask_creation_mode(False)
        self.create_mask_button.setEnabled(True)
        self.finish_mask_button.setEnabled(False)
        self.cancel_mask_button.setEnabled(False)

    def add_mask_point_to_list(self, point):
        pass # Removed log list to save space

    def link_mask_to_markers(self):
        # Determine which mask to link based on dropdown or selection
        target_mask_name = None
        if self.tabs.currentIndex() == 0 and hasattr(self, 'setup_link_mask_combo'):
            target_mask_name = self.setup_link_mask_combo.currentText()
        elif hasattr(self, 'workspace_link_mask_combo'):
            target_mask_name = self.workspace_link_mask_combo.currentText()

        mask = None
        if target_mask_name:
            for m in self.masks:
                if m.name == target_mask_name:
                    mask = m
                    break

        if not mask:
            current_item = self.cue_list_widget.currentItem()
            if current_item:
                row = self.cue_list_widget.row(current_item)
                if 0 <= row < len(self.masks):
                    mask = self.masks[row]

        if not mask:
            self.statusBar().showMessage("Please select a mask to link.", 3000)
            return

        if not self.selected_markers:
            self.statusBar().showMessage("Please select IR markers first.", 3000)
            return

        if not mask.source_points:
            self.statusBar().showMessage("Please draw the mask points first.", 3000)
            return

        if mask.is_linked:
            self.statusBar().showMessage(f"Mask '{mask.name}' is already linked. Redraw it to change its shape.", 5000)
            return

        # Normalize points to reference frame so they stay locked to the guitar
        # regardless of where it was when "Link" was clicked.
        mask.source_points = self.worker.normalize_points_to_reference(mask.source_points)

        mask.is_linked = True

        # Update Status Labels
        status_text = "Status: LINKED"
        status_color = "#00c853"

        if getattr(self, 'setup_link_status_label', None):
            try:
                self.setup_link_status_label.setText(status_text)
                self.setup_link_status_label.setStyleSheet(f"font-weight: bold; color: {status_color};")
            except RuntimeError: pass

        if getattr(self, 'workspace_link_status_label', None):
            try:
                self.workspace_link_status_label.setText(status_text)
                self.workspace_link_status_label.setStyleSheet(f"font-weight: bold; color: {status_color};")
            except RuntimeError: pass

        self.update_cue_table() # Refresh table to show link status
        self.statusBar().showMessage(f"Mask '{mask.name}' linked to guitar tracking.", 3000)
        self.maybe_auto_save()

    def update_ir_threshold(self, value):
        self.worker.set_ir_threshold(value)

    def update_tracker_label(self, count):
        try:
            if self.ir_trackers_label:
                self.ir_trackers_label.setText(f"Trackers detected: {count}")
        except RuntimeError: pass

    def toggle_verify_alignment(self, checked):
        self.worker.show_calibration_verify = checked
        if checked:
            self.statusBar().showMessage("Verify: Projected circles should match camera dots.", 0)
        else:
            self.statusBar().showMessage("", 0)

    def toggle_test_pattern(self, checked):
        self.worker.show_calibration_pattern = checked

    def toggle_warping(self, checked):
        self.projector_window.set_calibration_mode(checked)
        self.worker.show_camera_on_projector = checked
        if checked:
            self.enable_warping_button.setText("Disable Warping")
        else:
            self.enable_warping_button.setText("Enable Warping")

    def add_vj_generator_to_lib(self):
        pattern, ok = QInputDialog.getItem(self, "Select Generator", "Pattern:",
                                         ["grid", "scan", "radial", "tunnel", "plasma", "vortex",
                                          "polytunnel", "stardust", "hypergrid", "prism_move",
                                          "nebula", "blackhole"], 0, False)
        if ok and pattern:
            path = f"generator:{pattern}"
            if path not in self.media_library:
                self.media_library.append(path)
                self.media_list.addItem(f"GEN: {pattern}")

    def change_camera(self, index):
        if self.available_cameras:
            new_camera_index = self.available_cameras[index]
            self.worker.set_video_source(new_camera_index)

    def change_projector(self, index):
        if index < len(self.screens):
            screen = self.screens[index]
            geom = screen.geometry()
            self.worker.projector_width = geom.width()
            self.worker.projector_height = geom.height()
            self.worker._warp_map_dirty = True

            self.projector_window.hide()
            self.projector_window.show() # Ensure window handle is created
            handle = self.projector_window.windowHandle()
            if handle:
                handle.setScreen(screen)
            self.projector_window.setGeometry(geom)
            self.projector_window.showFullScreen()

    def remove_cue(self):
        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            self.cue_list_widget.takeItem(row)
            del self.masks[row]
            self.update_cue_table()
            self.update_mask_combos()
            self.worker.set_masks(self.masks)
            self.maybe_auto_save()

    def mask_to_front(self):
        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            mask = self.masks[row]
            max_z = max((m.z_order for m in self.masks), default=0)
            mask.z_order = max_z + 1
            self.worker.set_masks(self.masks)
            self.statusBar().showMessage(f"'{mask.name}' brought to front", 3000)

    def mask_to_back(self):
        current_item = self.cue_list_widget.currentItem()
        if current_item:
            row = self.cue_list_widget.row(current_item)
            mask = self.masks[row]
            min_z = min((m.z_order for m in self.masks), default=0)
            mask.z_order = min_z - 1
            self.worker.set_masks(self.masks)
            self.statusBar().showMessage(f"'{mask.name}' sent to back", 3000)

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        if hasattr(self, 'midi_handler'):
            self.midi_handler.stop()
            if hasattr(self, 'midi_thread'):
                self.midi_thread.quit()
                self.midi_thread.wait()
        if hasattr(self, 'audio_handler'):
            self.audio_handler.stop()
        if hasattr(self, 'osc_handler'):
            self.osc_handler.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    try:
        with open(resource_path('style.qss'), 'r') as f:
            style = f.read()
        app.setStyleSheet(style)
    except FileNotFoundError:
        print("Stylesheet not found. Using default style.")

    main_win = ProjectionMappingApp()
    main_win.show()
    splash.finish(main_win)
    sys.exit(app.exec_())
