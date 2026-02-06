
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QDialog, QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QPolygon, QPolygonF, QColor

class MarkerImageLabel(QLabel):
    point_selected = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.detected_points = []
        self.rejected_points = []
        self.selected_points = []
        self.guide_points = []
        self.hover_snap_pt = None
        self.pix = None

    def set_data(self, pixmap, detected_points, rejected_points=[], guide_points=None):
        self.pix = pixmap
        self.detected_points = detected_points
        self.rejected_points = rejected_points
        self.guide_points = guide_points or []
        self.selected_points = []
        self.hover_snap_pt = None
        self.setPixmap(self.pix)
        self.update()

    def mouseMoveEvent(self, event):
        if not self.pix: return
        px, py = self.map_pos_to_pixmap(event.pos())
        pix_w = self.pix.width()

        snap_threshold = max(30, int(pix_w * 0.04))
        best_pt = None
        min_dist = snap_threshold

        click_pt = QPoint(int(px), int(py))
        for dp in self.detected_points:
            dp_pt = QPoint(int(dp[0] * self.pix.width()), int(dp[1] * self.pix.height()))
            dist = (click_pt - dp_pt).manhattanLength()
            if dist < min_dist:
                min_dist = dist
                best_pt = dp_pt

        if best_pt != self.hover_snap_pt:
            self.hover_snap_pt = best_pt
            self.update()

    def map_pos_to_pixmap(self, pos):
        lbl_w, lbl_h = self.width(), self.height()
        pix_w, pix_h = self.pix.width(), self.pix.height()

        sw, sh = pix_w, pix_h
        aspect = pix_w / pix_h
        if sw > lbl_w:
            sw = lbl_w
            sh = int(sw / aspect)
        if sh > lbl_h:
            sh = lbl_h
            sw = int(sh * aspect)

        offset_x = (lbl_w - sw) // 2
        offset_y = (lbl_h - sh) // 2

        if sw == 0 or sh == 0: return 0, 0

        px = (pos.x() - offset_x) * pix_w / sw
        py = (pos.y() - offset_y) * pix_h / sh
        return px, py

    def mousePressEvent(self, event):
        if not self.pix: return
        px, py = self.map_pos_to_pixmap(event.pos())
        pix_w, pix_h = self.pix.width(), self.pix.height()

        if 0 <= px < pix_w and 0 <= py < pix_h:
            click_pt = QPoint(int(px), int(py))
            best_pt = click_pt

            # Snap logic
            snap_threshold = max(30, int(pix_w * 0.04))
            for dp in self.detected_points:
                dp_pt = QPoint(int(dp[0] * pix_w), int(dp[1] * pix_h))
                dist = (click_pt - dp_pt).manhattanLength()
                if dist < snap_threshold:
                    best_pt = dp_pt
                    break # Take the first one in range

            self.selected_points.append(best_pt)
            self.point_selected.emit(best_pt)
            self.update()

    def paintEvent(self, event):
        if not self.pix:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        lbl_w, lbl_h = self.width(), self.height()
        pix_w, pix_h = self.pix.width(), self.pix.height()

        sw, sh = pix_w, pix_h
        aspect = pix_w / pix_h
        if sw > lbl_w:
            sw = lbl_w
            sh = int(sw / aspect)
        if sh > lbl_h:
            sh = lbl_h
            sw = int(sh * aspect)

        offset_x = (lbl_w - sw) // 2
        offset_y = (lbl_h - sh) // 2

        painter.drawPixmap(offset_x, offset_y, sw, sh, self.pix)

        # Draw rejected dots (subtle)
        for rp in self.rejected_points:
            sx = offset_x + rp[0] * sw
            sy = offset_y + rp[1] * sh
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1))
            painter.setBrush(QBrush(QColor(100, 100, 100, 30)))
            painter.drawEllipse(QPointF(sx, sy), 5, 5)

        # Draw detected dots (more visible)
        for dp in self.detected_points:
            sx = offset_x + dp[0] * sw
            sy = offset_y + dp[1] * sh

            # Glowing effect for detected dots
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 255, 0, 100)))
            painter.drawEllipse(QPoint(int(sx), int(sy)), 15, 15)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(int(sx), int(sy)), 8, 8)

        # Draw hover snap indicator
        if self.hover_snap_pt:
            hx = offset_x + (self.hover_snap_pt.x() * sw / pix_w)
            hy = offset_y + (self.hover_snap_pt.y() * sh / pix_h)
            painter.setPen(QPen(Qt.yellow, 2))
            painter.drawEllipse(QPoint(int(hx), int(hy)), 20, 20)

        # Draw guide points
        painter.setPen(QPen(QColor(0, 255, 255, 150), 1, Qt.DotLine))
        for gp in self.guide_points:
            sx = offset_x + gp[0] * sw
            sy = offset_y + gp[1] * sh
            painter.drawEllipse(QPoint(int(sx), int(sy)), 12, 12)

        # Draw selected markers
        for i, sp in enumerate(self.selected_points):
            sx = offset_x + (sp.x() * sw / pix_w)
            sy = offset_y + (sp.y() * sh / pix_h)
            pt = QPoint(int(sx), int(sy))

            painter.setPen(QPen(Qt.magenta, 3))
            painter.drawLine(pt.x()-15, pt.y(), pt.x()+15, pt.y())
            painter.drawLine(pt.x(), pt.y()-15, pt.x(), pt.y()+15)
            painter.drawEllipse(pt, 6, 6)

            # Index number
            painter.setPen(Qt.white)
            painter.drawText(pt.x() + 10, pt.y() - 10, str(i + 1))

class MarkerSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select IR Markers")
        self.setMinimumSize(800, 600)

        self.layout = QVBoxLayout(self)
        self.image_label = MarkerImageLabel()
        self.take_picture_button = QPushButton("Take Picture")

        self.confirm_button = QPushButton("Confirm Markers")
        self.confirm_button.clicked.connect(self.accept)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.take_picture_button)
        self.layout.addWidget(self.confirm_button)

    def set_pixmap(self, pixmap, detected_points, rejected_points=[], guide_points=None):
        self.image_label.set_data(pixmap, detected_points, rejected_points, guide_points)

    def get_selected_points(self):
        return self.image_label.selected_points

    def clear_selection(self):
        self.image_label.selected_points = []
        self.image_label.update()

class MaskDrawingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Detailed Mask")
        self.setMinimumSize(1000, 800)

        self.layout = QVBoxLayout(self)
        self.video_display = VideoDisplay()
        self.video_display.set_mask_creation_mode(True)

        self.take_picture_button = QPushButton("Capture High-Res Frame")

        self.btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Points")
        self.clear_btn.clicked.connect(self.video_display.clear_mask_points)
        self.confirm_btn = QPushButton("Save & Finish")
        self.confirm_btn.clicked.connect(self.accept)
        self.confirm_btn.setStyleSheet("background-color: #00c853; color: white; font-weight: bold;")

        self.btn_layout.addWidget(self.clear_btn)
        self.btn_layout.addWidget(self.confirm_btn)

        self.layout.addWidget(self.video_display)
        self.layout.addWidget(self.take_picture_button)
        self.layout.addLayout(self.btn_layout)

    def set_image(self, qimage):
        self.video_display.set_image(qimage)

    def get_points(self):
        return self.video_display.get_mask_points()

    def set_points(self, points):
        self.video_display.set_mask_points(points)

class VideoDisplay(QOpenGLWidget):
    mask_point_added = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mask_creation_mode = False
        self.snap_to_markers = False # Default to False for better drawing experience
        self.mask_points = []
        self.detected_markers = []
        self.hover_pos = None
        self.current_pixmap = None
        self.current_mask_color = Qt.magenta
        self.dragging_idx = -1

    def set_mask_color(self, color):
        self.current_mask_color = color
        self.update()

    def set_detected_markers(self, points):
        self.detected_markers = points

    def set_image(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        pix_w, pix_h = 640, 480
        if self.current_pixmap:
            pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()
            # Scale to fit while maintaining aspect ratio (FastTransformation for performance)
            scaled_pixmap = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
            sw, sh = scaled_pixmap.width(), scaled_pixmap.height()
        else:
            # Fallback scaling for empty display
            sw, sh = self.width(), self.height()
            if sw * 480 > sh * 640:
                sw = sh * 640 // 480
            else:
                sh = sw * 480 // 640

        x = (self.width() - sw) // 2
        y = (self.height() - sh) // 2

        if self.current_pixmap:
            painter.drawPixmap(x, y, scaled_pixmap)

            # Draw detected IR markers as subtle hints
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.setBrush(QBrush(QColor(0, 255, 0, 40)))
            for marker in self.detected_markers:
                mx = x + marker[0] * sw
                my = y + marker[1] * sh
                painter.drawEllipse(QPointF(mx, my), 8, 8)

            # Draw snap preview
            if self.mask_creation_mode and self.snap_to_markers and self.hover_pos:
                px, py = self.map_to_pixmap(self.hover_pos)
                if 0 <= px < pix_w and 0 <= py < pix_h:
                    snap_radius = max(20, int(pix_w * 0.04))
                    best_m = None
                    min_d = snap_radius
                    for marker in self.detected_markers:
                        m_pt = QPoint(int(marker[0] * pix_w), int(marker[1] * pix_h))
                        dist = (QPoint(int(px), int(py)) - m_pt).manhattanLength()
                        if dist < min_d:
                            min_d = dist
                            best_m = marker

                    if best_m:
                        smx = x + best_m[0] * sw
                        smy = y + best_m[1] * sh
                        painter.setPen(QPen(Qt.yellow, 2, Qt.DotLine))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(QPointF(smx, smy), 15, 15)

        else:
            # Draw a dark gray rectangle to represent the camera FOV
            painter.fillRect(x, y, sw, sh, QColor(30, 30, 30))
            painter.setPen(QPen(Qt.gray, 1, Qt.DashLine))
            painter.drawRect(x, y, sw, sh)
            painter.drawText(x + 10, y + 20, "Waiting for Camera...")

        if self.mask_points:
            is_editing = self.mask_creation_mode
            pen_width = 3 if is_editing else 1
            alpha = 150 if is_editing else 80

            color = QColor(self.current_mask_color)
            color.setAlpha(alpha)

            painter.setPen(QPen(color, pen_width))
            painter.setBrush(QBrush(color, Qt.Dense6Pattern if is_editing else Qt.NoBrush))

            # Denormalize mask points for drawing
            draw_pts = []
            for p in self.mask_points:
                # p is normalized (0-1)
                dx = x + (p.x() * sw)
                dy = y + (p.y() * sh)
                draw_pts.append(QPoint(int(dx), int(dy)))

            if len(draw_pts) >= 2:
                poly = QPolygon(draw_pts)
                painter.drawPolygon(poly)

            # Draw handles only when editing
            if is_editing:
                painter.setBrush(Qt.white)
                painter.setPen(QPen(Qt.black, 1))
                for pt in draw_pts:
                    painter.drawEllipse(pt, 6, 6)
    
    def mousePressEvent(self, event):
        if self.mask_creation_mode:
            # Map point to pixmap coordinates
            if not self.current_pixmap: return
            pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()

            # Use a helper to map coordinates
            px, py = self.map_to_pixmap(event.pos())

            if 0 <= px < pix_w and 0 <= py < pix_h:
                click_pt = QPoint(int(px), int(py))

                # Check if we are clicking an existing point to drag
                # Reduced handle radius to allow closer points (1% of width)
                handle_radius = max(8, int(pix_w * 0.01))
                for i, p in enumerate(self.mask_points):
                    # Denormalize mask point to pixmap space for distance check
                    pt = QPoint(int(p.x() * pix_w), int(p.y() * pix_h))
                    if (click_pt - pt).manhattanLength() < handle_radius:
                        self.dragging_idx = i
                        return

                # Auto-Snapping to detected IR markers
                snapped_pt = click_pt
                if self.snap_to_markers:
                    # Snapping radius relative to width
                    min_dist = max(15, int(pix_w * 0.02))
                    for marker in self.detected_markers:
                        # marker is normalized
                        m_pt = QPoint(int(marker[0] * pix_w), int(marker[1] * pix_h))
                        dist = (click_pt - m_pt).manhattanLength()
                        if dist < min_dist:
                            min_dist = dist
                            snapped_pt = m_pt

                # Store normalized point
                norm_pt = QPointF(snapped_pt.x() / pix_w, snapped_pt.y() / pix_h)
                self.mask_points.append(norm_pt)
                self.mask_point_added.emit(norm_pt)
                self.update()

    def mouseMoveEvent(self, event):
        self.hover_pos = event.pos()
        if self.mask_creation_mode:
            if self.dragging_idx != -1:
                if not self.current_pixmap: return
                pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()
                px, py = self.map_to_pixmap(event.pos())
                self.mask_points[self.dragging_idx] = QPointF(px / pix_w, py / pix_h)
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_idx = -1

    def map_to_pixmap(self, pos):
        if not self.current_pixmap: return 0, 0
        lbl_w, lbl_h = self.width(), self.height()
        pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()
        scaled_size = self.current_pixmap.size()
        scaled_size.scale(self.size(), Qt.KeepAspectRatio)
        sw, sh = scaled_size.width(), scaled_size.height()
        offset_x = (lbl_w - sw) // 2
        offset_y = (lbl_h - sh) // 2
        px = (pos.x() - offset_x) * pix_w / sw
        py = (pos.y() - offset_y) * pix_h / sh
        return px, py

    def set_mask_creation_mode(self, enabled, color=Qt.magenta):
        self.mask_creation_mode = enabled
        self.current_mask_color = color
        if not enabled:
            self.clear_mask_points()
        self.update()

    def set_snap_to_markers(self, enabled):
        self.snap_to_markers = enabled

    def get_mask_points(self):
        return self.mask_points

    def set_mask_points(self, points):
        self.mask_points = points
        self.update()

    def clear_mask_points(self):
        self.mask_points = []
        self.update()

class AudioMonitor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.levels = [0, 0, 0] # Bass, Mid, High

    def set_levels(self, levels):
        self.levels = levels
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        w = self.width()
        h = self.height()
        bar_w = w // 3 - 10

        colors = [Qt.red, Qt.green, Qt.blue]
        labels = ["BASS", "MID", "HIGH"]

        for i in range(3):
            val = self.levels[i]
            bar_h = int(val * h)
            x = i * (w // 3) + 5

            # Draw bar
            painter.setBrush(QBrush(colors[i]))
            painter.drawRect(x, h - bar_h, bar_w, bar_h)

            # Label
            painter.setPen(Qt.white)
            painter.drawText(x, h - 5, labels[i])

class ProjectorWindow(QOpenGLWidget):
    warp_points_changed = pyqtSignal(list, int) # points, resolution

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projector Output")
        # Ensure window is frameless and always on top for reliable 1:1 mapping
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setStyleSheet("background-color: black;")
        self.current_pixmap = None

        self.calibration_mode = False
        self.grid_res = 3
        self.warp_points = []
        self.reset_warp_points()

        self.dragging_point_index = -1

    def set_image(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update()

    def set_calibration_mode(self, enabled):
        self.calibration_mode = enabled
        self.update() # Trigger a repaint

    def reset_warp_points(self, res=None):
        if res: self.grid_res = res
        self.warp_points = []
        for i in range(self.grid_res):
            y = i / (self.grid_res - 1)
            for j in range(self.grid_res):
                x = j / (self.grid_res - 1)
                self.warp_points.append(QPointF(x, y))
        self.warp_points_changed.emit(self.get_warp_points_normalized(), self.grid_res)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        # Ensure we fill the physical rect of the screen
        painter.fillRect(self.rect(), Qt.black)

        if self.current_pixmap:
            # Optimization: Smooth transformation for high-res output
            painter.drawPixmap(self.rect(), self.current_pixmap)

        if self.calibration_mode:
            pen = QPen(Qt.red, 10)
            painter.setPen(pen)
            
            # Denormalize points for drawing
            denormalized_points = [
                QPoint(int(p.x() * self.width()), int(p.y() * self.height()))
                for p in self.warp_points
            ]
            
            for point in denormalized_points:
                painter.drawPoint(point)

    def mousePressEvent(self, event):
        if self.calibration_mode:
            self.dragging_point_index = self.get_point_at(event.pos())

    def mouseMoveEvent(self, event):
        if self.calibration_mode and self.dragging_point_index != -1:
            self.warp_points[self.dragging_point_index] = self.normalize_point(event.pos())
            self.warp_points_changed.emit(self.get_warp_points_normalized(), self.grid_res)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.calibration_mode:
            self.dragging_point_index = -1
    
    def get_point_at(self, pos):
        denormalized_points = [
            QPoint(int(p.x() * self.width()), int(p.y() * self.height()))
            for p in self.warp_points
        ]
        for i, point in enumerate(denormalized_points):
            if (pos - point).manhattanLength() < 20: # Click tolerance
                return i
        return -1

    def normalize_point(self, pos):
        return QPointF(pos.x() / self.width(), pos.y() / self.height())

    def get_warp_points_normalized(self):
        return [[p.x(), p.y()] for p in self.warp_points]

    def resizeEvent(self, event):
        super().resizeEvent(event)
