
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QDialog
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QPolygon, QPolygonF

class MarkerImageLabel(QLabel):
    point_selected = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.detected_points = []
        self.selected_points = []
        self.guide_points = []
        self.pix = None

    def set_data(self, pixmap, detected_points, guide_points=None):
        self.pix = pixmap
        self.detected_points = detected_points
        self.guide_points = guide_points or []
        self.selected_points = []
        self.setPixmap(self.pix)
        self.update()

    def mousePressEvent(self, event):
        if not self.pix: return
        pos = event.pos()

        # Map to pixmap coordinates
        lbl_w, lbl_h = self.width(), self.height()
        pix_w, pix_h = self.pix.width(), self.pix.height()

        offset_x = (lbl_w - pix_w) // 2
        offset_y = (lbl_h - pix_h) // 2

        px = pos.x() - offset_x
        py = pos.y() - offset_y

        if 0 <= px < pix_w and 0 <= py < pix_h:
            click_pt = QPoint(px, py)
            best_pt = click_pt
            min_dist = 30
            for dp in self.detected_points:
                dp_pt = QPoint(int(dp[0]), int(dp[1]))
                dist = (click_pt - dp_pt).manhattanLength()
                if dist < min_dist:
                    min_dist = dist
                    best_pt = dp_pt

            self.selected_points.append(best_pt)
            self.point_selected.emit(best_pt)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pix: return

        painter = QPainter(self)
        lbl_w, lbl_h = self.width(), self.height()
        pix_w, pix_h = self.pix.width(), self.pix.height()
        offset_x = (lbl_w - pix_w) // 2
        offset_y = (lbl_h - pix_h) // 2

        painter.translate(offset_x, offset_y)

        # Draw detected points as faint red circles
        painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
        for dp in self.detected_points:
            painter.drawEllipse(QPoint(int(dp[0]), int(dp[1])), 12, 12)

        # Draw guide points (from template) as faint cyan circles
        painter.setPen(QPen(Qt.cyan, 2, Qt.DotLine))
        for gp in self.guide_points:
            painter.drawEllipse(QPoint(int(gp[0]), int(gp[1])), 15, 15)

        # Draw selected points as solid green targets
        painter.setPen(QPen(Qt.green, 3))
        for sp in self.selected_points:
            painter.drawLine(sp.x()-15, sp.y(), sp.x()+15, sp.y())
            painter.drawLine(sp.x(), sp.y()-15, sp.x(), sp.y()+15)
            painter.drawEllipse(sp, 5, 5)

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

    def set_pixmap(self, pixmap, detected_points, guide_points=None):
        self.image_label.set_data(pixmap, detected_points, guide_points)

    def get_selected_points(self):
        return self.image_label.selected_points

    def clear_selection(self):
        self.image_label.selected_points = []
        self.image_label.update()

class VideoDisplay(QWidget):
    mask_point_added = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mask_creation_mode = False
        self.mask_points = []
        self.current_pixmap = None

    def set_image(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if self.current_pixmap:
            # Scale to fit while maintaining aspect ratio
            scaled_pixmap = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Center the pixmap
            sw, sh = scaled_pixmap.width(), scaled_pixmap.height()
            x = (self.width() - sw) // 2
            y = (self.height() - sh) // 2
            self.draw_rect = scaled_pixmap.rect().translated(x, y)
            painter.drawPixmap(x, y, scaled_pixmap)

            if self.mask_creation_mode and self.mask_points:
                painter.setPen(QPen(Qt.green, 2))
                painter.setBrush(QBrush(Qt.green, Qt.Dense6Pattern))

                # Denormalize mask points for drawing
                pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()
                draw_pts = []
                for p in self.mask_points:
                    dx = x + (p.x() * sw / pix_w)
                    dy = y + (p.y() * sh / pix_h)
                    draw_pts.append(QPoint(int(dx), int(dy)))

                poly = QPolygon(draw_pts)
                painter.drawPolygon(poly)
    
    def mousePressEvent(self, event):
        if self.mask_creation_mode:
            # Map point to pixmap coordinates
            if not self.current_pixmap: return

            lbl_w, lbl_h = self.width(), self.height()
            pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()

            # Re-calculate scaling used in paintEvent
            scaled_size = self.current_pixmap.size()
            scaled_size.scale(self.size(), Qt.KeepAspectRatio)

            sw = scaled_size.width()
            sh = scaled_size.height()

            offset_x = (lbl_w - sw) // 2
            offset_y = (lbl_h - sh) // 2

            px = (event.pos().x() - offset_x) * pix_w / sw
            py = (event.pos().y() - offset_y) * pix_h / sh

            if 0 <= px < pix_w and 0 <= py < pix_h:
                point = QPoint(int(px), int(py))
                self.mask_points.append(point)
                self.mask_point_added.emit(point)
                self.update()

    def set_mask_creation_mode(self, enabled):
        self.mask_creation_mode = enabled
        if not enabled:
            self.clear_mask_points()
        self.update()

    def get_mask_points(self):
        return self.mask_points

    def clear_mask_points(self):
        self.mask_points = []
        self.update()

class ProjectorWindow(QWidget):
    warp_points_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projector Output")
        self.setStyleSheet("background-color: black;")
        self.current_pixmap = None

        self.calibration_mode = False
        # 3x3 Grid
        self.warp_points = []
        for y in [0.0, 0.5, 1.0]:
            for x in [0.0, 0.5, 1.0]:
                self.warp_points.append(QPointF(x, y))

        self.dragging_point_index = -1

    def set_image(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update()

    def set_calibration_mode(self, enabled):
        self.calibration_mode = enabled
        self.update() # Trigger a repaint

    def reset_warp_points(self):
        self.warp_points = []
        for y in [0.0, 0.5, 1.0]:
            for x in [0.0, 0.5, 1.0]:
                self.warp_points.append(QPointF(x, y))
        self.warp_points_changed.emit(self.get_warp_points_normalized())
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if self.current_pixmap:
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
            self.warp_points_changed.emit(self.get_warp_points_normalized())
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
