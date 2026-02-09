from PyQt5.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QPushButton, QWidget


class MarkerSelectionDialog(QDialog):
    marker_selected = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select IR Markers")
        self.setMinimumSize(800, 600)
        self.selected_points = []
        self.original_pixmap = None

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel("Press 'Take Picture' to begin.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.image_clicked
        self.take_picture_button = QPushButton("Take Picture")

        self.confirm_button = QPushButton("Confirm Markers")
        self.confirm_button.clicked.connect(self.accept)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.take_picture_button)
        self.layout.addWidget(self.confirm_button)

    def set_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self._render_preview()

    def _get_draw_rect(self):
        if not self.original_pixmap:
            return None

        label_size = self.image_label.size()
        pixmap_size = self.original_pixmap.size()
        if pixmap_size.width() == 0 or pixmap_size.height() == 0:
            return None

        scaled = pixmap_size.scaled(label_size, Qt.KeepAspectRatio)
        x = (label_size.width() - scaled.width()) // 2
        y = (label_size.height() - scaled.height()) // 2
        return x, y, scaled.width(), scaled.height()

    def _label_to_image(self, point):
        rect = self._get_draw_rect()
        if not rect or not self.original_pixmap:
            return None

        x, y, draw_w, draw_h = rect
        rel_x = point.x() - x
        rel_y = point.y() - y
        if rel_x < 0 or rel_y < 0 or rel_x > draw_w or rel_y > draw_h:
            return None

        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()
        img_x = int(rel_x * img_w / max(draw_w, 1))
        img_y = int(rel_y * img_h / max(draw_h, 1))
        return QPoint(img_x, img_y)

    def _render_preview(self):
        if not self.original_pixmap:
            return

        preview = self.original_pixmap.copy()
        painter = QPainter(preview)
        painter.setPen(QPen(Qt.green, 10))
        for point in self.selected_points:
            painter.drawPoint(point)
        painter.end()

        scaled_preview = preview.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_preview)

    def image_clicked(self, event):
        point = self._label_to_image(event.pos())
        if point is None:
            return
        self.selected_points.append(point)
        self.marker_selected.emit(point)
        self._render_preview()

    def get_selected_points(self):
        return self.selected_points

    def clear_selection(self):
        self.selected_points = []
        self._render_preview()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_preview()


class VideoDisplay(QWidget):
    mask_point_added = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.mask_creation_mode = False
        self.mask_points = []
        self.current_pixmap = None
        self._draw_rect = None

    def set_image(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update()

    def _widget_to_image_point(self, point):
        if not self.current_pixmap or not self._draw_rect:
            return None

        x, y, draw_w, draw_h = self._draw_rect
        rel_x = point.x() - x
        rel_y = point.y() - y
        if rel_x < 0 or rel_y < 0 or rel_x > draw_w or rel_y > draw_h:
            return None

        img_x = int(rel_x * self.current_pixmap.width() / max(draw_w, 1))
        img_y = int(rel_y * self.current_pixmap.height() / max(draw_h, 1))
        return QPoint(img_x, img_y)

    def _image_to_widget_point(self, point):
        if not self.current_pixmap or not self._draw_rect:
            return None

        x, y, draw_w, draw_h = self._draw_rect
        widget_x = x + point.x() * draw_w / max(self.current_pixmap.width(), 1)
        widget_y = y + point.y() * draw_h / max(self.current_pixmap.height(), 1)
        return QPointF(widget_x, widget_y)

    def paintEvent(self, event):
        if self.current_pixmap:
            painter = QPainter(self)
            scaled = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            self._draw_rect = (x, y, scaled.width(), scaled.height())
            painter.drawPixmap(x, y, scaled)

            if self.mask_creation_mode and self.mask_points:
                painter.setPen(QPen(Qt.green, 2))
                widget_points = [self._image_to_widget_point(p) for p in self.mask_points]
                widget_points = [p for p in widget_points if p is not None]
                if len(widget_points) >= 2:
                    painter.drawPolyline(QPolygonF(widget_points))
        else:
            self._draw_rect = None

    def mousePressEvent(self, event):
        if self.mask_creation_mode:
            point = self._widget_to_image_point(event.pos())
            if point is None:
                return
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
        self.layout = QVBoxLayout()
        self.label = QLabel()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.setStyleSheet("background-color: black;")

        self.calibration_mode = False
        self.warp_points = [QPointF(0.0, 0.0), QPointF(1.0, 0.0), QPointF(1.0, 1.0), QPointF(0.0, 1.0)]
        self.dragging_point_index = -1
        self.show()

    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def set_calibration_mode(self, enabled):
        self.calibration_mode = enabled
        self.update()

    def reset_warp_points(self):
        self.warp_points = [QPointF(0.0, 0.0), QPointF(1.0, 0.0), QPointF(1.0, 1.0), QPointF(0.0, 1.0)]
        self.warp_points_changed.emit(self.get_warp_points_normalized())
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.calibration_mode:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 10))

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
            if (pos - point).manhattanLength() < 20:
                return i
        return -1

    def normalize_point(self, pos):
        if self.width() <= 0 or self.height() <= 0:
            return QPointF(0.0, 0.0)
        x = min(max(pos.x() / self.width(), 0.0), 1.0)
        y = min(max(pos.y() / self.height(), 0.0), 1.0)
        return QPointF(float(x), float(y))

    def deserialize_warp_points(self, points):
        result = []
        for p in points:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                result.append(QPointF(float(p[0]), float(p[1])))
        if len(result) != 4:
            return [QPointF(0.0, 0.0), QPointF(1.0, 0.0), QPointF(1.0, 1.0), QPointF(0.0, 1.0)]
        return result

    def get_warp_points_normalized(self):
        return [[float(p.x()), float(p.y())] for p in self.warp_points]

    def resizeEvent(self, event):
        super().resizeEvent(event)
