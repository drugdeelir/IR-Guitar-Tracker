from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QDialog
from PyQt5.QtCore import Qt, QPoint, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QPolygon


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
        self.image_label.setPixmap(self.original_pixmap)

    def image_clicked(self, event):
        point = event.pos()
        self.selected_points.append(point)
        self.marker_selected.emit(point)
        self.update()

    def get_selected_points(self):
        return self.selected_points

    def clear_selection(self):
        self.selected_points = []
        if self.original_pixmap:
            self.image_label.setPixmap(self.original_pixmap)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.original_pixmap:
            pixmap = self.original_pixmap.copy()
            painter = QPainter(pixmap)
            pen = QPen(Qt.green, 10)
            painter.setPen(pen)
            for point in self.selected_points:
                painter.drawPoint(point)
            painter.end()
            self.image_label.setPixmap(pixmap)


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

    def set_image(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event):
        if self.current_pixmap:
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self.current_pixmap)

            if self.mask_creation_mode and self.mask_points:
                painter.setPen(QPen(Qt.green, 2))
                poly = QPolygon(self.mask_points)
                painter.drawPolyline(poly)

    def mousePressEvent(self, event):
        if self.mask_creation_mode:
            point = event.pos()
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
        self.warp_points = [QPointF(0, 0), QPointF(1, 0), QPointF(1, 1), QPointF(0, 1)]
        self.dragging_point_index = -1
        self.show()

    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def set_calibration_mode(self, enabled):
        self.calibration_mode = enabled
        self.update()

    def set_warp_points(self, points):
        self.warp_points = [QPointF(p[0], p[1]) for p in points]
        self.warp_points_changed.emit(self.get_warp_points_normalized())
        self.update()

    def reset_warp_points(self):
        self.warp_points = [QPointF(0, 0), QPointF(1, 0), QPointF(1, 1), QPointF(0, 1)]
        self.warp_points_changed.emit(self.get_warp_points_normalized())
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.calibration_mode:
            painter = QPainter(self)
            pen = QPen(Qt.red, 10)
            painter.setPen(pen)

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
        return QPointF(pos.x() / max(self.width(), 1), pos.y() / max(self.height(), 1))

    def get_warp_points_normalized(self):
        return [[p.x(), p.y()] for p in self.warp_points]

    def resizeEvent(self, event):
        super().resizeEvent(event)
