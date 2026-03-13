from PyQt5.QtCore import QPoint, QPointF, Qt, pyqtSignal
import cv2
import numpy as np
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QPushButton, QWidget


class MarkerSelectionDialog(QDialog):
    marker_selected = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select IR Markers")
        self.setMinimumSize(800, 600)
        self.selected_points = []
        self.original_pixmap = None
        self.detected_ir_points = []
        self.max_markers = 4
        self.ir_assist_enabled = True

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel("Press 'Take Picture' to begin.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.image_clicked
        self.take_picture_button = QPushButton("Take Picture")
        self.auto_select_button = QPushButton("Auto-Select Best 4")
        self.auto_select_button.clicked.connect(self.auto_select_markers)

        self.confirm_button = QPushButton("Confirm Markers")
        self.confirm_button.clicked.connect(self.accept)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.take_picture_button)
        self.layout.addWidget(self.auto_select_button)
        self.layout.addWidget(self.confirm_button)

    def set_ir_assist_enabled(self, enabled):
        self.ir_assist_enabled = bool(enabled)
        self.auto_select_button.setEnabled(self.ir_assist_enabled)
        if self.original_pixmap:
            self.detected_ir_points = (
                self._detect_ir_points(self.original_pixmap) if self.ir_assist_enabled else []
            )
            self._render_preview()

    def set_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.detected_ir_points = self._detect_ir_points(pixmap) if self.ir_assist_enabled else []
        self._render_preview()

    def _nms_points(self, scored_points, min_distance=28, limit=24):
        selected = []
        for score, point in sorted(scored_points, key=lambda item: item[0], reverse=True):
            if all((point.x() - p.x()) ** 2 + (point.y() - p.y()) ** 2 >= min_distance ** 2 for _, p in selected):
                selected.append((score, point))
            if len(selected) >= limit:
                break
        return [pt for _, pt in selected]

    def _detect_ir_points(self, pixmap):
        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        w, h = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        percentile_threshold = int(np.percentile(enhanced, 99.6))
        _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, pct_thresh = cv2.threshold(enhanced, percentile_threshold, 255, cv2.THRESH_BINARY)
        adaptive = cv2.bitwise_or(otsu_thresh, pct_thresh)

        _, bright = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_or(adaptive, bright)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        frame_area = float(enhanced.shape[0] * enhanced.shape[1])
        min_area = max(3.0, frame_area * 0.000005)
        max_area = max(1200.0, frame_area * 0.08)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.2:
                continue

            contour_mask = np.zeros(enhanced.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            peak = float(cv2.minMaxLoc(enhanced, mask=contour_mask)[1])
            mean_intensity = float(cv2.mean(enhanced, mask=contour_mask)[0])
            if peak < 145 and mean_intensity < 90:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            score = peak * 2.8 + mean_intensity * 1.0 + circularity * 90.0 + min(area, 2500.0) * 0.05
            candidates.append((score, QPoint(cx, cy)))

        return self._nms_points(candidates, min_distance=26, limit=24)

    def _snap_to_ir_point(self, point, max_distance=40):
        if not self.detected_ir_points:
            return point

        nearest = min(
            self.detected_ir_points,
            key=lambda candidate: (candidate.x() - point.x()) ** 2 + (candidate.y() - point.y()) ** 2,
        )
        distance = ((nearest.x() - point.x()) ** 2 + (nearest.y() - point.y()) ** 2) ** 0.5
        if distance <= max_distance:
            return nearest
        return point

    def auto_select_markers(self):
        if not self.ir_assist_enabled:
            return
        self.selected_points = [QPoint(p.x(), p.y()) for p in self.detected_ir_points[: self.max_markers]]
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
        painter.setPen(QPen(Qt.red, 5))
        for point in self.detected_ir_points:
            painter.drawEllipse(point, 4, 4)
        painter.setPen(QPen(Qt.green, 10))
        for i, point in enumerate(self.selected_points, start=1):
            painter.drawPoint(point)
            painter.drawText(point.x() + 6, point.y() - 6, str(i))
        painter.end()

        scaled_preview = preview.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_preview)

    def image_clicked(self, event):
        if len(self.selected_points) >= self.max_markers:
            return
        point = self._label_to_image(event.pos())
        if point is None:
            return
        if self.ir_assist_enabled:
            point = self._snap_to_ir_point(point)

        for existing in self.selected_points:
            distance = ((existing.x() - point.x()) ** 2 + (existing.y() - point.y()) ** 2) ** 0.5
            if distance < 12:
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
        self._render_preview()


class VideoDisplay(QWidget):
    mask_point_added = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.setStyleSheet("background-color: black;")

        self.calibration_mode = False
        self.pattern_mode = False
        self.pattern_brightness = 255
        self.pattern_margin_ratio = 0.08
        self.warp_points = [QPointF(0.0, 0.0), QPointF(1.0, 0.0), QPointF(1.0, 1.0), QPointF(0.0, 1.0)]
        self.dragging_point_index = -1
        self.show()

    def set_image(self, image):
        if self.pattern_mode:
            return
        pixmap = QPixmap.fromImage(image)
        size = self.label.size()
        if size.width() > 1 and size.height() > 1:
            pixmap = pixmap.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)

    def set_pattern_mode(self, enabled, brightness=255):
        self.pattern_mode = bool(enabled)
        self.pattern_brightness = int(max(1, min(255, brightness)))
        if self.pattern_mode:
            self.render_calibration_pattern()
        else:
            self.label.clear()
        self.raise_()
        self.activateWindow()
        self.repaint()

    def render_calibration_pattern(self):
        size = self.label.size()
        w = max(2, size.width())
        h = max(2, size.height())

        image = QImage(w, h, QImage.Format_RGB888)
        image.fill(Qt.black)

        painter = QPainter(image)
        margin_x = int(w * self.pattern_margin_ratio)
        margin_y = int(h * self.pattern_margin_ratio)
        rect_w = max(2, w - 2 * margin_x)
        rect_h = max(2, h - 2 * margin_y)

        fill = QColor(self.pattern_brightness, self.pattern_brightness, self.pattern_brightness)
        painter.setPen(QPen(Qt.white, 5))
        painter.setBrush(QBrush(fill))
        painter.drawRect(margin_x, margin_y, rect_w, rect_h)
        painter.end()

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
        if self.pattern_mode:
            self.render_calibration_pattern()


class PolygonMaskDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 650)
        self.original_pixmap = None
        self.points = []

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel("Capture a still frame first.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.image_clicked
        self.layout.addWidget(self.image_label)

        buttons_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)
        self.confirm_button = QPushButton("Confirm Mask")
        self.confirm_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.confirm_button)
        self.layout.addLayout(buttons_layout)

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

    def image_clicked(self, event):
        point = self._label_to_image(event.pos())
        if point is None:
            return
        self.points.append(point)
        self._render_preview()

    def clear_points(self):
        self.points = []
        self._render_preview()

    def set_points(self, points):
        self.points = [QPoint(int(p.x()), int(p.y())) for p in points]
        self._render_preview()

    def get_points(self):
        return self.points

    def _render_preview(self):
        if not self.original_pixmap:
            return
        preview = self.original_pixmap.copy()
        painter = QPainter(preview)
        painter.setPen(QPen(Qt.yellow, 8))
        for p in self.points:
            painter.drawPoint(p)
        if len(self.points) >= 2:
            painter.setPen(QPen(Qt.green, 3))
            painter.drawPolyline(QPolygonF([QPointF(p) for p in self.points]))
        if len(self.points) >= 3:
            painter.setPen(QPen(Qt.green, 3))
            painter.drawLine(self.points[-1], self.points[0])
        painter.end()
        self.image_label.setPixmap(preview.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_preview()
