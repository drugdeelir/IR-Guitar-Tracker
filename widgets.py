from PyQt5.QtCore import QPoint, QPointF, QTimer, Qt, pyqtSignal
import cv2
import numpy as np
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QSizePolicy,
    QVBoxLayout, QPushButton, QWidget,
)


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
        self.take_picture_button.setToolTip("Capture a still frame from the live camera feed.")
        self.auto_select_button = QPushButton("Auto-Select Best 4")
        self.auto_select_button.setToolTip("Automatically pick the 4 brightest IR candidates.")
        self.auto_select_button.clicked.connect(self.auto_select_markers)

        # Improvement: marker count badge label
        self.count_label = QLabel("Markers selected: 0 / 4")

        btn_row = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm Markers")
        self.confirm_button.setToolTip("Accept the selected marker positions.")
        self.confirm_button.clicked.connect(self.accept)
        # Improvement: clear all button
        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.setToolTip("Remove all selected marker points.")
        self.clear_all_button.clicked.connect(self.clear_selection)
        btn_row.addWidget(self.confirm_button)
        btn_row.addWidget(self.clear_all_button)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.count_label)
        self.layout.addWidget(self.take_picture_button)
        self.layout.addWidget(self.auto_select_button)
        self.layout.addLayout(btn_row)

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
        min_area = max(40.0, frame_area * 0.00004)
        max_area = max(5000.0, frame_area * 0.08)
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

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 1.9:
                continue
            (_, _), radius = cv2.minEnclosingCircle(contour)
            if radius < 4.5:
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
            score = (
                peak * 2.5
                + mean_intensity * 1.1
                + circularity * 120.0
                + min(area, 5000.0) * 0.18
                + min(radius, 40.0) * 12.0
            )
            candidates.append((score, QPoint(cx, cy)))

        return self._nms_points(candidates, min_distance=34, limit=24)

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
        img_x = max(0, min(round(rel_x * img_w / max(draw_w, 1)), img_w - 1))
        img_y = max(0, min(round(rel_y * img_h / max(draw_h, 1)), img_h - 1))
        return QPoint(img_x, img_y)

    def _render_preview(self):
        if not self.original_pixmap:
            return

        preview = self.original_pixmap.copy()
        painter = QPainter(preview)

        # Draw detected IR candidates in red with small candidate numbers
        painter.setPen(QPen(Qt.red, 2))
        for i, point in enumerate(self.detected_ir_points, start=1):
            painter.drawEllipse(point, 8, 8)
            painter.setPen(QPen(Qt.yellow, 1))
            painter.drawText(point.x() + 10, point.y() - 2, str(i))
            painter.setPen(QPen(Qt.red, 2))

        # Draw selected markers in green with large numbered labels
        painter.setPen(QPen(Qt.green, 3))
        for i, point in enumerate(self.selected_points, start=1):
            painter.drawEllipse(point, 12, 12)
            painter.setPen(QPen(Qt.white, 1))
            painter.drawText(point.x() + 14, point.y() - 8, str(i))
            painter.setPen(QPen(Qt.green, 3))
        painter.end()

        scaled_preview = preview.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_preview)

        # Update count badge
        if hasattr(self, 'count_label'):
            n = len(self.selected_points)
            self.count_label.setText(f"Markers selected: {n} / {self.max_markers}")

    def image_clicked(self, event):
        # Improvement: right-click removes the last selected marker
        if event.button() == Qt.RightButton:
            if self.selected_points:
                self.selected_points.pop()
                self._render_preview()
            return

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


class VideoDisplay(QWidget):
    mask_point_added = pyqtSignal(QPoint)
    mask_point_removed = pyqtSignal()  # Improvement: right-click undo

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mask_creation_mode = False
        self.mask_points = []
        self.current_pixmap = None
        # Improvement: grid overlay for alignment
        self.grid_overlay_enabled = False
        self._grid_divisions = 10
        # Improvement: tracking state indicator dot
        self._tracking_state = "idle"  # "idle" | "tracking" | "lost" | "error"

    def set_grid_overlay(self, enabled: bool) -> None:
        self.grid_overlay_enabled = bool(enabled)
        self.update()

    def set_tracking_state(self, state: str) -> None:
        self._tracking_state = state
        self.update()

    def set_image(self, image):
        self.current_pixmap = QPixmap.fromImage(image)
        self.update()

    def _compute_draw_rect(self):
        if not self.current_pixmap:
            return None
        pixmap_size = self.current_pixmap.size()
        if pixmap_size.width() == 0 or pixmap_size.height() == 0:
            return None
        scaled = pixmap_size.scaled(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return (x, y, scaled.width(), scaled.height())

    def _widget_to_image_point(self, point):
        rect = self._compute_draw_rect()
        if not rect:
            return None

        x, y, draw_w, draw_h = rect
        rel_x = point.x() - x
        rel_y = point.y() - y
        if rel_x < 0 or rel_y < 0 or rel_x > draw_w or rel_y > draw_h:
            return None

        img_w = self.current_pixmap.width()
        img_h = self.current_pixmap.height()
        img_x = max(0, min(round(rel_x * img_w / max(draw_w, 1)), img_w - 1))
        img_y = max(0, min(round(rel_y * img_h / max(draw_h, 1)), img_h - 1))
        return QPoint(img_x, img_y)

    def _image_to_widget_point(self, point):
        rect = self._compute_draw_rect()
        if not rect:
            return None

        x, y, draw_w, draw_h = rect
        widget_x = x + point.x() * draw_w / max(self.current_pixmap.width(), 1)
        widget_y = y + point.y() * draw_h / max(self.current_pixmap.height(), 1)
        return QPointF(widget_x, widget_y)

    def paintEvent(self, event):
        rect = self._compute_draw_rect()
        if self.current_pixmap and rect:
            painter = QPainter(self)
            x, y, draw_w, draw_h = rect
            scaled = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(x, y, scaled)

            # Improvement: grid overlay
            if self.grid_overlay_enabled and draw_w > 0 and draw_h > 0:
                grid_pen = QPen(QColor(200, 200, 200, 80), 1)
                painter.setPen(grid_pen)
                n = self._grid_divisions
                for i in range(1, n):
                    gx = x + i * draw_w // n
                    gy = y + i * draw_h // n
                    painter.drawLine(gx, y, gx, y + draw_h)
                    painter.drawLine(x, gy, x + draw_w, gy)

            if self.mask_creation_mode and self.mask_points:
                painter.setPen(QPen(Qt.green, 2))
                widget_points = [self._image_to_widget_point(p) for p in self.mask_points]
                widget_points = [wp for wp in widget_points if wp is not None]
                if len(widget_points) >= 2:
                    painter.drawPolyline(QPolygonF(widget_points))
                # Close polygon when ≥3 points
                if len(widget_points) >= 3:
                    painter.setPen(QPen(QColor(0, 255, 0, 120), 1))
                    painter.drawLine(widget_points[-1], widget_points[0])

            # Improvement: tracking state dot in top-right corner of video area
            if rect:
                _state_colors = {
                    "tracking": QColor(0, 220, 80),
                    "lost":     QColor(255, 170, 0),
                    "error":    QColor(220, 40, 40),
                    "idle":     QColor(100, 100, 100),
                }
                dot_color = _state_colors.get(self._tracking_state, QColor(100, 100, 100))
                dot_r = 7
                dot_x = x + draw_w - dot_r - 6
                dot_y = y + dot_r + 6
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(dot_color))
                painter.drawEllipse(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2)

    def mousePressEvent(self, event):
        if self.mask_creation_mode:
            # Improvement: right-click removes last mask point
            if event.button() == Qt.RightButton:
                if self.mask_points:
                    self.mask_points.pop()
                    self.mask_point_removed.emit()
                    self.update()
                return
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
        self.layout.setContentsMargins(0, 0, 0, 0)
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
        # Improvement: blackout flag
        self._blackout = False
        self.show()

    def set_image(self, image):
        if self.pattern_mode or self._blackout:
            return
        pixmap = QPixmap.fromImage(image)
        size = self.label.size()
        if size.width() > 1 and size.height() > 1:
            pixmap = pixmap.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)

    def set_blackout(self, enabled: bool) -> None:
        """Improvement: instantly blackout projector window."""
        self._blackout = bool(enabled)
        if self._blackout:
            self.label.clear()
        self.repaint()

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
        self.calibration_mode = bool(enabled)
        # Improvement: change cursor in calibration mode for clear affordance
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def reset_warp_points(self):
        self.warp_points = [QPointF(0.0, 0.0), QPointF(1.0, 0.0), QPointF(1.0, 1.0), QPointF(0.0, 1.0)]
        self.warp_points_changed.emit(self.get_warp_points_normalized())
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.calibration_mode:
            painter = QPainter(self)
            w, h = self.width(), self.height()

            denorm = [
                QPoint(int(p.x() * w), int(p.y() * h))
                for p in self.warp_points
            ]

            # Improvement: draw semi-transparent warp quad
            if len(denorm) == 4:
                painter.setPen(QPen(QColor(255, 100, 0, 180), 1))
                painter.setBrush(QBrush(QColor(255, 100, 0, 30)))
                painter.drawPolygon(QPolygonF([QPointF(p) for p in denorm]))

            # Improvement: draw corner handles as circles + center dot
            corner_labels = ["TL", "TR", "BR", "BL"]
            for i, pt in enumerate(denorm):
                # Outer ring
                painter.setPen(QPen(QColor(255, 80, 0), 2))
                painter.setBrush(QBrush(QColor(255, 80, 0, 60)))
                painter.drawEllipse(pt, 14, 14)
                # Inner filled dot
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 200, 0)))
                painter.drawEllipse(pt, 5, 5)
                # Corner label
                painter.setPen(QPen(Qt.white, 1))
                painter.drawText(pt.x() + 16, pt.y() - 10, corner_labels[i])

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
        # Improvement: larger 25px hit radius for easier dragging
        denormalized_points = [
            QPoint(int(p.x() * self.width()), int(p.y() * self.height()))
            for p in self.warp_points
        ]
        for i, point in enumerate(denormalized_points):
            if (pos - point).manhattanLength() < 25:
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
        img_x = max(0, min(round(rel_x * img_w / max(draw_w, 1)), img_w - 1))
        img_y = max(0, min(round(rel_y * img_h / max(draw_h, 1)), img_h - 1))
        return QPoint(img_x, img_y)

    def image_clicked(self, event):
        # Improvement: right-click removes last point (undo)
        if event.button() == Qt.RightButton:
            self.undo_last_point()
            return
        point = self._label_to_image(event.pos())
        if point is None:
            return
        self.points.append(point)
        # Update title with point count
        self.setWindowTitle(f"{self.windowTitle().split(' (')[0]} ({len(self.points)} pts)")
        self._render_preview()

    def undo_last_point(self) -> None:
        """Improvement: remove the last added polygon point."""
        if self.points:
            self.points.pop()
            self.setWindowTitle(f"{self.windowTitle().split(' (')[0]} ({len(self.points)} pts)")
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

        # Improvement: semi-transparent green fill for closed polygon
        if len(self.points) >= 3:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 255, 0, 50)))
            painter.drawPolygon(QPolygonF([QPointF(p) for p in self.points]))

        # Draw polygon outline
        if len(self.points) >= 2:
            painter.setPen(QPen(Qt.green, 3))
            painter.drawPolyline(QPolygonF([QPointF(p) for p in self.points]))
        if len(self.points) >= 3:
            painter.setPen(QPen(Qt.green, 2))
            painter.drawLine(self.points[-1], self.points[0])

        # Improvement: numbered point markers
        for i, p in enumerate(self.points, start=1):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(Qt.yellow))
            painter.drawEllipse(p, 6, 6)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(p.x() + 8, p.y() - 4, str(i))

        painter.end()
        self.image_label.setPixmap(
            preview.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def accept(self):
        """Validate minimum point count before closing (#38)."""
        if len(self.points) < 3:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Not Enough Points",
                "A mask needs at least 3 points to form a valid polygon.\n"
                "Click on the image to add more points."
            )
            return  # keep dialog open
        super().accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_preview()
