import logging
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSplashScreen, QVBoxLayout

_ASSET_DIR = Path(__file__).resolve().parent
_log = logging.getLogger(__name__)

# Video-based splash intro is optional — the .mkv file is not shipped in all builds.
_LOGO_VIDEO = _ASSET_DIR / 'logo.mkv'
_LOGO_IMAGE = _ASSET_DIR / 'logo.png'

# Maximum frames the splash animation plays before force-closing.
# Prevents the splash from sticking if the video is unexpectedly long.
_MAX_SPLASH_FRAMES = 150  # ~5 s at 30 fps


class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)

        self.video_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.cap = None
        self._frame_count = 0
        self.timer = None

        # Only attempt video if the file exists — avoids confusing OpenCV error logs
        if _LOGO_VIDEO.exists():
            try:
                cap = cv2.VideoCapture(str(_LOGO_VIDEO))
                if cap.isOpened():
                    # Validate first frame before committing to video path
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self._display_frame(frame)
                        self._frame_count = 1
                        self.timer = QTimer(self)
                        self.timer.timeout.connect(self.update_frame)
                        self.timer.start(33)  # ~30 FPS
                        return
                    else:
                        _log.debug("Splash video opened but returned no first frame")
                        cap.release()
                else:
                    cap.release()
            except Exception as exc:
                _log.debug("Splash video init error: %s", exc)

        # Fallback: static logo image
        self._set_fallback_logo()

    def _set_fallback_logo(self):
        if _LOGO_IMAGE.exists():
            pixmap = QPixmap(str(_LOGO_IMAGE))
            if not pixmap.isNull():
                self.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                return
        # Last resort: blank white 200×200 pixmap
        fallback = QPixmap(200, 200)
        fallback.fill(Qt.white)
        self.setPixmap(fallback)

    def _display_frame(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qt_image = QImage(frame_rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as exc:
            _log.debug("Splash frame display error: %s", exc)

    def update_frame(self):
        if self.cap is None:
            if self.timer:
                self.timer.stop()
            return
        if self._frame_count >= _MAX_SPLASH_FRAMES:
            _log.debug("Splash max frames reached — closing")
            self._stop_video()
            return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self._display_frame(frame)
            self._frame_count += 1
        else:
            self._stop_video()

    def _stop_video(self):
        if self.timer:
            self.timer.stop()
            self.timer = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.close()

    def closeEvent(self, event):
        if self.timer:
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        event.accept()
