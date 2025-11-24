
import cv2
from PyQt5.QtWidgets import QSplashScreen, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
        
        self.video_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)
        
        self.cap = cv2.VideoCapture('logo.mkv')
        if not self.cap.isOpened():
            print("Warning: Could not open logo.mkv. Attempting to use logo.png fallback.")
            pixmap = QPixmap('logo.png')
            if not pixmap.isNull():
                self.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return
            
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33) # ~30 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            self.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)) # Small logo
        else:
            self.timer.stop()
            self.cap.release()
            self.close()

    def closeEvent(self, event):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'cap'):
            self.cap.release()
        event.accept()
