
import sounddevice as sd
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

class AudioHandler(QObject):
    bands_updated = pyqtSignal(float, float, float) # bass, mid, high

    def __init__(self, device_index=None, samplerate=44100):
        super().__init__()
        self.device_index = device_index
        self.samplerate = samplerate
        self.stream = None
        self.is_running = False

        # FFT Settings
        self.window_size = 1024
        self.buffer = np.zeros(self.window_size)

    def start(self):
        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.samplerate,
                callback=self.audio_callback,
                blocksize=self.window_size
            )
            self.stream.start()
            self.is_running = True
        except Exception as e:
            print(f"Audio Error: {e}")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        data = indata[:, 0]
        fft_data = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / self.samplerate)

        # Define bands
        bass = np.mean(fft_data[(freqs >= 20) & (freqs <= 150)]) if any((freqs >= 20) & (freqs <= 150)) else 0
        mid = np.mean(fft_data[(freqs > 150) & (freqs <= 2000)]) if any((freqs > 150) & (freqs <= 2000)) else 0
        high = np.mean(fft_data[(freqs > 2000)]) if any(freqs > 2000) else 0

        # Normalize roughly (depends on gain)
        self.bands_updated.emit(
            float(np.clip(bass * 10, 0, 1)),
            float(np.clip(mid * 20, 0, 1)),
            float(np.clip(high * 40, 0, 1))
        )

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False

def get_audio_devices():
    try:
        return sd.query_devices()
    except Exception:
        return []
