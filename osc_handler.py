from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import threading
from PyQt5.QtCore import QObject, pyqtSignal

class OscHandler(QObject):
    preset_requested = pyqtSignal(int)

    def __init__(self, ip="0.0.0.0", port=8000):
        super().__init__()
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/preset", self.handle_preset)

        try:
            self.server = BlockingOSCUDPServer((ip, port), self.dispatcher)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            print(f"OSC Server started on {ip}:{port}")
        except Exception as e:
            print(f"Error initializing OSC: {e}")

    def handle_preset(self, address, *args):
        if args:
            try:
                # Expecting 1-based index from OSC (e.g. /preset 1)
                preset_idx = int(args[0]) - 1
                if 0 <= preset_idx < 8:
                    self.preset_requested.emit(preset_idx)
            except (ValueError, TypeError):
                pass

    def stop(self):
        if hasattr(self, 'server'):
            self.server.shutdown()
