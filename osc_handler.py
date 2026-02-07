
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import threading
from PyQt5.QtCore import QObject, pyqtSignal

class OSCHandler(QObject):
    message_received = pyqtSignal(str, list)

    def __init__(self, ip="0.0.0.0", port=8000):
        super().__init__()
        self.ip = ip
        self.port = port
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/*", self.on_message)
        self.server = None
        self.thread = None

    def on_message(self, address, *args):
        self.message_received.emit(address, list(args))

    def start(self):
        try:
            self.server = BlockingOSCUDPServer((self.ip, self.port), self.dispatcher)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            print(f"OSC Server started on {self.ip}:{self.port}")
        except Exception as e:
            print(f"OSC Error: {e}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server = None
