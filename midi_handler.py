import mido
from PyQt5.QtCore import QObject, pyqtSignal

class MidiHandler(QObject):
    preset_requested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.inport = None
        try:
            port_names = mido.get_input_names()
            # Filter out potentially problematic ports or just take the first available
            available_ports = [n for n in port_names if 'RtMidi' not in n] # Simple filter
            if not available_ports and port_names:
                available_ports = port_names

            if available_ports:
                self.inport = mido.open_input(available_ports[0], callback=self.midi_callback)
                print(f"MIDI Input opened: {available_ports[0]}")
            else:
                print("No MIDI input ports found.")
        except Exception as e:
            print(f"Error initializing MIDI: {e}")

    def midi_callback(self, msg):
        if msg.type == 'note_on' and msg.velocity > 0:
            # Map notes to presets (e.g., notes 60-67 -> presets 0-7)
            if 60 <= msg.note <= 67:
                preset_idx = msg.note - 60
                self.preset_requested.emit(preset_idx)
        elif msg.type == 'control_change' and msg.value > 0:
            # Map CC to presets (e.g., CC 20-27 -> presets 0-7)
            if 20 <= msg.control <= 27:
                preset_idx = msg.control - 20
                self.preset_requested.emit(preset_idx)

    def close(self):
        if self.inport:
            self.inport.close()
