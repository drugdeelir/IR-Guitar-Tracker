
import mido
import time
from PyQt5.QtCore import QObject, pyqtSignal

class MIDIHandler(QObject):
    note_on = pyqtSignal(int, int) # note, velocity
    control_change = pyqtSignal(int, int) # cc, value
    beat = pyqtSignal(float) # bpm
    beat_pulse = pyqtSignal() # emitted once per quarter note
    learned_message = pyqtSignal(str, int) # type ('note' or 'cc'), number

    def __init__(self, port_name=None):
        super().__init__()
        self.port_name = port_name
        self.inport = None
        self.clock_count = 0
        self.last_clock_time = 0
        self.bpm = 120.0
        self.clock_times = []
        self.learning_mode = False

    def set_port(self, port_name):
        self.port_name = port_name

    def start_listening(self):
        if not self.port_name:
            print("No MIDI port selected.")
            return

        print(f"Opening MIDI port: {self.port_name}")
        try:
            self.inport = mido.open_input(self.port_name, callback=self.midi_callback)
        except Exception as e:
            print(f"MIDI Error: {e}")

    def midi_callback(self, msg):
        if self.learning_mode:
            if msg.type == 'note_on':
                self.learned_message.emit('note', msg.note)
            elif msg.type == 'control_change':
                self.learned_message.emit('cc', msg.control)

        if msg.type == 'note_on':
            if msg.velocity > 0:
                self.note_on.emit(msg.note, msg.velocity)
        elif msg.type == 'control_change':
            self.control_change.emit(msg.control, msg.value)
        elif msg.type == 'clock':
            self.handle_clock()

    def handle_clock(self):
        current_time = time.time()
        if self.last_clock_time > 0:
            delta = current_time - self.last_clock_time
            # Avoid jitters by averaging over a few clocks
            self.clock_times.append(delta)
            if len(self.clock_times) > 24: # Average over one quarter note
                self.clock_times.pop(0)

            if len(self.clock_times) == 24:
                avg_delta = sum(self.clock_times) / 24.0
                if avg_delta > 0:
                    new_bpm = 60.0 / (avg_delta * 24.0)
                    if abs(new_bpm - self.bpm) > 0.5: # Tolerance
                        self.bpm = new_bpm
                        self.beat.emit(self.bpm)

        self.last_clock_time = current_time
        self.clock_count = (self.clock_count + 1) % 24
        if self.clock_count == 0:
            self.beat_pulse.emit()

    def stop(self):
        if self.inport:
            self.inport.close()
            self.inport = None

def get_midi_ports():
    try:
        return mido.get_input_names()
    except Exception as e:
        print(f"Could not list MIDI ports: {e}")
        return []
