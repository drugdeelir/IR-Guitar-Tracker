class Mask:
    def __init__(self, name, points, video_path=None):
        self.name = name
        self.source_points = points
        self.video_path = video_path
        self.type = 'dynamic'
        self.linked_marker_count = 0

        # New cue model: each mask has its own queue.
        self.cues = []
        if video_path:
            self.cues.append(video_path)
        self.active_cue = 0
        self.midi_cc_map = {}  # cc -> cue index

    def add_cue(self, path):
        if path:
            self.cues.append(path)
            if len(self.cues) == 1:
                self.active_cue = 0

    def remove_cue(self, index):
        if 0 <= index < len(self.cues):
            del self.cues[index]
            self.active_cue = max(0, min(self.active_cue, len(self.cues) - 1))
            for cc, cue_index in list(self.midi_cc_map.items()):
                if cue_index == index:
                    del self.midi_cc_map[cc]
                elif cue_index > index:
                    self.midi_cc_map[cc] = cue_index - 1

    def get_active_video_path(self):
        if self.cues and 0 <= self.active_cue < len(self.cues):
            return self.cues[self.active_cue]
        return self.video_path
