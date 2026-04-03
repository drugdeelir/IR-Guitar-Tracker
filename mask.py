class Mask:
    def __init__(self, name, points, video_path=None):
        self.name = name
        self.source_points = points
        self.type = 'dynamic'
        self.linked_marker_count = 0
        self.marker_anchor_points = []

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
        return self.cues[0] if self.cues else None

    def to_dict(self):
        """Serialise to a JSON-safe dict for settings persistence."""
        # source_points may be QPoint objects or plain (x, y) tuples/lists
        def _pt(p):
            if hasattr(p, 'x'):
                return [p.x(), p.y()]
            return list(p)

        return {
            "name": self.name,
            "source_points": [_pt(p) for p in self.source_points],
            "type": self.type,
            "linked_marker_count": self.linked_marker_count,
            "cues": list(self.cues),
            "active_cue": self.active_cue,
            "midi_cc_map": {str(k): v for k, v in self.midi_cc_map.items()},
        }

    @classmethod
    def from_dict(cls, d):
        """Restore a Mask from a previously serialised dict."""
        pts = [tuple(p) for p in d.get("source_points", [])]
        cues = d.get("cues", [])
        video_path = cues[0] if cues else None
        m = cls(d.get("name", "Mask"), pts, video_path=video_path)
        m.type = d.get("type", "dynamic")
        m.linked_marker_count = d.get("linked_marker_count", 0)
        m.cues = cues
        m.active_cue = d.get("active_cue", 0)
        m.midi_cc_map = {int(k): v for k, v in d.get("midi_cc_map", {}).items()}
        return m
