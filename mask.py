import os
import logging

_log = logging.getLogger(__name__)

BLEND_MODES = ("normal", "additive", "multiply")
LOOP_MODES = ("loop", "oneshot")


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

        # Improvement 1: Enable/disable a mask without removing it
        self.enabled = True

        # Improvement 2: Per-mask opacity (0.0 = transparent, 1.0 = fully opaque)
        self.opacity = 1.0

        # Improvement 3: Blend mode for compositing against projector buffer
        self.blend_mode = "normal"  # "normal" | "additive" | "multiply"

        # Improvement 4: Video loop mode per mask
        self.loop_mode = "loop"  # "loop" | "oneshot"

        # Improvement 5: Render order for z-sorting (lower = rendered first)
        self.render_order = 0

        # Improvement 6: Locked flag — prevent accidental edits on stage
        self.locked = False

        # Improvement 7: Label colour for UI colour-coding (#RRGGBB or None)
        self.label_color = None

        # Improvement 8: Fade-in/fade-out duration in seconds (0 = instant cut)
        self.fade_in = 0.0
        self.fade_out = 0.0

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

    def advance_cue(self):
        """Advance to the next cue; wraps in loop mode, stops at end in oneshot."""
        if not self.cues:
            return
        if self.loop_mode == "loop":
            self.active_cue = (self.active_cue + 1) % len(self.cues)
        else:  # oneshot: clamp at last cue
            self.active_cue = min(self.active_cue + 1, len(self.cues) - 1)

    def get_active_video_path(self):
        if self.cues and 0 <= self.active_cue < len(self.cues):
            return self.cues[self.active_cue]
        return self.cues[0] if self.cues else None

    def validate_cues(self):
        """Return list of cue paths that no longer exist on disk."""
        missing = []
        for cue in self.cues:
            if cue and not os.path.exists(cue):
                missing.append(cue)
                _log.warning("Mask '%s': cue file not found: %s", self.name, cue)
        return missing

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
            # Improvement 9: persist marker_anchor_points (was missing before!)
            "marker_anchor_points": [list(p) if not hasattr(p, 'x') else [p.x(), p.y()]
                                     for p in self.marker_anchor_points],
            "cues": list(self.cues),
            "active_cue": self.active_cue,
            "midi_cc_map": {str(k): v for k, v in self.midi_cc_map.items()},
            "render_order": self.render_order,
            "locked": self.locked,
            # New fields
            "enabled": self.enabled,
            "opacity": float(self.opacity),
            "blend_mode": self.blend_mode,
            "loop_mode": self.loop_mode,
            "label_color": self.label_color,
            "fade_in": float(self.fade_in),
            "fade_out": float(self.fade_out),
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
        # Improvement 10: restore marker_anchor_points (was missing before!)
        m.marker_anchor_points = [tuple(p) for p in d.get("marker_anchor_points", [])]
        m.cues = cues
        m.active_cue = d.get("active_cue", 0)
        m.midi_cc_map = {int(k): v for k, v in d.get("midi_cc_map", {}).items()}
        m.render_order = d.get("render_order", 0)
        m.locked = d.get("locked", False)
        # New fields with safe defaults for backward compatibility
        m.enabled = d.get("enabled", True)
        m.opacity = float(d.get("opacity", 1.0))
        m.blend_mode = d.get("blend_mode", "normal")
        if m.blend_mode not in BLEND_MODES:
            m.blend_mode = "normal"
        m.loop_mode = d.get("loop_mode", "loop")
        if m.loop_mode not in LOOP_MODES:
            m.loop_mode = "loop"
        m.label_color = d.get("label_color", None)
        m.fade_in = float(d.get("fade_in", 0.0))
        m.fade_out = float(d.get("fade_out", 0.0))

        # Warn about missing cue files on load
        m.validate_cues()
        return m
