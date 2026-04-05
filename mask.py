import os
import logging
from typing import List, Optional

_log = logging.getLogger(__name__)

BLEND_MODES = ("normal", "additive", "multiply")
LOOP_MODES = ("loop", "oneshot")


def _clamp_float(value, lo: float, hi: float, default: float) -> float:
    """Safely coerce *value* to a float clamped to [lo, hi], returning *default* on error."""
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


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
        self.opacity: float = 1.0

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

    def set_opacity(self, value: float) -> None:
        """Set opacity, clamping to [0.0, 1.0]."""
        self.opacity = _clamp_float(value, 0.0, 1.0, 1.0)

    def set_fade_in(self, seconds: float) -> None:
        """Set fade-in duration in seconds (0 = instant cut)."""
        self.fade_in = _clamp_float(seconds, 0.0, 300.0, 0.0)

    def set_fade_out(self, seconds: float) -> None:
        """Set fade-out duration in seconds (0 = instant cut)."""
        self.fade_out = _clamp_float(seconds, 0.0, 300.0, 0.0)

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
        m.enabled = bool(d.get("enabled", True))
        m.opacity = _clamp_float(d.get("opacity", 1.0), 0.0, 1.0, 1.0)
        blend = d.get("blend_mode", "normal")
        m.blend_mode = blend if blend in BLEND_MODES else "normal"
        loop = d.get("loop_mode", "loop")
        m.loop_mode = loop if loop in LOOP_MODES else "loop"
        label_color = d.get("label_color", None)
        # Validate label_color is a hex string or None
        if label_color is not None and not (isinstance(label_color, str) and label_color.startswith('#')):
            label_color = None
        m.label_color = label_color
        m.fade_in = _clamp_float(d.get("fade_in", 0.0), 0.0, 300.0, 0.0)
        m.fade_out = _clamp_float(d.get("fade_out", 0.0), 0.0, 300.0, 0.0)
        # Validate active_cue is in bounds
        if not (0 <= m.active_cue < max(len(m.cues), 1)):
            m.active_cue = 0

        # Warn about missing cue files on load
        m.validate_cues()
        return m
