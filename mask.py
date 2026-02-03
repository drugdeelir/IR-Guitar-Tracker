
class Mask:
    def __init__(self, name, points, video_path=None, mask_type='dynamic', tag=None):
        self.name = name
        self.source_points = points
        self.video_path = video_path
        self.playlist = [video_path] if video_path else []
        self.playlist_index = 0
        self.type = mask_type # 'dynamic' (tracked) or 'static' (background)
        self.linked_marker_count = 0
        self.tag = tag # e.g., 'amp', 'background'
        self.visible = True
        self.is_linked = False
        self.active_fx = [] # 'strobe', 'blur', 'invert', 'edges', 'tint', 'kaleidoscope', 'mirror_h', 'mirror_v', 'rgb_shift', 'glitch', 'trails', 'hue_cycle', 'feedback'
        self.tint_color = (255, 255, 255) # BGR
        self.design_overlay = 'none' # 'none', 'spiral', 'moon', 'mushroom', 'star', 'hexagon', 'heart'
        self.blend_mode = 'normal' # 'normal', 'add', 'screen', 'multiply'
        self.bezier_enabled = False
        self.feather = 0 # 0 to 100
        self.z_order = 0
        self.video_bpm = 120.0
        self.fx_params = {
            'kaleidoscope_segments': 6,
            'lfo_enabled': False,
            'lfo_target': 'none', # 'blur', 'tint', 'rgb_shift', 'hue'
            'lfo_speed': 1.0 # multiplier of BPM
        }

    def to_dict(self):
        return {
            'name': self.name,
            'source_points': self.source_points,
            'video_path': self.video_path,
            'playlist': self.playlist,
            'playlist_index': self.playlist_index,
            'type': self.type,
            'is_linked': self.is_linked,
            'tag': self.tag,
            'visible': self.visible,
            'active_fx': self.active_fx,
            'tint_color': list(self.tint_color),
            'design_overlay': self.design_overlay,
            'blend_mode': self.blend_mode,
            'bezier_enabled': self.bezier_enabled,
            'feather': self.feather,
            'z_order': self.z_order,
            'video_bpm': self.video_bpm,
            'fx_params': self.fx_params
        }

    @classmethod
    def from_dict(cls, d):
        mask = cls(d['name'], d['source_points'], d['video_path'], d['type'], d['tag'])
        mask.playlist = d.get('playlist', [mask.video_path] if mask.video_path else [])
        mask.playlist_index = d.get('playlist_index', 0)
        mask.is_linked = d.get('is_linked', False)
        mask.visible = d.get('visible', True)
        mask.active_fx = d.get('active_fx', [])
        mask.tint_color = tuple(d.get('tint_color', [255, 255, 255]))
        mask.design_overlay = d.get('design_overlay', 'none')
        mask.blend_mode = d.get('blend_mode', 'normal')
        mask.bezier_enabled = d.get('bezier_enabled', False)
        mask.feather = d.get('feather', 0)
        mask.z_order = d.get('z_order', 0)
        mask.video_bpm = d.get('video_bpm', 120.0)
        mask.fx_params = d.get('fx_params', {'kaleidoscope_segments': 6})
        return mask
