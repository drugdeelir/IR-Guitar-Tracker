
class Mask:
    def __init__(self, name, points, video_path=None, mask_type='dynamic', tag=None):
        self.name = name
        self.source_points = points
        self.video_path = video_path
        self.type = mask_type # 'dynamic' (tracked) or 'static' (background)
        self.linked_marker_count = 0
        self.tag = tag # e.g., 'amp', 'background'
        self.visible = True
        self.active_fx = [] # List of enabled FX names: 'strobe', 'blur', 'invert', 'edges', 'tint'
        self.tint_color = (255, 255, 255) # BGR

    def to_dict(self):
        return {
            'name': self.name,
            'source_points': self.source_points,
            'video_path': self.video_path,
            'type': self.type,
            'linked_marker_count': self.linked_marker_count,
            'tag': self.tag,
            'visible': self.visible,
            'active_fx': self.active_fx,
            'tint_color': list(self.tint_color)
        }

    @classmethod
    def from_dict(cls, d):
        mask = cls(d['name'], d['source_points'], d['video_path'], d['type'], d['tag'])
        mask.linked_marker_count = d.get('linked_marker_count', 0)
        mask.visible = d.get('visible', True)
        mask.active_fx = d.get('active_fx', [])
        mask.tint_color = tuple(d.get('tint_color', [255, 255, 255]))
        return mask
