
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
