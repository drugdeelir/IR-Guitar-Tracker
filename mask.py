
class Mask:
    def __init__(self, name, points, video_path=None):
        self.name = name
        self.source_points = points
        self.video_path = video_path
        self.type = 'dynamic' # Default to dynamic for now
        self.linked_marker_count = 0
