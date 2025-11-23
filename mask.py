
class Mask:
    def __init__(self, name, points, video_path=None):
        self.name = name
        self.source_points = points
        self.video_path = video_path
        self.type = 'dynamic' # Default to dynamic for now
        self.tracker_ids = [0, 1, 2, 3] # Default to first 4 trackers
