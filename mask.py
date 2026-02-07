
class Mask:
    def __init__(self, name, points, video_path=None, cue_type='video', generator_type=None):
        self.name = name
        self.source_points = points
        self.video_path = video_path
        self.cue_type = cue_type
        self.generator_type = generator_type
        self.type = 'dynamic' # Default to dynamic for now
        self.linked_marker_count = 0
