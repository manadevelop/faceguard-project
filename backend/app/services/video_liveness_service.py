from .antispoof_service import AntispoofService

class VideoLivenessService:
    def __init__(self):
        self.antispoof = AntispoofService()

    def verify_frames(self, frames, model_name=None):
        return self.antispoof.predict_frames(frames, model_name)
