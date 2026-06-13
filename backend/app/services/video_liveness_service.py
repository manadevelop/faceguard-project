from .antispoof_service import AntispoofService


# Servicio envoltorio para validación de vida usando múltiples frames.
class VideoLivenessService:
    def __init__(self):
        # Inicializa el servicio anti-spoofing usado para procesar frames.
        self.antispoof = AntispoofService()

    def verify_frames(self, frames, model_name=None):
        # Delega la verificación de frames al servicio anti-spoofing principal.
        return self.antispoof.predict_frames(frames, model_name)