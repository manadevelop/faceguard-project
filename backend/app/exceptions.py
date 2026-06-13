# Define una excepción base para errores controlados dentro de FaceGuard.
class FaceGuardError(Exception):
    """Base exception for controlled FaceGuard API errors."""


# Representa errores cuando una imagen subida no puede decodificarse o validarse.
class InvalidImageError(FaceGuardError):
    """Raised when an uploaded image cannot be decoded or validated."""


# Representa errores cuando se solicita un modelo anti-spoofing no soportado.
class UnsupportedModelError(FaceGuardError):
    """Raised when a requested anti-spoofing model is not supported."""