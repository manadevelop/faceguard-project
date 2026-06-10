class FaceGuardError(Exception):
    """Base exception for controlled FaceGuard API errors."""


class InvalidImageError(FaceGuardError):
    """Raised when an uploaded image cannot be decoded or validated."""


class UnsupportedModelError(FaceGuardError):
    """Raised when a requested anti-spoofing model is not supported."""
