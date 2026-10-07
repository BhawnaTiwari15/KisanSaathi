"""Vision analysis package.

Provides protocols and implementations for image analysis with explicit uncertainty.
"""

from kisansathi.vision.models import (
    VisionResult,
    VisualObservation,
    VisionStatus,
    VisionError,
    ImageValidationError,
    AnalyzerUnavailableError,
    AnalyzerTimeoutError,
    UnsupportedFormatError,
    ImageTooLargeError,
    LowConfidenceError,
    VisionAnalyzer,
)
from kisansathi.vision.image import validate_image_input, ValidatedImage
from kisansathi.vision.fake import FakeVisionAnalyzer, make_fake_vision_analyzer, TrackingFakeVisionAnalyzer, make_tracking_fake_vision
from kisansathi.vision.providers import create_vision_analyzer, GeminiVisionAnalyzer

__all__ = [
    "VisionResult",
    "VisualObservation",
    "VisionStatus",
    "VisionError",
    "ImageValidationError",
    "AnalyzerUnavailableError",
    "AnalyzerTimeoutError",
    "UnsupportedFormatError",
    "ImageTooLargeError",
    "LowConfidenceError",
    "VisionAnalyzer",
    "validate_image_input",
    "ValidatedImage",
    "FakeVisionAnalyzer",
    "make_fake_vision_analyzer",
    "TrackingFakeVisionAnalyzer",
    "make_tracking_fake_vision",
    "create_vision_analyzer",
    "GeminiVisionAnalyzer",
]