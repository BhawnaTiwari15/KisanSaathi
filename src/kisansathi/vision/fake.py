"""Fake VisionAnalyzer implementations for testing.

These fakes allow testing without network calls or model weights.
"""

from kisansathi.vision.models import (
    VisionResult,
    VisualObservation,
    VisionStatus,
    VisionError,
    AnalyzerUnavailableError,
    AnalyzerTimeoutError,
    UnsupportedFormatError,
    ImageTooLargeError,
    LowConfidenceError,
    VisionAnalyzer,
)
from kisansathi.domain.schemas import Language


class FakeVisionAnalyzer:
    """Fake VisionAnalyzer implementation for testing.

    Returns canned responses or raises configured exceptions.
    """

    def __init__(
        self,
        result: "VisionResult | VisionError | None" = None,
    ) -> None:
        self._result = result or VisionResult(
            status=VisionStatus.SUCCESS,
            observations=(
                VisualObservation(
                    label="tomato plant",
                    category="crop",
                    confidence=0.9,
                    description="Healthy tomato plant with green leaves",
                ),
            ),
            confidence_overall=0.9,
            language="en",
        )
        self.calls: list[dict] = []

    def analyze(
        self,
        image_data: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        max_observations: int = 5,
        confidence_threshold: float = 0.3,
    ) -> "VisionResult":
        self.calls.append({
            "image_size": len(image_data),
            "content_type": content_type,
            "filename": filename,
            "max_observations": max_observations,
            "confidence_threshold": confidence_threshold,
        })
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def set_result(self, result: "VisionResult | VisionError") -> None:
        """Update the result returned by analyze()."""
        self._result = result


def make_fake_vision_analyzer(
    text: str = "Fake vision result",
    language: str = "en",
    confidence: float = 1.0,
    status: str = "success",
    error: Exception | None = None,
    observations: tuple[VisualObservation, ...] | None = None,
) -> FakeVisionAnalyzer:
    """Create a FakeVisionAnalyzer with a canned result or error."""
    from kisansathi.vision.models import VisionStatus, VisionResult, VisualObservation

    if error is not None:
        return FakeVisionAnalyzer(error)

    if status == "success":
        if observations is None:
            observations = (
                VisualObservation(
                    label="tomato plant",
                    category="crop",
                    confidence=0.9,
                    description="Healthy tomato plant",
                ),
            )
        result = VisionResult(
            status=VisionStatus.SUCCESS,
            observations=observations,
            confidence_overall=confidence,
            language=language,
        )
    elif status == "low_confidence":
        result = VisionResult(
            status=VisionStatus.LOW_CONFIDENCE,
            observations=(),
            language=language,
        )
    elif status == "unable_to_analyze":
        result = VisionResult(
            status=VisionStatus.UNABLE_TO_ANALYZE,
            observations=(),
            language=language,
        )
    elif status == "invalid_image":
        result = VisionResult(
            status=VisionStatus.INVALID_IMAGE,
            observations=(),
            language=language,
        )
    elif status == "analyzer_unavailable":
        result = VisionResult(
            status=VisionStatus.ANALYZER_UNAVAILABLE,
            observations=(),
            language=language,
        )
    elif status == "no_usable_information":
        result = VisionResult(
            status=VisionStatus.NO_USABLE_INFORMATION,
            observations=(),
            language=language,
        )
    else:
        result = VisionResult(
            status=VisionStatus.SUCCESS,
            observations=(
                VisualObservation(
                    label="test observation",
                    category="other",
                    confidence=0.5,
                    description="Test observation",
                ),
            ),
            confidence_overall=0.5,
            language=language,
        )

    return FakeVisionAnalyzer(result)


class TrackingFakeVisionAnalyzer:
    """Fake VisionAnalyzer that tracks calls and can be configured per call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._default_result: VisionResult = VisionResult(
            status=VisionStatus.SUCCESS,
            observations=(
                VisualObservation(
                    label="default crop",
                    category="crop",
                    confidence=0.8,
                    description="Default observation",
                ),
            ),
            confidence_overall=0.8,
            language="en",
        )
        self._overrides: dict[str, VisionResult | VisionError] = {}

    def set_result(self, key: str, result: "VisionResult | VisionError") -> None:
        """Configure a specific result for a given key (e.g., image size)."""
        self._overrides[key] = result

    def set_default(self, result: VisionResult) -> None:
        self._default_result = result

    def analyze(
        self,
        image_data: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        max_observations: int = 5,
        confidence_threshold: float = 0.3,
    ) -> "VisionResult":
        call_info = {
            "image_size": len(image_data),
            "content_type": content_type,
            "filename": filename,
            "max_observations": max_observations,
            "confidence_threshold": confidence_threshold,
        }
        self.calls.append(call_info)

        # Check for override based on image size (simple key)
        key = str(len(image_data))
        if key in self._overrides:
            result = self._overrides[key]
        else:
            result = self._default_result

        if isinstance(result, Exception):
            raise result
        return result


def make_tracking_fake_vision() -> TrackingFakeVisionAnalyzer:
    """Create a TrackingFakeVisionAnalyzer."""
    return TrackingFakeVisionAnalyzer()