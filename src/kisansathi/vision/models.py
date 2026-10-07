"""Domain models for vision analysis.

Vision is treated as an observation with explicit uncertainty.
The model never produces a definitive diagnosis.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from kisansathi.domain.schemas import Language


class VisionStatus(StrEnum):
    """Outcome of a vision analysis attempt."""

    SUCCESS = "success"
    UNABLE_TO_ANALYZE = "unable_to_analyze"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_IMAGE = "invalid_image"
    ANALYZER_UNAVAILABLE = "analyzer_unavailable"
    ANALYZER_TIMEOUT = "analyzer_timeout"
    UNSUPPORTED_FORMAT = "unsupported_format"
    IMAGE_TOO_LARGE = "image_too_large"
    NO_USABLE_INFORMATION = "no_usable_information"


@dataclass(frozen=True, slots=True)
class VisualObservation:
    """A single visual observation with explicit uncertainty."""

    label: str
    category: Literal["crop", "disease", "pest", "stress", "healthy", "other"]
    confidence: float
    description: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")

    @property
    def is_low_confidence(self) -> bool:
        """Return True if this observation has low confidence (< 0.5)."""
        return self.confidence < 0.5

    @property
    def is_uncertain(self) -> bool:
        """Return True if this observation should be treated as uncertain (< 0.7)."""
        return self.confidence < 0.7


@dataclass(frozen=True, slots=True)
class VisionResult:
    """Result of vision analysis with explicit uncertainty metadata."""

    status: VisionStatus
    observations: tuple[VisualObservation, ...] = ()
    language: str = "en"
    confidence_overall: float = 0.0
    analyzer_metadata: dict | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observations, tuple):
            raise TypeError("observations must be a tuple")
        if self.status == VisionStatus.SUCCESS and not self.observations:
            raise ValueError("SUCCESS status requires at least one observation")
        # LOW_CONFIDENCE and UNABLE_TO_ANALYZE are clarification statuses that may have observations
        if self.status not in (VisionStatus.SUCCESS, VisionStatus.LOW_CONFIDENCE, VisionStatus.UNABLE_TO_ANALYZE) and self.observations:
            raise ValueError("Failed status must have empty observations")
        if not (0.0 <= self.confidence_overall <= 1.0):
            raise ValueError("confidence_overall must be between 0.0 and 1.0")

    @property
    def is_success(self) -> bool:
        """Return True if analysis succeeded with observations."""
        return self.status == VisionStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Return True if analysis failed or produced no usable information."""
        return self.status != VisionStatus.SUCCESS

    @property
    def should_abstain(self) -> bool:
        """Return True if the result should cause the pipeline to abstain."""
        return self.status in (
            VisionStatus.INVALID_IMAGE,
            VisionStatus.UNSUPPORTED_FORMAT,
            VisionStatus.IMAGE_TOO_LARGE,
            VisionStatus.ANALYZER_UNAVAILABLE,
            VisionStatus.ANALYZER_TIMEOUT,
            VisionStatus.NO_USABLE_INFORMATION,
        )

    @property
    def should_clarify(self) -> bool:
        """Return True if the result should cause the pipeline to ask for clarification."""
        return self.status in (
            VisionStatus.LOW_CONFIDENCE,
            VisionStatus.UNABLE_TO_ANALYZE,
        )

    @property
    def has_uncertain_observations(self) -> bool:
        """Return True if any observation has low confidence."""
        return any(obs.is_uncertain for obs in self.observations)

    @property
    def has_disease_or_pest(self) -> bool:
        """Return True if any observation indicates disease or pest."""
        return any(obs.category in ("disease", "pest") for obs in self.observations)

    @property
    def has_crop_observation(self) -> bool:
        """Return True if any observation identifies a crop."""
        return any(obs.category == "crop" for obs in self.observations)

    def get_high_confidence_observations(self, threshold: float = 0.7) -> tuple[VisualObservation, ...]:
        """Return observations with confidence >= threshold."""
        return tuple(obs for obs in self.observations if obs.confidence >= threshold)

    def get_uncertain_observations(self, threshold: float = 0.7) -> tuple[VisualObservation, ...]:
        """Return observations with confidence < threshold."""
        return tuple(obs for obs in self.observations if obs.confidence < threshold)

    def format_for_retrieval(self, max_labels: int = 3) -> str:
        """Format observations for retrieval query augmentation.
        
        Only includes observations with confidence >= 0.5.
        """
        relevant = [obs for obs in self.observations if obs.confidence >= 0.5]
        if not relevant:
            return ""
        labels = [obs.label for obs in relevant[:max_labels]]
        return " ".join(labels)

    def format_for_generation(self) -> str:
        """Format observations for answer generation with uncertainty markers.
        
        Uncertain observations are marked with 'possibly' or 'may indicate'.
        """
        if not self.observations:
            return ""
        
        parts = []
        for obs in self.observations:
            if obs.is_uncertain:
                parts.append(f"possibly {obs.label} ({obs.category}, confidence: {obs.confidence:.2f})")
            else:
                parts.append(f"{obs.label} ({obs.category}, confidence: {obs.confidence:.2f})")
        return "; ".join(parts)


class VisionError(Exception):
    """Base exception for vision analysis failures."""

    pass


class ImageValidationError(VisionError):
    """Raised when input image fails validation."""

    pass


class AnalyzerUnavailableError(VisionError):
    """Raised when vision service is unavailable."""

    pass


class AnalyzerTimeoutError(VisionError):
    """Raised when analysis times out."""

    pass


class UnsupportedFormatError(VisionError):
    """Raised for unsupported image formats."""

    pass


class ImageTooLargeError(VisionError):
    """Raised when image exceeds size limit."""

    pass


class LowConfidenceError(VisionError):
    """Raised when confidence is below threshold but analysis completed."""

    pass


from typing import Protocol

class VisionAnalyzer(Protocol):
    """Injectable vision analysis interface.

    Implementations live in infrastructure layer (hosted API, local model, etc.).
    Domain code depends only on this protocol.
    """

    def analyze(
        self,
        image_data: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        max_observations: int = 5,
        confidence_threshold: float = 0.3,
    ) -> "VisionResult":
        """Analyze an image and return structured observations.

        Args:
            image_data: Raw image bytes.
            content_type: MIME type if known (e.g., "image/jpeg").
            filename: Original filename if available.
            max_observations: Maximum observations to return.
            confidence_threshold: Minimum confidence to include observation.

        Returns:
            VisionResult with status and observations.

        Raises:
            ImageValidationError: Invalid/corrupted image.
            AnalyzerUnavailableError: Service unavailable.
            AnalyzerTimeoutError: Request timed out.
            UnsupportedFormatError: Unsupported format.
            ImageTooLargeError: Image exceeds size limit.
        """
        ...