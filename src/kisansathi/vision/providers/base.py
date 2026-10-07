"""Base class for vision providers with shared validation and retry logic."""

import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from kisansathi.config import Settings
from kisansathi.vision.models import (
    VisionResult,
    VisualObservation,
    VisionStatus,
    ImageValidationError,
    AnalyzerUnavailableError,
    AnalyzerTimeoutError,
    UnsupportedFormatError,
    ImageTooLargeError,
    LowConfidenceError,
)
from kisansathi.vision.image import validate_image_input, ValidatedImage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderObservation:
    """Raw observation from provider before validation."""

    label: str
    category: str
    confidence: float
    description: str


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Raw response from provider before validation."""

    observations: tuple[ProviderObservation, ...]
    language: str
    overall_confidence: float
    notes: str | None


class ProviderError(Exception):
    """Base exception for provider-specific errors."""

    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class AuthenticationError(ProviderError):
    """Provider authentication failed (401, 403)."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message, retryable=False, status_code=status_code)


class RateLimitError(ProviderError):
    """Provider rate limit exceeded (429)."""

    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message, retryable=True, status_code=status_code)


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""

    def __init__(self, message: str):
        super().__init__(message, retryable=True, status_code=None)


class ProviderUnavailableError(ProviderError):
    """Provider service unavailable (5xx)."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message, retryable=True, status_code=status_code)


class InvalidRequestError(ProviderError):
    """Invalid request to provider (400)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, retryable=False, status_code=status_code)


class MalformedResponseError(ProviderError):
    """Provider returned malformed or invalid structured output."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False, status_code=None)


class BaseVisionAnalyzer(ABC):
    """Abstract base class for vision providers.

    Handles:
    - Image validation (delegates to domain validator)
    - Retry logic for transient failures
    - Structured logging/metrics
    - Provider response validation against VisionResult schema
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=settings.vision_connect_timeout_seconds,
                read=settings.vision_timeout_seconds,
                write=settings.vision_timeout_seconds,
                pool=settings.vision_timeout_seconds,
            ),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )

    def analyze(
        self,
        image_data: bytes,
        *,
        content_type: str | None = None,
        filename: str | None = None,
        max_observations: int = 5,
        confidence_threshold: float = 0.3,
    ) -> VisionResult:
        """Analyze an image and return structured observations.

        This is the main entry point implementing the VisionAnalyzer protocol.
        """
        start_time = time.monotonic()
        retry_count = 0

        # Validate image before any provider call
        validated = self._validate_image(image_data, content_type, filename)

        # Check image size against provider limit
        if validated.size_bytes > self._settings.vision_max_image_bytes:
            self._log_failure("image_too_large", start_time, retry_count)
            raise ImageTooLargeError(
                f"Image too large for provider: {validated.size_bytes} bytes "
                f"(maximum {self._settings.vision_max_image_bytes})"
            )

        last_error: Exception | None = None

        while retry_count <= self._settings.vision_max_retries:
            try:
                provider_response = self._call_provider(validated, max_observations)
                vision_result = self._validate_and_build_result(
                    provider_response, confidence_threshold
                )
                self._log_success(start_time, retry_count, vision_result)
                return vision_result

            except ProviderError as e:
                last_error = e
                if not e.retryable or retry_count >= self._settings.vision_max_retries:
                    self._log_failure(type(e).__name__, start_time, retry_count, str(e))
                    raise self._map_provider_error(e)
                retry_count += 1
                self._wait_before_retry(retry_count)
                logger.warning(
                    "Vision provider error (retry %d/%d): %s",
                    retry_count,
                    self._settings.vision_max_retries,
                    e,
                )

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = ProviderTimeoutError(f"Provider request failed: {e}")
                if retry_count >= self._settings.vision_max_retries:
                    self._log_failure("timeout", start_time, retry_count, str(e))
                    raise self._map_provider_error(last_error)
                retry_count += 1
                self._wait_before_retry(retry_count)
                logger.warning(
                    "Vision provider timeout (retry %d/%d): %s",
                    retry_count,
                    self._settings.vision_max_retries,
                    e,
                )

            except Exception as e:
                last_error = ProviderUnavailableError(f"Unexpected provider error: {e}", 500)
                if retry_count >= self._settings.vision_max_retries:
                    self._log_failure("unexpected_error", start_time, retry_count, str(e))
                    raise self._map_provider_error(last_error)
                retry_count += 1
                self._wait_before_retry(retry_count)
                logger.warning(
                    "Vision provider unexpected error (retry %d/%d): %s",
                    retry_count,
                    self._settings.vision_max_retries,
                    e,
                )

        # Should not reach here, but safety net
        assert last_error is not None
        self._log_failure("max_retries_exceeded", start_time, retry_count, str(last_error))
        raise self._map_provider_error(last_error)

    def _validate_image(
        self, image_data: bytes, content_type: str | None, filename: str | None
    ) -> ValidatedImage:
        """Validate image using domain validator."""
        try:
            return validate_image_input(
                image_data,
                content_type or "image/jpeg",
                max_size=self._settings.vision_max_image_bytes,
                filename=filename,
            )
        except (ImageValidationError, UnsupportedFormatError, ImageTooLargeError):
            raise
        except Exception as e:
            raise ImageValidationError(f"Image validation failed: {e}") from e

    @abstractmethod
    def _call_provider(self, validated: ValidatedImage, max_observations: int) -> ProviderResponse:
        """Call the provider API and return parsed provider response."""
        ...

    def _validate_and_build_result(
        self, provider_response: ProviderResponse, confidence_threshold: float
    ) -> VisionResult:
        """Validate provider response and build VisionResult."""
        # Filter observations by confidence threshold
        filtered_obs = [
            obs for obs in provider_response.observations
            if obs.confidence >= confidence_threshold
        ]

        # Convert to domain observations
        observations = []
        for obs in filtered_obs:
            try:
                observations.append(VisualObservation(
                    label=obs.label.strip()[:100],  # Cap label length
                    category=obs.category.lower().strip(),
                    confidence=max(0.0, min(1.0, obs.confidence)),  # Clamp
                    description=obs.description.strip()[:500],  # Cap description
                ))
            except ValueError as e:
                logger.warning("Skipping invalid observation from provider: %s", e)
                continue

        # Determine status
        if not observations:
            status = VisionStatus.NO_USABLE_INFORMATION
        elif any(obs.confidence < 0.5 for obs in observations):
            status = VisionStatus.LOW_CONFIDENCE
        else:
            status = VisionStatus.SUCCESS

        # Compute overall confidence (mean of filtered observations)
        if observations:
            overall = sum(obs.confidence for obs in observations) / len(observations)
        else:
            overall = provider_response.overall_confidence

        return VisionResult(
            status=status,
            observations=tuple(observations),
            language=provider_response.language or "en",
            confidence_overall=max(0.0, min(1.0, overall)),
            analyzer_metadata={
                "provider": self.provider_name,
                "model": self._settings.vision_model_name,
                "notes": provider_response.notes,
            },
            error_message=None,
        )

    def _map_provider_error(self, error: ProviderError) -> Exception:
        """Map provider errors to domain VisionError exceptions."""
        if isinstance(error, AuthenticationError):
            return AnalyzerUnavailableError(f"Vision provider authentication failed: {error}")
        elif isinstance(error, RateLimitError):
            return AnalyzerUnavailableError(f"Vision provider rate limited: {error}")
        elif isinstance(error, ProviderTimeoutError):
            return AnalyzerTimeoutError(f"Vision provider timeout: {error}")
        elif isinstance(error, ProviderUnavailableError):
            return AnalyzerUnavailableError(f"Vision provider unavailable: {error}")
        elif isinstance(error, InvalidRequestError):
            return ImageValidationError(f"Invalid vision request: {error}")
        elif isinstance(error, MalformedResponseError):
            return AnalyzerUnavailableError(f"Vision provider returned invalid response: {error}")
        else:
            return AnalyzerUnavailableError(f"Vision provider error: {error}")

    def _wait_before_retry(self, attempt: int) -> None:
        """Wait with exponential backoff before retry."""
        wait_time = self._settings.vision_retry_backoff_base * (2 ** (attempt - 1))
        time.sleep(wait_time)

    def _log_success(
        self,
        start_time: float,
        retry_count: int,
        result: VisionResult,
    ) -> None:
        """Log successful analysis."""
        latency_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Vision analysis completed",
            extra={
                "provider": self.provider_name,
                "model": self._settings.vision_model_name,
                "latency_ms": round(latency_ms, 1),
                "status": result.status.value,
                "observations_count": len(result.observations),
                "max_confidence": round(max((o.confidence for o in result.observations), default=0.0), 2),
                "retry_count": retry_count,
            },
        )

    def _log_failure(
        self,
        error_category: str,
        start_time: float,
        retry_count: int,
        error_message: str | None = None,
    ) -> None:
        """Log failed analysis."""
        latency_ms = (time.monotonic() - start_time) * 1000
        logger.warning(
            "Vision analysis failed: %s",
            error_category,
            extra={
                "provider": self.provider_name,
                "model": self._settings.vision_model_name,
                "latency_ms": round(latency_ms, 1),
                "error_category": error_category,
                "error_message": error_message,
                "retry_count": retry_count,
            },
        )

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name for logging."""
        ...

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()