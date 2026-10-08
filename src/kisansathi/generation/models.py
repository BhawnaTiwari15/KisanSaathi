"""Domain models and protocols for answer generation.

The LLM is treated as a replaceable infrastructure dependency. Domain code depends only
on the ``AnswerGenerator`` protocol; provider-specific implementations live elsewhere.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kisansathi.citations.models import CitationBatch
from kisansathi.domain.schemas import Language, ResponseStatus, UserMessage
from kisansathi.eligibility.models import EligibilityDecision
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.weather.models import WeatherResponse
from kisansathi.vision.models import VisionResult


class GenerationError(Exception):
    """Base error for answer generation failures."""


class LLMError(GenerationError):
    """Raised when the LLM call fails (network, timeout, provider error)."""


class MalformedOutputError(GenerationError):
    """Raised when the LLM output does not match the required format."""


class GroundingError(GenerationError):
    """Raised when the model cites an id not present in the supplied CitationBatch."""


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """All grounded context the generator needs, assembled by the graph."""

    message: UserMessage
    citations: CitationBatch
    eligibility_decision: EligibilityDecision | None
    weather: WeatherResponse | None
    vision_result: VisionResult | None
    retrieved_chunks: tuple[SearchResult, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Structured output from the generator, before final validation."""

    text: str
    citation_ids: tuple[str, ...]
    status: ResponseStatus
    language: Language


@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM interface. Implementations live in infrastructure, not domain."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Return raw model text. Raises LLMError on transport failure."""
        ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """High-level generation entry point. Domain code depends on this."""

    def generate(self, context: GenerationContext) -> GeneratedAnswer:
        ...