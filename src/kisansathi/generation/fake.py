"""Fake implementations for testing answer generation without network calls or models."""

from typing import Any

from kisansathi.citations.models import CitationBatch
from kisansathi.domain.schemas import Language, ResponseStatus, UserMessage
from kisansathi.eligibility.models import EligibilityDecision
from kisansathi.generation.models import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationContext,
    LLMClient,
    LLMError,
)
from kisansathi.weather.models import WeatherResponse


class FakeLLMClient:
    """Fake LLM client that returns a canned response or raises a configured error."""

    def __init__(
        self,
        response: str | Exception = "ANSWER: This is a test answer.\nCITATIONS:\n[chunk-1]",
    ) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeAnswerGenerator:
    """Fake AnswerGenerator that returns a canned GeneratedAnswer or raises an error."""

    def __init__(
        self,
        answer: GeneratedAnswer | Exception = None,
    ) -> None:
        if answer is None:
            answer = GeneratedAnswer(
                text="Fake answer from test generator.",
                citation_ids=(),
                status=ResponseStatus.ANSWERED,
                language=Language.ENGLISH,
            )
        self._answer = answer
        self.calls: list[GenerationContext] = []

    def generate(self, context: GenerationContext) -> GeneratedAnswer:
        self.calls.append(context)
        if isinstance(self._answer, Exception):
            raise self._answer
        # If the answer has no explicit citation_ids, use the ones from the context
        if self._answer.citation_ids:
            return self._answer
        # Use citation IDs from the context's citation batch
        citation_ids = tuple(context.citations.citation_ids) if context.citations else ()
        return GeneratedAnswer(
            text=self._answer.text,
            citation_ids=citation_ids,
            status=self._answer.status,
            language=self._answer.language,
        )


def make_fake_generated_answer(
    text: str = "Fake answer.",
    citation_ids: tuple[str, ...] = (),
    status: ResponseStatus = ResponseStatus.ANSWERED,
    language: Language = Language.ENGLISH,
) -> GeneratedAnswer:
    """Helper to create a GeneratedAnswer for tests."""
    return GeneratedAnswer(
        text=text,
        citation_ids=citation_ids,
        status=status,
        language=language,
    )


def make_fake_generation_context(
    message_text: str = "Test question",
    citations: CitationBatch | None = None,
    eligibility_decision: EligibilityDecision | None = None,
    weather: WeatherResponse | None = None,
    vision_result: "VisionResult | None" = None,
    language: Language = Language.ENGLISH,
) -> GenerationContext:
    """Helper to create a GenerationContext for tests."""
    from kisansathi.citations.models import CitationBatch
    from kisansathi.vision.models import VisionResult

    return GenerationContext(
        message=UserMessage(text=message_text, language=language),
        citations=citations or CitationBatch(),
        eligibility_decision=eligibility_decision,
        weather=weather,
        vision_result=vision_result,
    )