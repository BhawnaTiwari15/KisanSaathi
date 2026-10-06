"""Default AnswerGenerator implementation using an injected LLMClient."""

from kisansathi.citations.models import CitationBatch
from kisansathi.citations.resolver import validate_referenced_citations
from kisansathi.domain.schemas import Language
from kisansathi.generation.models import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationContext,
    GroundingError,
    LLMClient,
    LLMError,
    MalformedOutputError,
)
from kisansathi.generation.prompts import (
    build_system_prompt,
    build_user_prompt,
    parse_generated_answer,
)


class DefaultAnswerGenerator:
    """Generates answers by calling an LLMClient and validating the output."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        if not isinstance(llm_client, LLMClient):
            # We can't use isinstance with Protocol at runtime, but we can check callable
            if not callable(getattr(llm_client, "generate", None)):
                raise TypeError("llm_client must have a generate() method")
        self._llm_client = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, context: GenerationContext) -> GeneratedAnswer:
        system_prompt = build_system_prompt(context.message.language or Language.ENGLISH)
        user_prompt = build_user_prompt(
            context.message,
            context.citations,
            context.eligibility_decision,
            context.weather,
        )

        allowed_ids = frozenset(context.citations.citation_ids)

        try:
            raw = self._llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}") from e

        try:
            answer = parse_generated_answer(raw, allowed_ids, context.message.language or Language.ENGLISH)
        except (MalformedOutputError, GroundingError):
            raise

        # Post-generation validation using the existing citation resolver gate
        try:
            validated = validate_referenced_citations(answer.citation_ids, context.citations)
        except Exception as e:
            raise GroundingError(f"Post-generation citation validation failed: {e}") from e

        # Replace citation_ids with the validated citation objects' chunk_ids
        # (validated is a tuple of Citation objects)
        validated_ids = tuple(c.chunk_id for c in validated if c.chunk_id is not None)

        return GeneratedAnswer(
            text=answer.text,
            citation_ids=validated_ids,
            status=answer.status,
            language=answer.language,
        )