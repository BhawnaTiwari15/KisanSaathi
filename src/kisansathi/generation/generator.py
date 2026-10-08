"""Default AnswerGenerator implementation using an injected LLMClient."""

import logging

from kisansathi.citations.resolver import validate_referenced_citations
from kisansathi.domain.schemas import Language
from kisansathi.generation.models import (
    GeneratedAnswer,
    GenerationContext,
    GroundingError,
    LLMClient,
    LLMError,
    MalformedOutputError,
)
from kisansathi.generation.prompts import (
    _extract_citation_ids,
    build_system_prompt,
    build_user_prompt,
    parse_generated_answer,
)

logger = logging.getLogger(__name__)


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
            context.vision_result,
            context.retrieved_chunks,
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
            answer = parse_generated_answer(
                raw, allowed_ids, context.message.language or Language.ENGLISH
            )
        except (MalformedOutputError, GroundingError) as e:
            self._log_parse_rejection(
                e, raw, allowed_ids, context.message.language or Language.ENGLISH
            )
            raise

        # Post-generation validation using the existing citation resolver gate
        try:
            validated = validate_referenced_citations(answer.citation_ids, context.citations)
        except Exception as e:
            logger.warning(
                "answer output rejected by post-generation citation validation (diagnostic): "
                "stage=post_generation referenced_ids=%d resolved_citations=%d reason=%s",
                len(answer.citation_ids),
                len(context.citations.citation_ids),
                str(e)[:200],
            )
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

    def _log_parse_rejection(
        self,
        exc: MalformedOutputError | GroundingError,
        raw: str,
        allowed_ids: frozenset[str],
        language: Language,
    ) -> None:
        """Log safe metadata about a rejected model response (diagnostic only).

        Only boolean presence flags, counts, and a truncated exception reason are
        logged - never the raw model response, the prompts, or retrieved source text.
        """
        answer_part = raw.partition("CITATIONS:")[0].replace("ANSWER:", "").strip()
        logger.warning(
            "answer output rejected by parsing (diagnostic): "
            "exception=%s answer_marker=%s answer_empty=%s citations_marker=%s "
            "citation_ids=%d allowed_ids=%d raw_chars=%d language=%s reason=%s",
            type(exc).__name__,
            "ANSWER:" in raw,
            not answer_part,
            "CITATIONS:" in raw,
            len(_extract_citation_ids(raw)),
            len(allowed_ids),
            len(raw),
            language.value,
            str(exc)[:200],
        )