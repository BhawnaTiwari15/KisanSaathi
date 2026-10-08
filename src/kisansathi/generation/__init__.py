"""Answer generation package.

Public API for LLM-based answer generation with citation grounding.
"""

from kisansathi.generation.fake import (
    FakeAnswerGenerator,
    FakeLLMClient,
    make_fake_generated_answer,
    make_fake_generation_context,
)
from kisansathi.generation.gemini import GeminiTextClient
from kisansathi.generation.generator import DefaultAnswerGenerator
from kisansathi.generation.models import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationContext,
    GenerationError,
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

__all__ = [
    "AnswerGenerator",
    "DefaultAnswerGenerator",
    "FakeAnswerGenerator",
    "FakeLLMClient",
    "GeminiTextClient",
    "GenerationContext",
    "GenerationError",
    "GeneratedAnswer",
    "GroundingError",
    "LLMClient",
    "LLMError",
    "MalformedOutputError",
    "build_system_prompt",
    "build_user_prompt",
    "parse_generated_answer",
    "make_fake_generated_answer",
    "make_fake_generation_context",
]