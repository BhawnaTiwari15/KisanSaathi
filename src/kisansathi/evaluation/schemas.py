"""Versioned evaluation dataset schemas."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
import json


class Language(StrEnum):
    ENGLISH = "en"
    HINDI = "hi"
    KANNADA = "kn"
    TELUGU = "te"


class RetrievalCategory(StrEnum):
    ELIGIBILITY = "eligibility"
    SCHEME_DETAILS = "scheme_details"
    WEATHER = "weather"
    GENERAL = "general"
    DOCUMENT_LOOKUP = "document_lookup"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    ABSTAINED = "abstained"


class RefusalType(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_SCHEME = "unsupported_scheme"
    AMBIGUOUS = "ambiguous"
    UNSAFE = "unsafe"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    """A single retrieval evaluation example."""

    id: str
    language: Language
    query: str
    expected_chunk_ids: tuple[str, ...]
    expected_source_ids: tuple[str, ...] = ()
    category: RetrievalCategory | None = None
    difficulty: Difficulty | None = None
    relevance_grades: dict[str, int] | None = None
    annotator: str | None = None
    annotation_timestamp_utc: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must not be empty")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.expected_chunk_ids:
            raise ValueError("expected_chunk_ids must not be empty")
        if any(not cid.strip() for cid in self.expected_chunk_ids):
            raise ValueError("expected_chunk_ids must not contain empty strings")
        if len(set(self.expected_chunk_ids)) != len(self.expected_chunk_ids):
            raise ValueError("expected_chunk_ids must not contain duplicates")
        if self.relevance_grades is not None:
            for chunk_id, grade in self.relevance_grades.items():
                if not isinstance(grade, int) or not (0 <= grade <= 3):
                    raise ValueError(f"relevance_grades must be integers 0-3, got {grade} for {chunk_id}")
                if chunk_id not in self.expected_chunk_ids:
                    raise ValueError(f"relevance_grades contains chunk_id not in expected_chunk_ids: {chunk_id}")


@dataclass(frozen=True, slots=True)
class RetrievalDataset:
    """Versioned retrieval evaluation dataset."""

    evaluation_set_version: str
    corpus_version: str
    examples: tuple[RetrievalExample, ...] = ()
    unavailable: bool = False
    unavailable_reason: str = ""
    annotation_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evaluation_set_version.strip():
            raise ValueError("evaluation_set_version must not be empty")
        if not self.corpus_version.strip():
            raise ValueError("corpus_version must not be empty")
        # Allow empty examples for unavailable datasets
        if not self.unavailable and not self.examples:
            raise ValueError("examples must not be empty for available datasets")
        ids = [ex.id for ex in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("example ids must be unique")
        # Validate all examples have the same language
        if self.examples:
            first_lang = self.examples[0].language
            for ex in self.examples:
                if ex.language != first_lang:
                    raise ValueError("all examples in a dataset must have the same language")

    @property
    def language(self) -> Language:
        """Return the language of this dataset (assumes single-language dataset)."""
        if not self.examples:
            return Language.ENGLISH
        return self.examples[0].language

    def filter_by_category(self, category: RetrievalCategory) -> "RetrievalDataset":
        """Return a new dataset with only examples of the given category."""
        filtered = tuple(ex for ex in self.examples if ex.category == category)
        return RetrievalDataset(
            evaluation_set_version=self.evaluation_set_version,
            corpus_version=self.corpus_version,
            examples=filtered,
        )

    def filter_by_difficulty(self, difficulty: Difficulty) -> "RetrievalDataset":
        """Return a new dataset with only examples of the given difficulty."""
        filtered = tuple(ex for ex in self.examples if ex.difficulty == difficulty)
        return RetrievalDataset(
            evaluation_set_version=self.evaluation_set_version,
            corpus_version=self.corpus_version,
            examples=filtered,
        )


@dataclass(frozen=True, slots=True)
class AnswerExample:
    """A single answer generation evaluation example."""

    id: str
    language: Language
    query: str
    expected_answer_characteristics: dict[str, Any] = field(default_factory=dict)
    expected_citation_source_ids: tuple[str, ...] = ()
    expected_status: AnswerStatus = AnswerStatus.ANSWERED
    expected_language: Language = Language.ENGLISH
    refusal_expected: bool = False
    refusal_type: RefusalType | None = None
    category: str | None = None
    difficulty: Difficulty | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must not be empty")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.refusal_expected and self.refusal_type is None:
            raise ValueError("refusal_type required when refusal_expected is True")
        if not self.refusal_expected and self.refusal_type is not None:
            raise ValueError("refusal_type only allowed when refusal_expected is True")


@dataclass(frozen=True, slots=True)
class AnswerDataset:
    """Versioned answer generation evaluation dataset."""

    evaluation_set_version: str
    corpus_version: str
    system_version: str = ""
    examples: tuple[AnswerExample, ...] = ()

    def __post_init__(self) -> None:
        if not self.evaluation_set_version.strip():
            raise ValueError("evaluation_set_version must not be empty")
        if not self.corpus_version.strip():
            raise ValueError("corpus_version must not be empty")
        if not self.examples:
            raise ValueError("examples must not be empty")
        ids = [ex.id for ex in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("example ids must be unique")

    @property
    def language(self) -> Language:
        if not self.examples:
            return Language.ENGLISH
        return self.examples[0].language


@dataclass(frozen=True, slots=True)
class SafetyExample:
    """A single safety/refusal evaluation example."""

    id: str
    language: Language
    query: str
    expected_status: AnswerStatus
    refusal_type: RefusalType | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must not be empty")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.expected_status in (AnswerStatus.NEEDS_CLARIFICATION, AnswerStatus.ABSTAINED):
            if self.refusal_type is None:
                raise ValueError("refusal_type required for clarification/abstention")
        else:
            if self.refusal_type is not None:
                raise ValueError("refusal_type only allowed for clarification/abstention")


@dataclass(frozen=True, slots=True)
class SafetyDataset:
    """Versioned safety evaluation dataset."""

    evaluation_set_version: str
    examples: tuple[SafetyExample, ...] = ()

    def __post_init__(self) -> None:
        if not self.evaluation_set_version.strip():
            raise ValueError("evaluation_set_version must not be empty")
        if not self.examples:
            raise ValueError("examples must not be empty")
        ids = [ex.id for ex in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("example ids must be unique")


@dataclass(frozen=True, slots=True)
class AnswerQualityEvaluationCase:
    """Versioned answer-quality evaluation case for LLM-judge/RAGAS metrics.

    This is separate from AnswerExample (which drives deterministic checks).
    This case contains all inputs needed for semantic evaluation metrics:
    - Faithfulness: generated_answer vs retrieved_contexts
    - Answer Relevance: generated_answer vs query
    - Context Precision: retrieved_contexts vs reference_contexts (when available)
    - Context Recall: retrieved_contexts vs reference_contexts (when available)

    All fields are optional except case_id, query, and language to allow
    partial evaluation when some inputs are unavailable.
    """

    case_id: str
    query: str
    language: Language
    retrieved_contexts: tuple[str, ...] = ()
    generated_answer: str = ""
    citation_ids: tuple[str, ...] = ()
    reference_answer: str | None = None
    reference_contexts: tuple[str, ...] = ()
    expected_response_status: AnswerStatus = AnswerStatus.ANSWERED
    benchmark_version: str = ""

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.reference_answer is not None and not self.reference_answer.strip():
            raise ValueError("reference_answer must not be empty string if provided")
        if not self.benchmark_version.strip():
            raise ValueError("benchmark_version must not be empty")

    @property
    def has_reference_answer(self) -> bool:
        """Whether reference answer is available for faithfulness/answer relevance."""
        return self.reference_answer is not None and bool(self.reference_answer.strip())

    @property
    def has_reference_contexts(self) -> bool:
        """Whether reference contexts are available for context precision/recall."""
        return bool(self.reference_contexts)

    @property
    def can_compute_faithfulness(self) -> bool:
        """Faithfulness needs generated_answer and retrieved_contexts."""
        return bool(self.generated_answer.strip()) and bool(self.retrieved_contexts)

    @property
    def can_compute_answer_relevance(self) -> bool:
        """Answer relevance needs generated_answer and query."""
        return bool(self.generated_answer.strip()) and bool(self.query.strip())

    @property
    def can_compute_context_precision(self) -> bool:
        """Context precision needs retrieved_contexts and reference_contexts."""
        return bool(self.retrieved_contexts) and self.has_reference_contexts

    @property
    def can_compute_context_recall(self) -> bool:
        """Context recall needs retrieved_contexts and reference_contexts."""
        return bool(self.retrieved_contexts) and self.has_reference_contexts


@dataclass(frozen=True, slots=True)
class AnswerQualityEvaluationDataset:
    """Versioned answer-quality evaluation dataset for LLM-judge metrics."""

    benchmark_version: str
    corpus_version: str
    system_version: str = ""
    examples: tuple[AnswerQualityEvaluationCase, ...] = ()

    def __post_init__(self) -> None:
        if not self.benchmark_version.strip():
            raise ValueError("benchmark_version must not be empty")
        if not self.corpus_version.strip():
            raise ValueError("corpus_version must not be empty")
        if not self.examples:
            raise ValueError("examples must not be empty")
        ids = [ex.case_id for ex in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("case_id values must be unique")


def load_answer_quality_dataset(path: str | Path) -> AnswerQualityEvaluationDataset:
    """Load and validate an answer-quality evaluation dataset from JSON."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read evaluation dataset {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evaluation dataset JSON: {error.msg}") from error

    if not isinstance(raw, dict):
        raise ValueError("evaluation dataset must be a JSON object")

    examples_data = raw.get("examples")
    if not isinstance(examples_data, list):
        raise ValueError("evaluation dataset examples must be a list")

    if not examples_data:
        raise ValueError("evaluation dataset examples must not be empty")

    examples: list[AnswerQualityEvaluationCase] = []
    for index, item in enumerate(examples_data):
        if not isinstance(item, dict):
            raise ValueError(f"evaluation example {index} must be a JSON object")

        try:
            examples.append(AnswerQualityEvaluationCase(
                case_id=item.get("case_id", f"auto-{index}"),
                query=item["query"],
                language=Language(item.get("language", "en")),
                retrieved_contexts=tuple(item.get("retrieved_contexts", [])),
                generated_answer=item.get("generated_answer", ""),
                citation_ids=tuple(item.get("citation_ids", [])),
                reference_answer=item.get("reference_answer") or None,
                reference_contexts=tuple(item.get("reference_contexts", [])),
                expected_response_status=AnswerStatus(item.get("expected_response_status", "answered")),
                benchmark_version=item.get("benchmark_version", raw.get("benchmark_version", "")),
            ))
        except (KeyError, ValueError) as error:
            raise ValueError(f"evaluation example {index} invalid: {error}") from error

    return AnswerQualityEvaluationDataset(
        benchmark_version=raw["benchmark_version"],
        corpus_version=raw["corpus_version"],
        system_version=raw.get("system_version", ""),
        examples=tuple(examples),
    )


def load_retrieval_dataset(path: str | Path) -> RetrievalDataset:
    """Load and validate a retrieval evaluation dataset from JSON."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read evaluation dataset {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evaluation dataset JSON: {error.msg}") from error

    if not isinstance(raw, dict):
        raise ValueError("evaluation dataset must be a JSON object")

    examples_data = raw.get("examples")
    if not isinstance(examples_data, list):
        raise ValueError("evaluation dataset examples must be a list")

    # Handle unavailable datasets
    if raw.get("unavailable"):
        # Support both 'reason' (legacy) and 'unavailable_reason' (new)
        unavailable_reason = raw.get("unavailable_reason") or raw.get("reason", "No genuine relevance labels available for this language")
        return RetrievalDataset(
            evaluation_set_version=raw["evaluation_set_version"],
            corpus_version=raw["corpus_version"],
            examples=(),
            unavailable=True,
            unavailable_reason=unavailable_reason,
            annotation_metadata=raw.get("annotation_metadata", {}),
        )

    if not examples_data:
        raise ValueError("evaluation dataset examples must be a list")

    examples: list[RetrievalExample] = []
    for index, item in enumerate(examples_data):
        if not isinstance(item, dict):
            raise ValueError(f"evaluation example {index} must be a JSON object")

        # Support both 'relevant_chunk_ids' (legacy) and 'expected_chunk_ids' (new)
        chunk_ids = item.get("expected_chunk_ids") or item.get("relevant_chunk_ids")
        if chunk_ids is None:
            raise ValueError(f"evaluation example {index} missing expected_chunk_ids or relevant_chunk_ids")

        try:
            examples.append(RetrievalExample(
                id=item.get("id", f"auto-{index}"),
                language=Language(item.get("language", "en")),
                query=item["query"],
                expected_chunk_ids=tuple(chunk_ids),
                expected_source_ids=tuple(item.get("expected_source_ids", [])),
                category=RetrievalCategory(item["category"]) if item.get("category") else None,
                difficulty=Difficulty(item["difficulty"]) if item.get("difficulty") else None,
                relevance_grades=item.get("relevance_grades") or None,
                annotator=item.get("annotator") or None,
                annotation_timestamp_utc=item.get("annotation_timestamp_utc") or None,
                notes=item.get("notes", ""),
            ))
        except (KeyError, ValueError) as error:
            raise ValueError(f"evaluation example {index} invalid: {error}") from error

    return RetrievalDataset(
        evaluation_set_version=raw["evaluation_set_version"],
        corpus_version=raw["corpus_version"],
        examples=tuple(examples),
        unavailable=raw.get("unavailable", False),
        unavailable_reason=raw.get("unavailable_reason", ""),
        annotation_metadata=raw.get("annotation_metadata", {}),
    )


def load_answer_dataset(path: str | Path) -> AnswerDataset:
    """Load and validate an answer generation evaluation dataset from JSON."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read evaluation dataset {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evaluation dataset JSON: {error.msg}") from error

    if not isinstance(raw, dict):
        raise ValueError("evaluation dataset must be a JSON object")

    examples_data = raw.get("examples")
    if not isinstance(examples_data, list):
        raise ValueError("evaluation dataset examples must be a list")

    examples: list[AnswerExample] = []
    for index, item in enumerate(examples_data):
        if not isinstance(item, dict):
            raise ValueError(f"evaluation example {index} must be a JSON object")

        try:
            examples.append(AnswerExample(
                id=item.get("id", f"auto-{index}"),
                language=Language(item.get("language", "en")),
                query=item["query"],
                expected_answer_characteristics=item.get("expected_answer_characteristics", {}),
                expected_citation_source_ids=tuple(item.get("expected_citation_source_ids", [])),
                expected_status=AnswerStatus(item.get("expected_status", "answered")),
                expected_language=Language(item.get("expected_language", "en")),
                refusal_expected=item.get("refusal_expected", False),
                refusal_type=RefusalType(item["refusal_type"]) if item.get("refusal_type") else None,
                category=item.get("category"),
                difficulty=Difficulty(item["difficulty"]) if item.get("difficulty") else None,
                notes=item.get("notes", ""),
            ))
        except (KeyError, ValueError) as error:
            raise ValueError(f"evaluation example {index} invalid: {error}") from error

    return AnswerDataset(
        evaluation_set_version=raw["evaluation_set_version"],
        corpus_version=raw["corpus_version"],
        system_version=raw.get("system_version", ""),
        examples=tuple(examples),
    )


def load_safety_dataset(path: str | Path) -> SafetyDataset:
    """Load and validate a safety evaluation dataset from JSON."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read evaluation dataset {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid evaluation dataset JSON: {error.msg}") from error

    if not isinstance(raw, dict):
        raise ValueError("evaluation dataset must be a JSON object")

    examples_data = raw.get("examples")
    if not isinstance(examples_data, list):
        raise ValueError("evaluation dataset examples must be a list")

    examples: list[SafetyExample] = []
    for index, item in enumerate(examples_data):
        if not isinstance(item, dict):
            raise ValueError(f"evaluation example {index} must be a JSON object")

        try:
            examples.append(SafetyExample(
                id=item.get("id", f"auto-{index}"),
                language=Language(item.get("language", "en")),
                query=item["query"],
                expected_status=AnswerStatus(item["expected_status"]),
                refusal_type=RefusalType(item["refusal_type"]) if item.get("refusal_type") else None,
                notes=item.get("notes", ""),
            ))
        except (KeyError, ValueError) as error:
            raise ValueError(f"evaluation example {index} invalid: {error}") from error

    return SafetyDataset(
        evaluation_set_version=raw["evaluation_set_version"],
        examples=tuple(examples),
    )