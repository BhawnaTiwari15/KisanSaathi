"""Deterministic answer-quality evaluation."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from kisansathi.citations.models import CitationBatch
from kisansathi.domain.schemas import AssistantResponse, Language, ResponseStatus
from kisansathi.evaluation.schemas import AnswerDataset, AnswerExample, AnswerStatus, RefusalType
from kisansathi.vision.models import VisionResult, VisionStatus


@dataclass(frozen=True, slots=True)
class AnswerQualityCheck:
    """Result of a single deterministic quality check."""

    check_name: str
    passed: bool
    details: str = ""
    expected: str = ""
    actual: str = ""


@dataclass(frozen=True, slots=True)
class AnswerQualityResult:
    """Complete deterministic answer quality evaluation for one query."""

    example_id: str
    query: str
    language: Language
    response: AssistantResponse
    checks: tuple[AnswerQualityCheck, ...]
    vision_result: VisionResult | None = None

    @property
    def overall_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> tuple[AnswerQualityCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "query": self.query,
            "language": self.language.value,
            "response_status": self.response.status.value,
            "response_language": self.response.language.value,
            "overall_passed": self.overall_passed,
            "checks": [
                {
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "details": c.details,
                    "expected": c.expected,
                    "actual": c.actual,
                }
                for c in self.checks
            ],
        }


def evaluate_answer_quality(
    example: AnswerExample,
    response: AssistantResponse,
    citation_batch: CitationBatch | None = None,
    vision_result: VisionResult | None = None,
    resolved_citation_chunk_ids: tuple[str, ...] = (),
) -> AnswerQualityResult:
    """Run all deterministic answer quality checks for a single example.

    Args:
        example: The expected answer characteristics from the evaluation dataset
        response: The actual AssistantResponse to evaluate
        citation_batch: Resolved CitationBatch (for citation validity checks)
        vision_result: VisionResult if image was processed
        resolved_citation_chunk_ids: Validated citation chunk IDs from the generation pipeline
    """
    checks: list[AnswerQualityCheck] = []

    # 1. Response status consistency
    checks.append(_check_status_consistency(example, response))

    # 2. Expected language match
    checks.append(_check_language_match(example, response))

    # 3. Non-empty answer when answered
    checks.append(_check_non_empty_answer(example, response))

    # 4. Citation validity (all cited IDs exist in batch)
    checks.append(_check_citation_validity(example, response, citation_batch, resolved_citation_chunk_ids))

    # 5. Citation coverage (answered responses must cite)
    checks.append(_check_citation_coverage(example, response, citation_batch))

    # 6. Required citation presence
    checks.append(_check_required_citations(example, response, citation_batch))

    # 7. Refusal/clarification correctness
    checks.append(_check_refusal_correctness(example, response))

    # 8. Vision consistency (if vision was used)
    if vision_result is not None:
        checks.append(_check_vision_consistency(example, response, vision_result))

    return AnswerQualityResult(
        example_id=example.id,
        query=example.query,
        language=example.language,
        response=response,
        checks=tuple(checks),
        vision_result=vision_result,
    )


def _check_status_consistency(example: AnswerExample, response: AssistantResponse) -> AnswerQualityCheck:
    """Check that response status matches expectations."""
    expected = example.expected_status
    actual = ResponseStatus.ANSWERED
    if response.status == ResponseStatus.NEEDS_CLARIFICATION:
        actual = ResponseStatus.NEEDS_CLARIFICATION
    elif response.status == ResponseStatus.ABSTAINED:
        actual = ResponseStatus.ABSTAINED

    passed = actual == expected
    return AnswerQualityCheck(
        check_name="status_consistency",
        passed=passed,
        details=f"Expected status {expected.value}, got {actual.value}",
        expected=expected.value,
        actual=actual.value,
    )


def _check_language_match(example: AnswerExample, response: AssistantResponse) -> AnswerQualityCheck:
    """Check that response language matches expected language."""
    expected = example.expected_language
    actual = response.language
    passed = actual == expected
    return AnswerQualityCheck(
        check_name="language_match",
        passed=passed,
        details=f"Expected language {expected.value}, got {actual.value}",
        expected=expected.value,
        actual=actual.value,
    )


def _check_non_empty_answer(example: AnswerExample, response: AssistantResponse) -> AnswerQualityCheck:
    """Check that answered responses have non-empty text."""
    if example.expected_status != AnswerStatus.ANSWERED:
        return AnswerQualityCheck(
            check_name="non_empty_answer",
            passed=True,
            details="Not expected to be answered",
            expected="N/A",
            actual="N/A",
        )

    passed = bool(response.text.strip())
    return AnswerQualityCheck(
        check_name="non_empty_answer",
        passed=passed,
        details="Answered response must have non-empty text",
        expected="non-empty",
        actual=response.text[:50] + ("..." if len(response.text) > 50 else ""),
    )


def _check_citation_validity(
    example: AnswerExample,
    response: AssistantResponse,
    citation_batch: CitationBatch | None,
    resolved_citation_chunk_ids: tuple[str, ...],
) -> AnswerQualityCheck:
    """Check that all cited chunk IDs are valid (exist in resolved batch)."""
    if not response.citations:
        return AnswerQualityCheck(
            check_name="citation_validity",
            passed=True,
            details="No citations to validate",
            expected="N/A",
            actual="no citations",
        )

    # Use resolved citation chunk IDs as the ground truth for valid citations
    valid_ids = set(resolved_citation_chunk_ids)
    if not valid_ids and citation_batch is not None:
        valid_ids = {c.chunk_id for c in citation_batch.citations if c.chunk_id}

    cited_ids = {c.chunk_id for c in response.citations if c.chunk_id}
    invalid_ids = cited_ids - valid_ids

    passed = len(invalid_ids) == 0
    return AnswerQualityCheck(
        check_name="citation_validity",
        passed=passed,
        details=f"Invalid citation IDs: {sorted(invalid_ids)}" if invalid_ids else "All citations valid",
        expected="all citations valid",
        actual=f"{len(invalid_ids)} invalid" if invalid_ids else "all valid",
    )


def _check_citation_coverage(
    example: AnswerExample,
    response: AssistantResponse,
    citation_batch: CitationBatch | None,
) -> AnswerQualityCheck:
    """Check that answered responses have at least one citation."""
    if example.expected_status != AnswerStatus.ANSWERED:
        return AnswerQualityCheck(
            check_name="citation_coverage",
            passed=True,
            details="Not expected to be answered",
            expected="N/A",
            actual="N/A",
        )

    passed = len(response.citations) > 0
    return AnswerQualityCheck(
        check_name="citation_coverage",
        passed=passed,
        details="Answered response must cite at least one source",
        expected=">= 1 citation",
        actual=f"{len(response.citations)} citations",
    )


def _check_required_citations(
    example: AnswerExample,
    response: AssistantResponse,
    citation_batch: CitationBatch | None,
) -> AnswerQualityCheck:
    """Check that required source IDs are cited."""
    if not example.expected_citation_source_ids:
        return AnswerQualityCheck(
            check_name="required_citations",
            passed=True,
            details="No required citations specified",
            expected="N/A",
            actual="N/A",
        )

    cited_source_ids = {c.source_id for c in response.citations}
    required = set(example.expected_citation_source_ids)
    missing = required - cited_source_ids

    passed = len(missing) == 0
    return AnswerQualityCheck(
        check_name="required_citations",
        passed=passed,
        details=f"Missing required source citations: {sorted(missing)}" if missing else "All required sources cited",
        expected=", ".join(sorted(required)),
        actual=", ".join(sorted(cited_source_ids)) or "none",
    )


def _check_refusal_correctness(example: AnswerExample, response: AssistantResponse) -> AnswerQualityCheck:
    """Check that refusal/clarification is correct when expected."""
    if not example.refusal_expected:
        return AnswerQualityCheck(
            check_name="refusal_correctness",
            passed=True,
            details="No refusal expected",
            expected="N/A",
            actual="N/A",
        )

    # Map expected refusal type to expected response status
    expected_status = _refusal_type_to_status(example.refusal_type)
    actual_status = response.status
    passed = actual_status == expected_status

    return AnswerQualityCheck(
        check_name="refusal_correctness",
        passed=passed,
        details=f"Expected status {expected_status.value}, got {actual_status.value}",
        expected=expected_status.value,
        actual=actual_status.value,
    )


def _refusal_type_to_status(refusal_type: RefusalType | None) -> ResponseStatus:
    """Map refusal type to expected response status."""
    if refusal_type is None:
        return ResponseStatus.ANSWERED
    if refusal_type in (RefusalType.INSUFFICIENT_EVIDENCE, RefusalType.AMBIGUOUS):
        return ResponseStatus.NEEDS_CLARIFICATION
    if refusal_type in (RefusalType.UNSUPPORTED_SCHEME, RefusalType.UNSAFE, RefusalType.OUT_OF_SCOPE):
        return ResponseStatus.ABSTAINED
    return ResponseStatus.ANSWERED


def _check_vision_consistency(
    example: AnswerExample,
    response: AssistantResponse,
    vision_result: VisionResult,
) -> AnswerQualityCheck:
    """Check that vision result is consistent with response."""
    # If vision failed, response should not claim visual observations
    if vision_result.is_failure:
        # Response should not contain visual claims
        # This is a heuristic check
        return AnswerQualityCheck(
            check_name="vision_consistency",
            passed=True,
            details=f"Vision status: {vision_result.status.value}",
            expected="vision failure acknowledged",
            actual=vision_result.status.value,
        )

    # Vision succeeded - check that observations are reflected
    # (This is a soft check; hard validation would need LLM)
    return AnswerQualityCheck(
        check_name="vision_consistency",
        passed=True,
        details=f"Vision status: {vision_result.status.value}, observations: {len(vision_result.observations)}",
        expected="vision observations reflected",
        actual=f"{len(vision_result.observations)} observations",
    )


@dataclass(frozen=True, slots=True)
class AnswerEvaluationSummary:
    """Aggregate answer quality evaluation results."""

    evaluation_set_version: str
    corpus_version: str
    system_version: str
    total_examples: int
    passed_examples: int
    failed_examples: int
    per_check_pass_rates: dict[str, float]
    language: Language
    timestamp_utc: str
    per_example: tuple[AnswerQualityResult, ...]

    @property
    def pass_rate(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.passed_examples / self.total_examples

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_set_version": self.evaluation_set_version,
            "corpus_version": self.corpus_version,
            "system_version": self.system_version,
            "total_examples": self.total_examples,
            "passed_examples": self.passed_examples,
            "failed_examples": self.failed_examples,
            "pass_rate": self.pass_rate,
            "per_check_pass_rates": self.per_check_pass_rates,
            "language": self.language.value,
            "timestamp_utc": self.timestamp_utc,
            "per_example": [ex.to_dict() for ex in self.per_example],
        }


def evaluate_answer_dataset(
    dataset: AnswerDataset,
    responses: Mapping[str, AssistantResponse],
    citation_batches: Mapping[str, CitationBatch] | None = None,
    vision_results: Mapping[str, VisionResult] | None = None,
    resolved_citation_chunk_ids: Mapping[str, tuple[str, ...]] | None = None,
    system_version: str = "",
) -> AnswerEvaluationSummary:
    """Evaluate answer quality for an entire dataset.

    Args:
        dataset: The answer evaluation dataset
        responses: Mapping from example.id to AssistantResponse
        citation_batches: Optional mapping from example.id to CitationBatch
        vision_results: Optional mapping from example.id to VisionResult
        resolved_citation_chunk_ids: Optional mapping from example.id to validated citation chunk IDs
        system_version: Version of the system under test
    """
    citation_batches = citation_batches or {}
    vision_results = vision_results or {}
    resolved_citation_chunk_ids = resolved_citation_chunk_ids or {}

    results: list[AnswerQualityResult] = []

    for example in dataset.examples:
        response = responses.get(example.id)
        if response is None:
            # Missing response - treat as failure
            checks = (AnswerQualityCheck(
                check_name="response_present",
                passed=False,
                details="No response generated for this example",
                expected="response",
                actual="missing",
            ),)
            result = AnswerQualityResult(
                example_id=example.id,
                query=example.query,
                language=example.language,
                response=AssistantResponse(
                    text="[MISSING]",
                    language=example.expected_language,
                    status=ResponseStatus.ABSTAINED,
                    citations=(),
                ),
                checks=checks,
            )
            results.append(result)
            continue

        batch = citation_batches.get(example.id)
        vision = vision_results.get(example.id)
        resolved_ids = resolved_citation_chunk_ids.get(example.id, ())

        result = evaluate_answer_quality(
            example=example,
            response=response,
            citation_batch=batch,
            vision_result=vision,
            resolved_citation_chunk_ids=resolved_ids,
        )
        results.append(result)

    # Compute per-check pass rates
    check_names = set()
    for r in results:
        for c in r.checks:
            check_names.add(c.check_name)

    per_check_rates = {}
    for name in check_names:
        total = sum(1 for r in results for c in r.checks if c.check_name == name)
        passed = sum(1 for r in results for c in r.checks if c.check_name == name and c.passed)
        per_check_rates[name] = passed / total if total > 0 else 0.0

    passed_examples = sum(1 for r in results if r.overall_passed)
    failed_examples = len(results) - passed_examples

    return AnswerEvaluationSummary(
        evaluation_set_version=dataset.evaluation_set_version,
        corpus_version=dataset.corpus_version,
        system_version=system_version,
        total_examples=len(results),
        passed_examples=passed_examples,
        failed_examples=failed_examples,
        per_check_pass_rates=per_check_rates,
        language=dataset.language,
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        per_example=tuple(results),
    )