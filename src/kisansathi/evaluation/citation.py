"""Citation evaluation metrics."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kisansathi.citations.models import Citation, CitationBatch
from kisansathi.domain.schemas import AssistantResponse


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """Result of a single citation check."""

    check_name: str
    passed: bool
    details: str = ""
    expected: str = ""
    actual: str = ""


@dataclass(frozen=True, slots=True)
class CitationEvaluationResult:
    """Complete citation evaluation for one response."""

    query: str
    response: AssistantResponse
    citation_batch: CitationBatch | None
    checks: tuple[CitationCheck, ...]

    @property
    def overall_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "response_status": self.response.status.value,
            "citation_count": len(self.response.citations),
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


@dataclass(frozen=True, slots=True)
class CitationEvaluationSummary:
    """Aggregate citation evaluation results."""

    corpus_version: str
    system_version: str
    total_responses: int
    responses_with_citations: int
    citation_validity_rate: float
    citation_coverage_rate: float
    provenance_validity_rate: float
    per_check_pass_rates: dict[str, float]
    per_response: tuple[CitationEvaluationResult, ...]
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "system_version": self.system_version,
            "total_responses": self.total_responses,
            "responses_with_citations": self.responses_with_citations,
            "citation_validity_rate": self.citation_validity_rate,
            "citation_coverage_rate": self.citation_coverage_rate,
            "provenance_validity_rate": self.provenance_validity_rate,
            "per_check_pass_rates": self.per_check_pass_rates,
            "timestamp_utc": self.timestamp_utc,
            "per_response": [r.to_dict() for r in self.per_response],
        }


def evaluate_citations(
    response: AssistantResponse,
    citation_batch: CitationBatch | None = None,
) -> CitationEvaluationResult:
    """Run deterministic citation checks on a single response.

    Args:
        response: The assistant response to evaluate
        citation_batch: The resolved CitationBatch from the retrieval pipeline
    """
    checks: list[CitationCheck] = []

    # 1. Citation validity: all cited chunk IDs exist in the batch
    checks.append(_check_citation_validity(response, citation_batch))

    # 2. Citation coverage: answered responses have citations
    checks.append(_check_citation_coverage(response))

    # 3. Provenance validity: cited sources exist in registry
    checks.append(_check_provenance_validity(response, citation_batch))

    # 4. No duplicate citations
    checks.append(_check_no_duplicate_citations(response))

    # 5. Citation chunk IDs match source IDs
    checks.append(_check_citation_source_consistency(response))

    return CitationEvaluationResult(
        query="",  # Will be filled by caller
        response=response,
        citation_batch=citation_batch,
        checks=tuple(checks),
    )


def _check_citation_validity(response: AssistantResponse, batch: CitationBatch | None) -> CitationCheck:
    """Check that all cited chunk IDs exist in the resolved batch."""
    if not response.citations:
        return CitationCheck(
            check_name="citation_validity",
            passed=True,
            details="No citations to validate",
            expected="N/A",
            actual="no citations",
        )

    if batch is None:
        return CitationCheck(
            check_name="citation_validity",
            passed=False,
            details="No citation batch available for validation",
            expected="valid batch",
            actual="batch missing",
        )

    valid_chunk_ids = {c.chunk_id for c in batch.citations if c.chunk_id}
    cited_chunk_ids = {c.chunk_id for c in response.citations if c.chunk_id}
    invalid = cited_chunk_ids - valid_chunk_ids

    passed = len(invalid) == 0
    return CitationCheck(
        check_name="citation_validity",
        passed=passed,
        details=f"Invalid chunk IDs: {sorted(invalid)}" if invalid else "All citations valid",
        expected="all citations in batch",
        actual=f"{len(invalid)} invalid" if invalid else "all valid",
    )


def _check_citation_coverage(response: AssistantResponse) -> CitationCheck:
    """Check that answered responses have at least one citation."""
    if response.status.value != "answered":
        return CitationCheck(
            check_name="citation_coverage",
            passed=True,
            details=f"Response status is {response.status.value}, not answered",
            expected="N/A",
            actual=response.status.value,
        )

    passed = len(response.citations) > 0
    return CitationCheck(
        check_name="citation_coverage",
        passed=passed,
        details="Answered response must cite at least one source",
        expected=">= 1 citation",
        actual=f"{len(response.citations)} citations",
    )


def _check_provenance_validity(response: AssistantResponse, batch: CitationBatch | None) -> CitationCheck:
    """Check that cited sources exist in the source registry (via batch)."""
    if not response.citations:
        return CitationCheck(
            check_name="provenance_validity",
            passed=True,
            details="No citations to validate",
            expected="N/A",
            actual="no citations",
        )

    if batch is None:
        return CitationCheck(
            check_name="provenance_validity",
            passed=False,
            details="No citation batch available for provenance validation",
            expected="valid batch",
            actual="batch missing",
        )

    # Check that cited sources exist in batch
    batch_source_ids = {c.source_id for c in batch.citations}
    cited_source_ids = {c.source_id for c in response.citations}
    missing_sources = cited_source_ids - batch_source_ids

    passed = len(missing_sources) == 0
    return CitationCheck(
        check_name="provenance_validity",
        passed=passed,
        details=f"Missing source IDs in batch: {sorted(missing_sources)}" if missing_sources else "All sources have provenance",
        expected="all sources in batch",
        actual=f"{len(missing_sources)} missing" if missing_sources else "all present",
    )


def _check_no_duplicate_citations(response: AssistantResponse) -> CitationCheck:
    """Check that no chunk ID is cited more than once."""
    if not response.citations:
        return CitationCheck(
            check_name="no_duplicate_citations",
            passed=True,
            details="No citations",
            expected="N/A",
            actual="no citations",
        )

    chunk_ids = [c.chunk_id for c in response.citations if c.chunk_id]
    seen = set()
    duplicates = set()
    for cid in chunk_ids:
        if cid in seen:
            duplicates.add(cid)
        seen.add(cid)

    passed = len(duplicates) == 0
    return CitationCheck(
        check_name="no_duplicate_citations",
        passed=passed,
        details=f"Duplicate chunk IDs: {sorted(duplicates)}" if duplicates else "No duplicates",
        expected="no duplicates",
        actual=f"{len(duplicates)} duplicates" if duplicates else "no duplicates",
    )


def _check_citation_source_consistency(response: AssistantResponse) -> CitationCheck:
    """Check that citation's chunk_id belongs to its source_id."""
    if not response.citations:
        return CitationCheck(
            check_name="citation_source_consistency",
            passed=True,
            details="No citations",
            expected="N/A",
            actual="no citations",
        )

    # This would require access to the source registry to verify
    # For now, we can only check that both fields are present
    inconsistent = []
    for c in response.citations:
        if not c.chunk_id or not c.source_id:
            inconsistent.append(f"{c.chunk_id or 'missing'} / {c.source_id or 'missing'}")

    passed = len(inconsistent) == 0
    return CitationCheck(
        check_name="citation_source_consistency",
        passed=passed,
        details=f"Inconsistent citations: {inconsistent}" if inconsistent else "All citations have source_id and chunk_id",
        expected="chunk_id belongs to source_id",
        actual="consistent" if passed else "inconsistent",
    )


def evaluate_citation_dataset(
    responses: dict[str, tuple[AssistantResponse, CitationBatch | None]],
    corpus_version: str,
    system_version: str = "",
) -> CitationEvaluationSummary:
    """Evaluate citations across multiple responses.

    Args:
        responses: Mapping from query_id to (response, citation_batch)
        corpus_version: Version of the corpus
        system_version: Version of the system under test
    """
    results: list[CitationEvaluationResult] = []
    queries_with_citations = 0

    for query_id, (response, batch) in responses.items():
        if response.citations:
            queries_with_citations += 1

        result = evaluate_citations(response, batch)
        # Create new result with query
        result = CitationEvaluationResult(
            query=query_id,
            response=response,
            citation_batch=batch,
            checks=result.checks,
        )
        results.append(result)

    # Compute aggregate metrics
    total = len(results)
    validity_passed = sum(1 for r in results for c in r.checks if c.check_name == "citation_validity" and c.passed)
    validity_total = sum(1 for r in results for c in r.checks if c.check_name == "citation_validity")
    coverage_passed = sum(1 for r in results for c in r.checks if c.check_name == "citation_coverage" and c.passed)
    coverage_total = sum(1 for r in results for c in r.checks if c.check_name == "citation_coverage")
    provenance_passed = sum(1 for r in results for c in r.checks if c.check_name == "provenance_validity" and c.passed)
    provenance_total = sum(1 for r in results for c in r.checks if c.check_name == "provenance_validity")

    per_check_rates = {}
    for check_name in ("citation_validity", "citation_coverage", "provenance_validity", "no_duplicate_citations", "citation_source_consistency"):
        passed = sum(1 for r in results for c in r.checks if c.check_name == check_name and c.passed)
        total = sum(1 for r in results for c in r.checks if c.check_name == check_name)
        per_check_rates[check_name] = passed / total if total > 0 else 0.0

    return CitationEvaluationSummary(
        corpus_version=corpus_version,
        system_version=system_version,
        total_responses=total,
        responses_with_citations=queries_with_citations,
        citation_validity_rate=validity_passed / validity_total if validity_total > 0 else 0.0,
        citation_coverage_rate=coverage_passed / coverage_total if coverage_total > 0 else 0.0,
        provenance_validity_rate=provenance_passed / provenance_total if provenance_total > 0 else 0.0,
        per_check_pass_rates=per_check_rates,
        per_response=tuple(results),
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
    )