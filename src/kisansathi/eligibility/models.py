"""Typed contracts for deterministic scheme eligibility evaluation.

Eligibility is decided by evaluating explicit, ordered rules over caller-supplied
facts. Nothing in this module infers, defaults, or otherwise guesses a fact, and no
LLM, retrieval, or network access is involved.
"""

import re
from dataclasses import dataclass, fields
from enum import StrEnum

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_ID = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*):([0-9a-f]{24})$")
_RULE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class EligibilityError(ValueError):
    """Raised when an eligibility request or rule set is malformed."""


class EligibilityStatus(StrEnum):
    """The deterministic outcome of evaluating a request against a rule set."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    UNSUPPORTED_SCHEME = "unsupported_scheme"


class RuleKind(StrEnum):
    """The closed set of conditions the evaluator knows how to test.

    A deliberately tiny vocabulary. Anything expressible outside these two kinds is
    treated as unsupported rather than approximated.
    """

    MUST_BE_TRUE = "must_be_true"
    MUST_BE_FALSE = "must_be_false"

    @property
    def expected(self) -> bool:
        """Return the value a fact must hold for the rule to be satisfied."""
        return self is RuleKind.MUST_BE_TRUE


@dataclass(frozen=True, slots=True)
class EligibilityRequest:
    """Structured facts supplied by the caller about one potential beneficiary.

    Every fact is tri-state. ``None`` means "not supplied" and is never treated as a
    value; it is reported back as a missing fact. Fields are limited to conditions the
    ingested official corpus actually supports.
    """

    scheme: str
    landholding_in_own_name: bool | None = None
    land_is_cultivable: bool | None = None
    land_used_for_non_agricultural_purpose: bool | None = None
    family_member_paid_income_tax_last_assessment_year: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, str) or not self.scheme.strip():
            raise EligibilityError("EligibilityRequest scheme must be a non-empty string")
        for fact in FACT_NAMES:
            value = getattr(self, fact)
            if value is None:
                continue
            if not isinstance(value, bool):
                raise EligibilityError(f"EligibilityRequest {fact} must be True, False, or None")

    @property
    def normalized_scheme(self) -> str:
        """Return the scheme identifier trimmed and case-folded for comparison."""
        return self.scheme.strip().casefold()

    def value_for(self, fact: str) -> bool | None:
        """Return the supplied value for ``fact``, or None when it was not supplied."""
        if fact not in FACT_NAMES:
            raise EligibilityError(f"unknown eligibility fact: {fact!r}")
        return getattr(self, fact)

    def missing_facts(self) -> tuple[str, ...]:
        """Return the names of every fact the caller did not supply, in field order."""
        return tuple(fact for fact in FACT_NAMES if getattr(self, fact) is None)


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A pointer to the exact ingested corpus chunk that supports a rule.

    Field names deliberately mirror ``retrieval.vector_store.VectorPayload`` so a rule's
    evidence can later be joined onto the retrieval citation layer mechanically.
    """

    source_id: str
    sha256: str
    chunk_id: str
    page_start: int
    page_end: int
    locator: str

    def __post_init__(self) -> None:
        for name in ("source_id", "chunk_id", "locator"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EligibilityError(f"EvidenceRef {name} must be a non-empty string")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise EligibilityError("EvidenceRef sha256 must be 64 lowercase hexadecimal characters")
        match = _CHUNK_ID.fullmatch(self.chunk_id)
        if match is None:
            raise EligibilityError("EvidenceRef chunk_id must be '<source_id>:<24 hex>'")
        if match.group(1) != self.source_id:
            raise EligibilityError("EvidenceRef chunk_id must belong to source_id")
        for name in ("page_start", "page_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EligibilityError(f"EvidenceRef {name} must be a positive integer")
        if self.page_end < self.page_start:
            raise EligibilityError("EvidenceRef page_end must not be before page_start")


@dataclass(frozen=True, slots=True)
class EligibilityRule:
    """One supported condition, with the corpus text that justifies it.

    ``verbatim_excerpt`` is an exact substring of the referenced chunk's ``text``,
    copied without correction. OCR damage in the source is preserved rather than
    repaired, so the excerpt stays auditable against the real document.
    """

    rule_id: str
    kind: RuleKind
    fact: str
    summary: str
    verbatim_excerpt: str
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not _RULE_ID.fullmatch(self.rule_id):
            raise EligibilityError("EligibilityRule rule_id must be a non-empty path-safe slug")
        if not isinstance(self.kind, RuleKind):
            raise EligibilityError("EligibilityRule kind must be a RuleKind")
        if self.fact not in FACT_NAMES:
            raise EligibilityError(f"EligibilityRule references an unknown fact: {self.fact!r}")
        for name in ("summary", "verbatim_excerpt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EligibilityError(f"EligibilityRule {name} must be a non-empty string")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise EligibilityError("EligibilityRule must cite at least one EvidenceRef")
        for reference in self.evidence:
            if not isinstance(reference, EvidenceRef):
                raise EligibilityError("EligibilityRule evidence must contain EvidenceRef objects")

    @property
    def expected(self) -> bool:
        """Return the value the fact must hold for this rule to be satisfied."""
        return self.kind.expected


@dataclass(frozen=True, slots=True)
class RuleSet:
    """An ordered, versioned set of rules for exactly one scheme.

    Order is significant: the evaluator reports the first violated rule in declaration
    order, which is what makes a decision reason reproducible.
    """

    scheme: str
    rule_set_version: str
    corpus_version: str
    rules: tuple[EligibilityRule, ...]

    def __post_init__(self) -> None:
        for name in ("scheme", "rule_set_version", "corpus_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EligibilityError(f"RuleSet {name} must be a non-empty string")
        if not isinstance(self.rules, tuple) or not self.rules:
            raise EligibilityError("RuleSet must contain at least one rule")
        for rule in self.rules:
            if not isinstance(rule, EligibilityRule):
                raise EligibilityError("RuleSet rules must contain EligibilityRule objects")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise EligibilityError("RuleSet rule_id values must be unique")


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """The deterministic result of evaluating one request against one rule set.

    ``evidence`` carries the provenance of the rule(s) that produced the verdict: the
    violated rule for ``INELIGIBLE``, every satisfied rule for ``ELIGIBLE``, and none
    for ``INSUFFICIENT_INFORMATION`` or ``UNSUPPORTED_SCHEME``, where no rule decided.
    """

    scheme: str
    status: EligibilityStatus
    rule_id: str | None = None
    summary: str = ""
    missing_facts: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    evaluated_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, str) or not self.scheme.strip():
            raise EligibilityError("EligibilityDecision scheme must be a non-empty string")
        if not isinstance(self.status, EligibilityStatus):
            raise EligibilityError("EligibilityDecision status must be an EligibilityStatus")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise EligibilityError("EligibilityDecision summary must be a non-empty string")
        if self.rule_id is not None:
            if not isinstance(self.rule_id, str) or not self.rule_id.strip():
                raise EligibilityError(
                    "EligibilityDecision rule_id must be a non-empty string or None"
                )
            if self.status in (EligibilityStatus.UNSUPPORTED_SCHEME,):
                raise EligibilityError("unsupported decisions must not name a rule")
        for name in ("missing_facts", "evaluated_rules"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise EligibilityError(f"EligibilityDecision {name} must be a tuple")
        if self.status is EligibilityStatus.INSUFFICIENT_INFORMATION and not self.missing_facts:
            raise EligibilityError("insufficient_information must name the missing facts")
        if self.status is EligibilityStatus.ELIGIBLE and self.missing_facts:
            raise EligibilityError("an eligible decision cannot have missing facts")


FACT_NAMES: tuple[str, ...] = tuple(
    field.name
    for field in fields(EligibilityRequest)
    if field.name != "scheme"
)