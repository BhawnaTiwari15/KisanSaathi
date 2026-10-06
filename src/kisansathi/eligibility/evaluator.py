"""Deterministic, dependency-free eligibility evaluation.

``evaluate`` is a pure function. It reads only the request and the rule set, performs no
retrieval, no I/O, and no model calls, and imports nothing from LangGraph or the
retrieval stack. The same request always yields an identical decision.

Precedence, highest first:

1. ``UNSUPPORTED_SCHEME``   - the request names a scheme with no rule set.
2. ``INELIGIBLE``           - a known exclusion holds, whatever else is unknown.
3. ``INSUFFICIENT_INFORMATION`` - a required fact was not supplied.
4. ``ELIGIBLE``             - every rule is satisfied.

A known exclusion is conclusive on its own, so it outranks missing facts: a caller who
has already failed one condition does not need to answer everything else to be told
they are not eligible.
"""

from kisansathi.eligibility.models import (
    EligibilityDecision,
    EligibilityRequest,
    EligibilityRule,
    EligibilityStatus,
    EvidenceRef,
    RuleSet,
)
from kisansathi.eligibility.pm_kisan_rules import PM_KISAN, PM_KISAN_SCHEME

_ELIGIBLE_SUMMARY = "All supported {scheme} conditions are satisfied."
_UNSUPPORTED_SUMMARY = (
    "No eligibility rule set exists for {scheme}; this tool supports {supported} only."
)


def evaluate(request: EligibilityRequest, *, rules: RuleSet | None = None) -> EligibilityDecision:
    """Evaluate ``request`` against ``rules`` and return a deterministic decision.

    Every rule is inspected so that the reported missing facts are complete; the
    evaluation is not short-circuited on the first unknown fact.
    """
    rule_set = PM_KISAN if rules is None else rules

    if request.normalized_scheme != rule_set.scheme.strip().casefold():
        return EligibilityDecision(
            scheme=request.scheme,
            status=EligibilityStatus.UNSUPPORTED_SCHEME,
            summary=_UNSUPPORTED_SUMMARY.format(
                scheme=request.scheme.strip(),
                supported=PM_KISAN_SCHEME,
            ),
        )

    violated = None
    missing: list[str] = []
    evaluated: list[str] = []

    for rule in rule_set.rules:
        value = request.value_for(rule.fact)
        if value is None:
            if rule.fact not in missing:
                missing.append(rule.fact)
            continue
        evaluated.append(rule.rule_id)
        if violated is None and value != rule.expected:
            violated = rule

    if violated is not None:
        return EligibilityDecision(
            scheme=request.scheme,
            status=EligibilityStatus.INELIGIBLE,
            rule_id=violated.rule_id,
            summary=violated.summary,
            missing_facts=tuple(missing),
            evidence=violated.evidence,
            evaluated_rules=tuple(evaluated),
        )

    if missing:
        return EligibilityDecision(
            scheme=request.scheme,
            status=EligibilityStatus.INSUFFICIENT_INFORMATION,
            summary=(
                "Eligibility cannot be determined yet. "
                f"{len(missing)} required fact(s) are still unknown."
            ),
            missing_facts=tuple(missing),
            evaluated_rules=tuple(evaluated),
        )

    return EligibilityDecision(
        scheme=request.scheme,
        status=EligibilityStatus.ELIGIBLE,
        summary=_ELIGIBLE_SUMMARY.format(scheme=rule_set.scheme),
        evidence=_evidence_for(rule_set.rules),
        evaluated_rules=tuple(evaluated),
    )


def _evidence_for(rules: tuple[EligibilityRule, ...]) -> tuple[EvidenceRef, ...]:
    """Collect the provenance of every rule that contributed to an eligible verdict."""
    collected: list[EvidenceRef] = []
    seen: set[str] = set()
    for rule in rules:
        for reference in rule.evidence:
            if reference.chunk_id not in seen:
                seen.add(reference.chunk_id)
                collected.append(reference)
    return tuple(collected)
