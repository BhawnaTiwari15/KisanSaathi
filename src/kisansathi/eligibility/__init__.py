"""Deterministic PM-KISAN scheme eligibility evaluation.

Kept independent from LangGraph and retrieval for this milestone: ``evaluate`` is a pure
function over a request and a versioned rule set, with no I/O and no model calls.
"""

from kisansathi.eligibility.evaluator import evaluate
from kisansathi.eligibility.models import (
    FACT_NAMES,
    EligibilityDecision,
    EligibilityError,
    EligibilityRequest,
    EligibilityRule,
    EligibilityStatus,
    EvidenceRef,
    RuleKind,
    RuleSet,
)
from kisansathi.eligibility.pm_kisan_rules import (
    CORPUS_VERSION,
    ELIGIBILITY_BASIS,
    PM_KISAN,
    PM_KISAN_RULES,
    PM_KISAN_SCHEME,
    RULE_SET_VERSION,
    UNSUPPORTED_CONDITIONS,
    UnsupportedCondition,
)

__all__ = [
    "CORPUS_VERSION",
    "ELIGIBILITY_BASIS",
    "FACT_NAMES",
    "PM_KISAN",
    "PM_KISAN_RULES",
    "PM_KISAN_SCHEME",
    "RULE_SET_VERSION",
    "UNSUPPORTED_CONDITIONS",
    "EligibilityDecision",
    "EligibilityError",
    "EligibilityRequest",
    "EligibilityRule",
    "EligibilityStatus",
    "EvidenceRef",
    "RuleKind",
    "RuleSet",
    "UnsupportedCondition",
    "evaluate",
]
