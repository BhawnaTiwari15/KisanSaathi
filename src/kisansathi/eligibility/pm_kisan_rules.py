"""PM-KISAN eligibility rules, each traceable to the ingested official FAQ corpus.

Every rule below cites an exact substring of a real chunk's ``text`` from
``data/processed/v1/pm-kisan-revised-faq/95d4e289...jsonl``. Excerpts are copied
verbatim, including OCR damage, so they remain auditable against the source.

Conditions that the corpus does not support are recorded in
``UNSUPPORTED_CONDITIONS`` with the reason, rather than being approximated from general
knowledge. They are documentation, not evaluation input: the evaluator never reads
them.
"""

from dataclasses import dataclass

from kisansathi.eligibility.models import (
    EvidenceRef,
    EligibilityRule,
    FACT_NAMES,
    RuleKind,
    RuleSet,
)

PM_KISAN_SCHEME = "PM-KISAN"
RULE_SET_VERSION = "pm-kisan-v1"

SOURCE_ID = "pm-kisan-revised-faq"
CORPUS_SHA256 = "95d4e2892b4b351e599bf903f8c6f89b20d4b74831eb8f1a0e61a2a66771ff86"
CORPUS_VERSION = f"processed-v1:{SOURCE_ID}@{CORPUS_SHA256}"


def _evidence(short_hash: str, page_start: int, page_end: int, locator: str) -> EvidenceRef:
    """Build an EvidenceRef pinned to the PM-KISAN corpus for one chunk."""
    return EvidenceRef(
        source_id=SOURCE_ID,
        sha256=CORPUS_SHA256,
        chunk_id=f"{SOURCE_ID}:{short_hash}",
        page_start=page_start,
        page_end=page_end,
        locator=locator,
    )


OWN_NAME = EligibilityRule(
    rule_id="pm-kisan-land-in-own-name",
    kind=RuleKind.MUST_BE_TRUE,
    fact="landholding_in_own_name",
    summary="The applicant must hold the land in their own name.",
    verbatim_excerpt="Land holding is the sole criteria to avail the benefit under the Scheme",
    evidence=(_evidence("33ff2db494c59ab1ca6c4f76", 4, 4, "Revised FAQ, Q18"),),
)

CULTIVABLE_LAND = EligibilityRule(
    rule_id="pm-kisan-land-must-be-cultivable",
    kind=RuleKind.MUST_BE_TRUE,
    fact="land_is_cultivable",
    summary="The landholding must be cultivable.",
    verbatim_excerpt=(
        "Micro land holdings, which are not cultivable, are excluded from the benefit "
        "under the scheme."
    ),
    evidence=(_evidence("b148dd12de05914c6bfaf324", 7, 7, "Revised FAQ, Q30"),),
)

AGRICULTURAL_USE = EligibilityRule(
    rule_id="pm-kisan-land-must-be-agricultural",
    kind=RuleKind.MUST_BE_FALSE,
    fact="land_used_for_non_agricultural_purpose",
    summary="Agricultural land used for non-agricultural purposes is not covered.",
    verbatim_excerpt=(
        "Agricultural land being used for non-agricultural purposes will not be covered "
        "for benefit under the scheme."
    ),
    evidence=(_evidence("f39a38edb9e1352d1315ece3", 7, 7, "Revised FAQ, Q33"),),
)

NO_FAMILY_INCOME_TAX = EligibilityRule(
    rule_id="pm-kisan-no-family-income-tax",
    kind=RuleKind.MUST_BE_FALSE,
    fact="family_member_paid_income_tax_last_assessment_year",
    summary=(
        "No member of the applicant's family paid income tax in the last assessment year."
    ),
    verbatim_excerpt="family is not eligible for benefit under the scheme",
    evidence=(_evidence("1fe005b874ba210ad336e5d9", 3, 4, "Revised FAQ, Q15"),),
)

# Declaration order is significant: the evaluator reports the first violated rule in
# this order, which is what makes the reported reason reproducible.
PM_KISAN_RULES: tuple[EligibilityRule, ...] = (
    OWN_NAME,
    CULTIVABLE_LAND,
    AGRICULTURAL_USE,
    NO_FAMILY_INCOME_TAX,
)

# The positive basis of the scheme, recorded for provenance and documentation only.
# It states the same conjunction the rules above test, so it is not a separate
# evaluated condition and must not be counted as a fifth rule.
ELIGIBILITY_BASIS = EligibilityRule(
    rule_id="pm-kisan-eligibility-basis",
    kind=RuleKind.MUST_BE_TRUE,
    fact="landholding_in_own_name",
    summary="Cultivable land in the family's names establishes eligibility.",
    verbatim_excerpt=(
        "All landholding farmers' families, which have cultivable landholding in their "
        "names are eligible to get benefit under the scheme"
    ),
    evidence=(_evidence("75ee388fe9c50b9419d6aea2", 1, 1, "Revised FAQ, Q6"),),
)

PM_KISAN = RuleSet(
    scheme=PM_KISAN_SCHEME,
    rule_set_version=RULE_SET_VERSION,
    corpus_version=CORPUS_VERSION,
    rules=PM_KISAN_RULES,
)


@dataclass(frozen=True, slots=True)
class UnsupportedCondition:
    """A condition deliberately left out of the rule set, and why.

    Documented so that a future change is a visible, reviewable decision instead of a
    silent gap. Nothing here is inferred or approximated.
    """

    condition_id: str
    reason: str
    locator: str


UNSUPPORTED_CONDITIONS: tuple[UnsupportedCondition, ...] = (
    UnsupportedCondition(
        condition_id="land-size-limit",
        reason=(
            "No land-size condition exists to evaluate. The FAQ states the scheme was "
            "extended to all farmer families irrespective of the size of their holdings, "
            "so the widely assumed two-hectare cap is contradicted by the source."
        ),
        locator="Revised FAQ, Q2 and Q10",
    ),
    UnsupportedCondition(
        condition_id="age",
        reason=(
            "Age appears only as a mandatory enrollment data field, not an eligibility "
            "condition."
        ),
        locator="Revised FAQ, Q20",
    ),
    UnsupportedCondition(
        condition_id="gender",
        reason=(
            "Gender appears only as a mandatory enrollment data field, not an eligibility "
            "condition."
        ),
        locator="Revised FAQ, Q20",
    ),
    UnsupportedCondition(
        condition_id="category",
        reason=(
            "Social category appears only as an enrollment data field; the corpus states no "
            "category-based eligibility condition."
        ),
        locator="Revised FAQ, Q20",
    ),
    UnsupportedCondition(
        condition_id="occupation",
        reason="The corpus contains no occupation-based eligibility condition for PM-KISAN.",
        locator="Revised FAQ, whole document",
    ),
    UnsupportedCondition(
        condition_id="income-or-bpl",
        reason=(
            "'Income' appears only in the scheme's purpose statement; the corpus states no "
            "income or BPL eligibility threshold."
        ),
        locator="Revised FAQ, purpose statement",
    ),
    UnsupportedCondition(
        condition_id="aadhaar",
        reason=(
            "Aadhaar is presented as an enrollment and payment requirement rather than an "
            "eligibility condition, and the FAQ carries conflicting dates for the exemptions."
        ),
        locator="Revised FAQ, Q20 and Q27",
    ),
    UnsupportedCondition(
        condition_id="bank-account",
        reason="Bank account details are a payment prerequisite, not an eligibility condition.",
        locator="Revised FAQ, Q26",
    ),
    UnsupportedCondition(
        condition_id="government-employment",
        reason=(
            "The carve-out for serving or retired government staff continues across a page "
            "break and is OCR-damaged beyond reliable quotation, so the condition cannot be "
            "evaluated from the ingested text."
        ),
        locator="Revised FAQ, Q9",
    ),
    UnsupportedCondition(
        condition_id="pension-threshold",
        reason=(
            "The monthly pension threshold was laid out in a two-column table that "
            "extraction flattened out of reading order; clause numbering is lost and the "
            "orphaned fragments cannot be attributed to a condition safely."
        ),
        locator="Revised FAQ, exclusion list",
    ),
    UnsupportedCondition(
        condition_id="institutional-land-holders",
        reason="Present only in a leaked chunk heading field and never in chunk text.",
        locator="Revised FAQ, exclusion list (a)",
    ),
    UnsupportedCondition(
        condition_id="constitutional-posts",
        reason="Present only in a leaked chunk heading field and never in chunk text.",
        locator="Revised FAQ, exclusion list (i)",
    ),
    UnsupportedCondition(
        condition_id="former-ministers",
        reason="Present only in a leaked chunk heading field and never in chunk text.",
        locator="Revised FAQ, exclusion list (ii)",
    ),
    UnsupportedCondition(
        condition_id="professionals",
        reason="Present only in a leaked chunk heading field and never in chunk text.",
        locator="Revised FAQ, exclusion list (vi)",
    ),
    UnsupportedCondition(
        condition_id="cutoff-and-succession",
        reason=(
            "The cut-off date and the five-year window are date-dependent and require a "
            "supplied reference date; none is provided, and the window stated in the source "
            "has already elapsed."
        ),
        locator="Revised FAQ, Q12 to Q14",
    ),
    UnsupportedCondition(
        condition_id="tenant-farmer",
        reason=(
            "Not an independent condition. It is expressed by the caller as "
            "landholding_in_own_name=False, which the own-name rule already covers."
        ),
        locator="Revised FAQ, Q18",
    ),
)


SUPPORTED_FACT_NAMES: frozenset[str] = frozenset(FACT_NAMES)
