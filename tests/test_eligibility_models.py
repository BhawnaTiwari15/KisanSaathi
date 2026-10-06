"""Validation and immutability tests for the eligibility contracts.

Fixtures are fixed literals so nothing here depends on the corpus or the environment.
"""

import unittest

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

SHA = "a" * 64
SOURCE_ID = "pm-kisan-revised-faq"
CHUNK_ID = f"{SOURCE_ID}:0123456789abcdef01234567"


def make_evidence(**overrides: object) -> EvidenceRef:
    """Return a valid EvidenceRef, overriding any field under test."""
    values: dict[str, object] = {
        "source_id": SOURCE_ID,
        "sha256": SHA,
        "chunk_id": CHUNK_ID,
        "page_start": 4,
        "page_end": 4,
        "locator": "Revised FAQ, Q18",
    }
    values.update(overrides)
    return EvidenceRef(**values)  # type: ignore[arg-type]


def make_rule(**overrides: object) -> EligibilityRule:
    """Return a valid EligibilityRule, overriding any field under test."""
    values: dict[str, object] = {
        "rule_id": "test-rule",
        "kind": RuleKind.MUST_BE_TRUE,
        "fact": "landholding_in_own_name",
        "summary": "summary",
        "verbatim_excerpt": "excerpt",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return EligibilityRule(**values)  # type: ignore[arg-type]


def make_rule_set(rules: tuple[EligibilityRule, ...] | None = None) -> RuleSet:
    """Return a valid RuleSet."""
    return RuleSet(
        scheme="PM-KISAN",
        rule_set_version="test-v1",
        corpus_version="processed-v1:test",
        rules=rules if rules is not None else (make_rule(),),
    )


class RuleKindTests(unittest.TestCase):
    def test_must_be_true_expects_true(self) -> None:
        self.assertIs(RuleKind.MUST_BE_TRUE.expected, True)

    def test_must_be_false_expects_false(self) -> None:
        self.assertIs(RuleKind.MUST_BE_FALSE.expected, False)

    def test_vocabulary_is_closed_to_two_kinds(self) -> None:
        self.assertEqual(
            {kind.value for kind in RuleKind},
            {"must_be_true", "must_be_false"},
        )


class EligibilityRequestTests(unittest.TestCase):
    def test_fact_defaults_are_unknown_not_false(self) -> None:
        request = EligibilityRequest("PM-KISAN")

        for fact in FACT_NAMES:
            self.assertIsNone(request.value_for(fact), fact)

    def test_missing_facts_lists_every_unknown_fact(self) -> None:
        request = EligibilityRequest("PM-KISAN", land_is_cultivable=True)

        self.assertNotIn("land_is_cultivable", request.missing_facts())
        self.assertEqual(len(request.missing_facts()), len(FACT_NAMES) - 1)

    def test_known_fact_values_round_trip(self) -> None:
        request = EligibilityRequest(
            "PM-KISAN",
            landholding_in_own_name=True,
            land_is_cultivable=False,
        )

        self.assertIs(request.value_for("landholding_in_own_name"), True)
        self.assertIs(request.value_for("land_is_cultivable"), False)

    def test_fact_set_is_exactly_the_supported_conditions(self) -> None:
        self.assertEqual(
            FACT_NAMES,
            (
                "landholding_in_own_name",
                "land_is_cultivable",
                "land_used_for_non_agricultural_purpose",
                "family_member_paid_income_tax_last_assessment_year",
            ),
        )

    def test_scheme_is_not_a_fact(self) -> None:
        self.assertNotIn("scheme", FACT_NAMES)

    def test_normalized_scheme_trims_and_casefolds(self) -> None:
        self.assertEqual(
            EligibilityRequest("  pm-Kisan  ").normalized_scheme,
            "pm-kisan",
        )

    def test_blank_scheme_is_rejected(self) -> None:
        for scheme in ("", "   "):
            with self.subTest(scheme=scheme):
                with self.assertRaises(EligibilityError):
                    EligibilityRequest(scheme)

    def test_non_string_scheme_is_rejected(self) -> None:
        with self.assertRaises(EligibilityError):
            EligibilityRequest(None)  # type: ignore[arg-type]

    def test_non_boolean_fact_is_rejected(self) -> None:
        for fact in FACT_NAMES:
            for value in ("yes", 1, 0, [], object()):
                with self.subTest(fact=fact, value=value):
                    with self.assertRaises(EligibilityError):
                        EligibilityRequest("PM-KISAN", **{fact: value})

    def test_truthy_and_falsy_non_booleans_are_still_rejected(self) -> None:
        for fact in FACT_NAMES:
            for value in ("", "no", 0.0):
                with self.subTest(fact=fact, value=value):
                    with self.assertRaises(EligibilityError):
                        EligibilityRequest("PM-KISAN", **{fact: value})

    def test_unknown_fact_lookup_is_rejected(self) -> None:
        with self.assertRaises(EligibilityError):
            EligibilityRequest("PM-KISAN").value_for("land_area_hectares")

    def test_request_is_frozen(self) -> None:
        request = EligibilityRequest("PM-KISAN")

        with self.assertRaises(AttributeError):
            request.scheme = "other"  # type: ignore[misc]

    def test_equality_is_value_based(self) -> None:
        self.assertEqual(EligibilityRequest("PM-KISAN"), EligibilityRequest("PM-KISAN"))


class EvidenceRefTests(unittest.TestCase):
    def test_blank_identifier_fields_are_rejected(self) -> None:
        for field in ("source_id", "chunk_id", "locator"):
            with self.subTest(field=field):
                with self.assertRaises(EligibilityError):
                    make_evidence(**{field: "   "})

    def test_malformed_sha256_is_rejected(self) -> None:
        for sha in ("A" * 64, "a" * 63, "z" * 64, "abc"):
            with self.subTest(sha=sha):
                with self.assertRaises(EligibilityError):
                    make_evidence(sha256=sha)

    def test_malformed_chunk_id_is_rejected(self) -> None:
        for chunk_id in (f"{SOURCE_ID}:short", f"{SOURCE_ID}:" + "a" * 23, "no-source-prefix"):
            with self.subTest(chunk_id=chunk_id):
                with self.assertRaises(EligibilityError):
                    make_evidence(chunk_id=chunk_id)

    def test_chunk_id_must_match_source_id(self) -> None:
        with self.assertRaises(EligibilityError):
            make_evidence(chunk_id="other-source:0123456789abcdef01234567")

    def test_non_positive_pages_are_rejected(self) -> None:
        for page in (0, -1):
            with self.subTest(page=page):
                with self.assertRaises(EligibilityError):
                    make_evidence(page_start=page)

    def test_booleans_are_not_accepted_as_pages(self) -> None:
        with self.assertRaises(EligibilityError):
            make_evidence(page_start=True)

    def test_reversed_page_range_is_rejected(self) -> None:
        with self.assertRaises(EligibilityError):
            make_evidence(page_start=5, page_end=4)

    def test_multi_page_range_is_allowed(self) -> None:
        self.assertEqual(make_evidence(page_start=3, page_end=4).page_end, 4)

    def test_evidence_is_frozen(self) -> None:
        with self.assertRaises(AttributeError):
            make_evidence().page_start = 9  # type: ignore[misc]


class EligibilityRuleTests(unittest.TestCase):
    def test_malformed_rule_id_is_rejected(self) -> None:
        for rule_id in ("", " ", "has space", "-leading", "slash/id"):
            with self.subTest(rule_id=rule_id):
                with self.assertRaises(EligibilityError):
                    make_rule(rule_id=rule_id)

    def test_rule_referencing_an_unknown_fact_is_rejected(self) -> None:
        with self.assertRaises(EligibilityError):
            make_rule(fact="land_area_hectares")

    def test_rule_must_have_evidence(self) -> None:
        with self.assertRaises(EligibilityError):
            make_rule(evidence=())

    def test_rule_evidence_must_be_evidence_refs(self) -> None:
        with self.assertRaises(EligibilityError):
            make_rule(evidence=("not-an-evidence-ref",))  # type: ignore[arg-type]

    def test_blank_summary_or_excerpt_is_rejected(self) -> None:
        for field in ("summary", "verbatim_excerpt"):
            with self.subTest(field=field):
                with self.assertRaises(EligibilityError):
                    make_rule(**{field: "  "})

    def test_rule_exposes_expected_value_from_kind(self) -> None:
        self.assertIs(make_rule(kind=RuleKind.MUST_BE_TRUE).expected, True)
        self.assertIs(make_rule(kind=RuleKind.MUST_BE_FALSE).expected, False)

    def test_invalid_kind_is_rejected(self) -> None:
        with self.assertRaises(EligibilityError):
            make_rule(kind="must_be_true")  # type: ignore[arg-type]

    def test_rule_is_frozen(self) -> None:
        with self.assertRaises(AttributeError):
            make_rule().rule_id = "other"  # type: ignore[misc]


class RuleSetTests(unittest.TestCase):
    def test_rule_set_requires_at_least_one_rule(self) -> None:
        with self.assertRaises(EligibilityError):
            make_rule_set(())

    def test_rule_set_requires_unique_rule_ids(self) -> None:
        duplicate = make_rule(rule_id="duplicate")

        with self.assertRaises(EligibilityError):
            make_rule_set((make_rule(rule_id="duplicate"), duplicate))

    def test_blank_version_strings_are_rejected(self) -> None:
        base = {
            "scheme": "PM-KISAN",
            "rule_set_version": "v1",
            "corpus_version": "processed-v1:test",
        }

        for field in base:
            for blank in ("", "   "):
                with self.subTest(field=field, blank=blank):
                    values = {**base, field: blank}
                    with self.assertRaises(EligibilityError):
                        RuleSet(rules=(make_rule(),), **values)  # type: ignore[arg-type]

    def test_non_rule_member_is_rejected(self) -> None:
        with self.assertRaises(EligibilityError):
            make_rule_set(("not-a-rule",))  # type: ignore[arg-type]

    def test_rule_set_is_frozen(self) -> None:
        with self.assertRaises(AttributeError):
            make_rule_set().scheme = "other"  # type: ignore[misc]


class EligibilityDecisionTests(unittest.TestCase):
    def test_insufficient_information_must_name_missing_facts(self) -> None:
        with self.assertRaises(EligibilityError):
            EligibilityDecision(
                scheme="PM-KISAN",
                status=EligibilityStatus.INSUFFICIENT_INFORMATION,
                summary="cannot decide",
            )

    def test_eligible_cannot_have_missing_facts(self) -> None:
        with self.assertRaises(EligibilityError):
            EligibilityDecision(
                scheme="PM-KISAN",
                status=EligibilityStatus.ELIGIBLE,
                summary="ok",
                missing_facts=("land_is_cultivable",),
            )

    def test_unsupported_scheme_must_not_name_a_rule(self) -> None:
        with self.assertRaises(EligibilityError):
            EligibilityDecision(
                scheme="OTHER",
                status=EligibilityStatus.UNSUPPORTED_SCHEME,
                rule_id="pm-kisan-land-in-own-name",
                summary="unsupported",
            )

    def test_blank_summary_is_rejected(self) -> None:
        with self.assertRaises(EligibilityError):
            EligibilityDecision(scheme="PM-KISAN", status=EligibilityStatus.ELIGIBLE, summary=" ")

    def test_list_valued_collections_are_rejected(self) -> None:
        with self.assertRaises(EligibilityError):
            EligibilityDecision(
                scheme="PM-KISAN",
                status=EligibilityStatus.ELIGIBLE,
                summary="ok",
                evaluated_rules=["a-rule"],  # type: ignore[arg-type]
            )

    def test_status_values_are_stable_strings(self) -> None:
        self.assertEqual(EligibilityStatus.ELIGIBLE, "eligible")
        self.assertEqual(EligibilityStatus.INELIGIBLE, "ineligible")
        self.assertEqual(EligibilityStatus.INSUFFICIENT_INFORMATION, "insufficient_information")
        self.assertEqual(EligibilityStatus.UNSUPPORTED_SCHEME, "unsupported_scheme")


if __name__ == "__main__":
    unittest.main()
