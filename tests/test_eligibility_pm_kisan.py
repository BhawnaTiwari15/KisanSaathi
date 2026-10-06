"""PM-KISAN rule and evaluator tests.

Fixtures are fixed literals. One test class verifies rule excerpts against the real
ingested corpus and is skipped when that local, gitignored artifact is absent.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from kisansathi.eligibility.evaluator import evaluate
from kisansathi.eligibility.models import (
    FACT_NAMES,
    EligibilityRequest,
    EligibilityRule,
    EligibilityStatus,
    EvidenceRef,
    RuleKind,
    RuleSet,
)
from kisansathi.eligibility.pm_kisan_rules import (
    CORPUS_SHA256,
    ELIGIBILITY_BASIS,
    PM_KISAN,
    PM_KISAN_RULES,
    PM_KISAN_SCHEME,
    RULE_SET_VERSION,
    SOURCE_ID,
    UNSUPPORTED_CONDITIONS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "processed" / "v1" / SOURCE_ID

ELIGIBLE_KWARGS: dict[str, bool] = {
    "landholding_in_own_name": True,
    "land_is_cultivable": True,
    "land_used_for_non_agricultural_purpose": False,
    "family_member_paid_income_tax_last_assessment_year": False,
}


def load_corpus_chunks() -> dict[str, dict] | None:
    """Return the ingested PM-KISAN chunks keyed by chunk id, or None if unavailable."""
    if not CORPUS_DIR.is_dir():
        return None
    files = sorted(CORPUS_DIR.glob("*.jsonl"))
    if not files:
        return None
    chunks: dict[str, dict] = {}
    for line in files[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_type") == "chunk":
            chunks[record["chunk_id"]] = record
    return chunks or None


class PmKisanRuleSetDefinitionTests(unittest.TestCase):
    def test_rule_set_is_pinned_to_a_versioned_corpus(self) -> None:
        self.assertEqual(PM_KISAN.scheme, PM_KISAN_SCHEME)
        self.assertEqual(PM_KISAN.rule_set_version, RULE_SET_VERSION)
        self.assertIn(CORPUS_SHA256, PM_KISAN.corpus_version)

    def test_only_corpus_supported_conditions_are_implemented(self) -> None:
        implemented = {rule.fact for rule in PM_KISAN_RULES}

        self.assertEqual(implemented, set(FACT_NAMES))

    def test_every_implemented_fact_is_covered_exactly_once(self) -> None:
        facts = [rule.fact for rule in PM_KISAN_RULES]

        self.assertEqual(len(facts), len(set(facts)))

    def test_every_rule_cites_the_pinned_source(self) -> None:
        for rule in PM_KISAN_RULES:
            with self.subTest(rule=rule.rule_id):
                self.assertTrue(rule.evidence)
                for reference in rule.evidence:
                    self.assertEqual(reference.source_id, SOURCE_ID)
                    self.assertEqual(reference.sha256, CORPUS_SHA256)

    def test_eligibility_basis_is_not_an_evaluated_rule(self) -> None:
        self.assertNotIn(ELIGIBILITY_BASIS, PM_KISAN_RULES)

    def test_unsupported_conditions_record_a_reason_and_locator(self) -> None:
        for condition in UNSUPPORTED_CONDITIONS:
            with self.subTest(condition=condition.condition_id):
                self.assertTrue(condition.condition_id.strip())
                self.assertTrue(condition.reason.strip())
                self.assertTrue(condition.locator.strip())

    def test_land_size_limit_is_documented_as_unsupported(self) -> None:
        recorded = {c.condition_id for c in UNSUPPORTED_CONDITIONS}

        self.assertIn("land-size-limit", recorded)

    def test_no_rule_depends_on_land_size(self) -> None:
        size_related = ("size", "hectare", "area", "land_area", "scale")

        for rule in (*PM_KISAN_RULES, ELIGIBILITY_BASIS):
            with self.subTest(rule=rule.rule_id):
                self.assertFalse(
                    [term for term in size_related if term in rule.fact.lower()],
                    f"{rule.rule_id} must not depend on land size",
                )

    def test_unsupported_conditions_are_unique(self) -> None:
        recorded = [c.condition_id for c in UNSUPPORTED_CONDITIONS]

        self.assertEqual(len(recorded), len(set(recorded)))


class EligibleDecisionTests(unittest.TestCase):
    def test_all_supported_facts_holding_is_eligible(self) -> None:
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME, **ELIGIBLE_KWARGS))

        self.assertEqual(decision.status, EligibilityStatus.ELIGIBLE)
        self.assertEqual(decision.missing_facts, ())
        self.assertIsNone(decision.rule_id)

    def test_eligible_decision_preserves_provenance_for_every_rule(self) -> None:
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME, **ELIGIBLE_KWARGS))

        self.assertEqual(set(decision.evaluated_rules), {r.rule_id for r in PM_KISAN_RULES})
        self.assertEqual(len(decision.evidence), len(PM_KISAN_RULES))

    def test_eligible_decision_cites_the_pinned_corpus(self) -> None:
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME, **ELIGIBLE_KWARGS))

        for reference in decision.evidence:
            self.assertIsInstance(reference, EvidenceRef)
            self.assertEqual(reference.source_id, SOURCE_ID)
            self.assertEqual(reference.sha256, CORPUS_SHA256)
            self.assertTrue(reference.chunk_id.startswith(f"{SOURCE_ID}:"))
            self.assertGreaterEqual(reference.page_start, 1)
            self.assertGreaterEqual(reference.page_end, reference.page_start)


class IneligibleDecisionTests(unittest.TestCase):
    def assert_ineligible(self, violated_fact: str, expected_rule_id: str) -> None:
        """Assert the single exclusion is detected and fully attributed."""
        facts = dict(ELIGIBLE_KWARGS)
        facts[violated_fact] = not facts[violated_fact]
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME, **facts))

        self.assertEqual(decision.status, EligibilityStatus.INELIGIBLE)
        self.assertEqual(decision.rule_id, expected_rule_id)
        summaries = {rule.rule_id: rule.summary for rule in PM_KISAN_RULES}
        self.assertEqual(decision.summary, summaries[expected_rule_id])
        self.assertEqual(decision.missing_facts, ())
        violated = next(rule for rule in PM_KISAN_RULES if rule.rule_id == expected_rule_id)
        self.assertEqual(
            [reference.chunk_id for reference in decision.evidence],
            [reference.chunk_id for reference in violated.evidence],
        )

    def test_land_not_in_own_name_is_ineligible(self) -> None:
        self.assert_ineligible("landholding_in_own_name", "pm-kisan-land-in-own-name")

    def test_non_cultivable_land_is_ineligible(self) -> None:
        self.assert_ineligible("land_is_cultivable", "pm-kisan-land-must-be-cultivable")

    def test_non_agricultural_land_use_is_ineligible(self) -> None:
        self.assert_ineligible(
            "land_used_for_non_agricultural_purpose",
            "pm-kisan-land-must-be-agricultural",
        )

    def test_family_income_tax_payee_is_ineligible(self) -> None:
        self.assert_ineligible(
            "family_member_paid_income_tax_last_assessment_year",
            "pm-kisan-no-family-income-tax",
        )

    def test_each_implemented_rule_has_a_reachable_ineligible_path(self) -> None:
        for rule in PM_KISAN_RULES:
            with self.subTest(rule=rule.rule_id):
                facts = dict(ELIGIBLE_KWARGS)
                facts[rule.fact] = not facts[rule.fact]
                decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME, **facts))
                self.assertEqual(decision.rule_id, rule.rule_id)

    def test_exclusion_outranks_unknown_facts(self) -> None:
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME, landholding_in_own_name=False))

        self.assertEqual(decision.status, EligibilityStatus.INELIGIBLE)
        self.assertEqual(decision.rule_id, "pm-kisan-land-in-own-name")
        self.assertNotEqual(decision.missing_facts, ())

    def test_first_violated_rule_in_declaration_order_is_reported(self) -> None:
        decision = evaluate(
            EligibilityRequest(
                PM_KISAN_SCHEME,
                landholding_in_own_name=False,
                land_is_cultivable=False,
            )
        )

        self.assertEqual(decision.rule_id, "pm-kisan-land-in-own-name")

    def test_ineligible_reason_is_reproducible(self) -> None:
        request = EligibilityRequest(PM_KISAN_SCHEME, land_is_cultivable=False)

        self.assertEqual(
            evaluate(request).rule_id,
            evaluate(request).rule_id,
        )


class InsufficientInformationTests(unittest.TestCase):
    def test_no_facts_yields_every_missing_fact(self) -> None:
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME))

        self.assertEqual(decision.status, EligibilityStatus.INSUFFICIENT_INFORMATION)
        self.assertEqual(decision.missing_facts, FACT_NAMES)

    def test_partially_known_request_names_only_the_unknown_facts(self) -> None:
        decision = evaluate(
            EligibilityRequest(
                PM_KISAN_SCHEME,
                landholding_in_own_name=True,
                land_is_cultivable=True,
            )
        )

        self.assertEqual(decision.status, EligibilityStatus.INSUFFICIENT_INFORMATION)
        self.assertEqual(
            decision.missing_facts,
            (
                "land_used_for_non_agricultural_purpose",
                "family_member_paid_income_tax_last_assessment_year",
            ),
        )

    def test_insufficient_decision_cites_no_rule(self) -> None:
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME))

        self.assertEqual(decision.evidence, ())
        self.assertIsNone(decision.rule_id)

    def test_insufficient_decision_does_not_claim_eligibility(self) -> None:
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME, landholding_in_own_name=True))

        self.assertNotEqual(decision.status, EligibilityStatus.ELIGIBLE)


class UnsupportedSchemeTests(unittest.TestCase):
    def test_unknown_scheme_is_unsupported(self) -> None:
        decision = evaluate(EligibilityRequest("PM-RKVY-PDMC"))

        self.assertEqual(decision.status, EligibilityStatus.UNSUPPORTED_SCHEME)

    def test_unsupported_scheme_evaluates_no_rules(self) -> None:
        decision = evaluate(EligibilityRequest("PM-RKVY-PDMC", **ELIGIBLE_KWARGS))

        self.assertEqual(decision.evaluated_rules, ())
        self.assertEqual(decision.evidence, ())
        self.assertIsNone(decision.rule_id)

    def test_scheme_comparison_ignores_case_and_padding(self) -> None:
        for scheme in ("pm-kisan", "  PM-KISAN  ", "Pm-Kisan"):
            with self.subTest(scheme=scheme):
                decision = evaluate(EligibilityRequest(scheme, **ELIGIBLE_KWARGS))
                self.assertEqual(decision.status, EligibilityStatus.ELIGIBLE)

    def test_unsupported_scheme_keeps_the_requested_scheme_name(self) -> None:
        self.assertEqual(evaluate(EligibilityRequest("pm-rkvy-pdMC")).scheme, "pm-rkvy-pdMC")


class DeterminismTests(unittest.TestCase):
    def test_repeated_evaluation_is_identical(self) -> None:
        for request in (
            EligibilityRequest(PM_KISAN_SCHEME, **ELIGIBLE_KWARGS),
            EligibilityRequest(PM_KISAN_SCHEME),
            EligibilityRequest(PM_KISAN_SCHEME, landholding_in_own_name=False),
            EligibilityRequest("OTHER"),
        ):
            with self.subTest(request=request):
                decisions = {evaluate(request) for _ in range(100)}

                self.assertEqual(len(decisions), 1)

    def test_decision_is_frozen(self) -> None:
        decision = evaluate(EligibilityRequest(PM_KISAN_SCHEME, **ELIGIBLE_KWARGS))

        with self.assertRaises(AttributeError):
            decision.status = EligibilityStatus.INELIGIBLE  # type: ignore[misc]

    def test_custom_rule_set_is_evaluated_without_mutation(self) -> None:
        strict = RuleSet(
            scheme="TEST-SCHEME",
            rule_set_version="test-v1",
            corpus_version="processed-v1:test",
            rules=(
                EligibilityRule(
                    rule_id="test-only",
                    kind=RuleKind.MUST_BE_TRUE,
                    fact="landholding_in_own_name",
                    summary="summary",
                    verbatim_excerpt="excerpt",
                    evidence=(
                        EvidenceRef(
                            source_id=SOURCE_ID,
                            sha256=CORPUS_SHA256,
                            chunk_id=f"{SOURCE_ID}:0123456789abcdef01234567",
                            page_start=1,
                            page_end=1,
                            locator="test",
                        ),
                    ),
                ),
            ),
        )

        decision = evaluate(
            EligibilityRequest("TEST-SCHEME", landholding_in_own_name=True),
            rules=strict,
        )

        self.assertEqual(decision.status, EligibilityStatus.ELIGIBLE)
        self.assertEqual(PM_KISAN_RULES, PM_KISAN.rules)

    def test_default_rule_set_is_the_pm_kisan_rule_set(self) -> None:
        request = EligibilityRequest(PM_KISAN_SCHEME, **ELIGIBLE_KWARGS)

        self.assertEqual(evaluate(request), evaluate(request, rules=PM_KISAN))


class ImportIsolationTests(unittest.TestCase):
    def test_importing_eligibility_does_not_import_retrieval_or_langgraph(self) -> None:
        source_root = REPO_ROOT / "src"
        code = """
import builtins, sys
original_import = builtins.__import__
FORBIDDEN = ('kisansathi.retrieval', 'kisansathi.orchestration', 'langgraph', 'langchain_core')

def guarded_import(name, *args, **kwargs):
    if any(name == f or name.startswith(f + '.') for f in FORBIDDEN):
        raise AssertionError(f'forbidden import during module import: {name}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import kisansathi.eligibility
from kisansathi.eligibility.evaluator import evaluate
for name in FORBIDDEN:
    assert name not in sys.modules, f'{name} was imported'
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source_root)

        subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )


class CorpusProvenanceTests(unittest.TestCase):
    """Verifies rule excerpts against the real ingested corpus.

    Skipped when data/processed is absent: it is gitignored local build output, so it
    cannot be relied on in CI. Structural provenance is asserted unconditionally by
    PmKisanRuleSetDefinitionTests instead.
    """

    chunks = load_corpus_chunks()

    @unittest.skipIf(chunks is None, "processed corpus not available locally")
    def test_every_rule_excerpt_is_verbatim_chunk_text(self) -> None:
        assert self.chunks is not None

        for rule in (*PM_KISAN_RULES, ELIGIBILITY_BASIS):
            with self.subTest(rule=rule.rule_id):
                chunk = self.chunks[rule.evidence[0].chunk_id]
                self.assertIn(rule.verbatim_excerpt, chunk["text"])

    @unittest.skipIf(chunks is None, "processed corpus not available locally")
    def test_every_evidence_page_range_matches_the_chunk(self) -> None:
        assert self.chunks is not None

        for rule in (*PM_KISAN_RULES, ELIGIBILITY_BASIS):
            for reference in rule.evidence:
                with self.subTest(rule=rule.rule_id, chunk=reference.chunk_id):
                    chunk = self.chunks[reference.chunk_id]
                    self.assertEqual(reference.page_start, chunk["page_start"])
                    self.assertEqual(reference.page_end, chunk["page_end"])
                    self.assertEqual(reference.sha256, chunk["sha256"])

    @unittest.skipIf(chunks is None, "processed corpus not available locally")
    def test_corpus_file_name_matches_the_pinned_checksum(self) -> None:
        assert self.chunks is not None

        self.assertIn(CORPUS_SHA256, {path.stem for path in CORPUS_DIR.glob("*.jsonl")})

    @unittest.skipIf(chunks is None, "processed corpus not available locally")
    def test_excerpts_are_not_only_present_as_leaked_headings(self) -> None:
        """Every implemented excerpt must live in text, not just in a heading field."""
        assert self.chunks is not None

        for rule in PM_KISAN_RULES:
            with self.subTest(rule=rule.rule_id):
                chunk = self.chunks[rule.evidence[0].chunk_id]
                self.assertNotEqual(chunk.get("heading", "").strip(), rule.verbatim_excerpt)


if __name__ == "__main__":
    unittest.main()
