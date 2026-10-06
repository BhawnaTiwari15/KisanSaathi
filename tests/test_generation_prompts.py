"""Unit tests for prompt building functions."""

import unittest
from kisansathi.citations.models import CitationBatch, RejectedCitation
from kisansathi.domain.schemas import Citation, Language, ResponseStatus, UserMessage
from kisansathi.eligibility.models import EligibilityDecision, EligibilityStatus
from kisansathi.generation.prompts import (
    build_system_prompt,
    build_user_prompt,
    parse_generated_answer,
)
from kisansathi.weather.models import WeatherCurrent, WeatherResponse


class TestBuildSystemPrompt(unittest.TestCase):
    def test_english(self) -> None:
        prompt = build_system_prompt(Language.ENGLISH)
        self.assertIn("English", prompt)
        self.assertIn("KisanSaathi", prompt)
        self.assertIn("Do not invent", prompt)
        self.assertIn("citation ID", prompt)

    def test_hindi(self) -> None:
        prompt = build_system_prompt(Language.HINDI)
        self.assertIn("Hindi", prompt)

    def test_marathi(self) -> None:
        prompt = build_system_prompt(Language.HINDI)
        self.assertIn("Hindi", prompt)


class TestBuildUserPrompt(unittest.TestCase):
    def setUp(self) -> None:
        self.message = UserMessage(
            text="What are the PM-KISAN eligibility rules?",
            language=Language.ENGLISH,
        )
        self.citation = Citation(
            source_id="pm-kisan-revised-faq",
            title="PM-KISAN FAQ",
            url="https://pmkisan.gov.in/Documents/RevisedFAQ.pdf",
            page_number=4,
            page_end=None,
            issuing_authority="Department of Agriculture",
            chunk_id="pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76",
        )
        self.batch = CitationBatch(citations=(self.citation,), rejected=())

    def test_includes_question(self) -> None:
        prompt = build_user_prompt(self.message, self.batch, None, None)
        self.assertIn("What are the PM-KISAN eligibility rules?", prompt)

    def test_includes_citations(self) -> None:
        prompt = build_user_prompt(self.message, self.batch, None, None)
        self.assertIn("pm-kisan-revised-faq:33ff2db494c59ab1ca6c4f76", prompt)
        self.assertIn("PM-KISAN FAQ", prompt)
        self.assertIn("Department of Agriculture", prompt)

    def test_includes_eligibility(self) -> None:
        decision = EligibilityDecision(
            scheme="PM-KISAN",
            status=EligibilityStatus.ELIGIBLE,
            summary="All conditions satisfied.",
            evidence=(),
        )
        prompt = build_user_prompt(self.message, self.batch, decision, None)
        self.assertIn("ELIGIBILITY", prompt)
        self.assertIn("All conditions satisfied", prompt)

    def test_includes_weather(self) -> None:
        weather = WeatherResponse(
            latitude=28.6139,
            longitude=77.2090,
            timezone="Asia/Kolkata",
            current=WeatherCurrent(
                temperature_c=31.2,
                precipitation_mm=0.0,
                wind_speed_mps=8.4,
                weather_code=2,
                time_iso="2026-10-06T09:00",
            ),
        )
        prompt = build_user_prompt(self.message, self.batch, None, weather)
        self.assertIn("WEATHER", prompt)
        self.assertIn("31.2", prompt)

    def test_empty_citations(self) -> None:
        empty_batch = CitationBatch()
        prompt = build_user_prompt(self.message, empty_batch, None, None)
        self.assertIn("No evidence available", prompt)


class TestParseGeneratedAnswer(unittest.TestCase):
    def setUp(self) -> None:
        self.citation_ids = frozenset(["chunk-1", "chunk-2"])

    def test_valid_output(self) -> None:
        raw = (
            "ANSWER:\nThis is the answer [chunk-1].\n\n"
            "CITATIONS:\n[chunk-1]\n[chunk-2]"
        )
        answer = parse_generated_answer(raw, self.citation_ids)
        self.assertEqual(answer.text, "This is the answer [chunk-1].")
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))
        self.assertEqual(answer.status, ResponseStatus.ANSWERED)

    def test_valid_output_without_brackets(self) -> None:
        raw = (
            "ANSWER:\nThis is the answer.\n\n"
            "CITATIONS:\nchunk-1\nchunk-2"
        )
        answer = parse_generated_answer(raw, self.citation_ids)
        self.assertEqual(answer.citation_ids, ("chunk-1", "chunk-2"))

    def test_missing_answer_section(self) -> None:
        raw = "CITATIONS:\n[chunk-1]"
        with self.assertRaises(Exception) as cm:
            parse_generated_answer(raw, self.citation_ids)
        self.assertIn("ANSWER section", str(cm.exception))

    def test_empty_answer(self) -> None:
        raw = "ANSWER:\n\nCITATIONS:\n[chunk-1]"
        with self.assertRaises(Exception) as cm:
            parse_generated_answer(raw, self.citation_ids)
        self.assertIn("empty", str(cm.exception))

    def test_unknown_citation_id(self) -> None:
        raw = (
            "ANSWER:\nThis is the answer.\n\n"
            "CITATIONS:\n[chunk-999]"
        )
        with self.assertRaises(Exception) as cm:
            parse_generated_answer(raw, self.citation_ids)
        self.assertIn("unknown citation ID", str(cm.exception))

    def test_insufficient_evidence_detected(self) -> None:
        raw = (
            "ANSWER:\nThere is insufficient evidence to answer.\n\n"
            "CITATIONS:\n[chunk-1]"
        )
        answer = parse_generated_answer(raw, self.citation_ids)
        self.assertEqual(answer.status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_hindi_insufficient_detected(self) -> None:
        raw = (
            "ANSWER:\nपर्याप्त सबूत नहीं हैं।\n\n"
            "CITATIONS:\n[chunk-1]"
        )
        answer = parse_generated_answer(raw, self.citation_ids)
        self.assertEqual(answer.status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_malformed_output_handled(self) -> None:
        raw = "This is not a valid format at all"
        with self.assertRaises(Exception):
            parse_generated_answer(raw, self.citation_ids)


if __name__ == "__main__":
    unittest.main()