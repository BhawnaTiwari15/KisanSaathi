"""Unit tests for the DefaultAnswerGenerator and parsing logic."""

import unittest
from kisansathi.citations.models import CitationBatch, Citation
from kisansathi.domain.schemas import Language, ResponseStatus, UserMessage
from kisansathi.eligibility.models import EligibilityDecision, EligibilityStatus
from kisansathi.generation.fake import FakeLLMClient, FakeAnswerGenerator
from kisansathi.generation.generator import DefaultAnswerGenerator
from kisansathi.generation.models import GenerationContext, LLMError, MalformedOutputError, GroundingError
from kisansathi.generation.prompts import parse_generated_answer
from kisansathi.weather.models import WeatherCurrent, WeatherResponse


class TestDefaultAnswerGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.citation = Citation(
            source_id="test-source",
            title="Test Title",
            url="https://example.com",
            page_number=1,
            page_end=None,
            issuing_authority="Test Authority",
            chunk_id="test-source:abcdef123456789012345678",
        )
        self.batch = CitationBatch(citations=(self.citation,), rejected=())
        self.context = GenerationContext(
            message=UserMessage(text="Test question", language=Language.ENGLISH),
            citations=self.batch,
            eligibility_decision=None,
            weather=None,
        )

    def test_success_with_valid_llm_response(self) -> None:
        fake_llm = FakeLLMClient(
            response=(
                "ANSWER:\nThis is the answer [test-source:abcdef123456789012345678].\n\n"
                "CITATIONS:\n[test-source:abcdef123456789012345678]"
            )
        )
        generator = DefaultAnswerGenerator(fake_llm)
        answer = generator.generate(self.context)

        self.assertEqual(answer.text, "This is the answer [test-source:abcdef123456789012345678].")
        self.assertEqual(answer.citation_ids, ("test-source:abcdef123456789012345678",))
        self.assertEqual(answer.status, ResponseStatus.ANSWERED)
        self.assertEqual(len(fake_llm.calls), 1)

    def test_llm_error_propagates(self) -> None:
        fake_llm = FakeLLMClient(response=Exception("Network timeout"))
        generator = DefaultAnswerGenerator(fake_llm)

        with self.assertRaises(LLMError):
            generator.generate(self.context)

    def test_malformed_output_raises(self) -> None:
        fake_llm = FakeLLMClient(response="Not a valid format")
        generator = DefaultAnswerGenerator(fake_llm)

        with self.assertRaises(MalformedOutputError):
            generator.generate(self.context)

    def test_unknown_citation_id_raises_grounding_error(self) -> None:
        fake_llm = FakeLLMClient(
            response=(
                "ANSWER:\nThis is the answer.\n\n"
                "CITATIONS:\n[unknown-chunk]"
            )
        )
        generator = DefaultAnswerGenerator(fake_llm)

        with self.assertRaises(GroundingError):
            generator.generate(self.context)

    def test_temperature_and_max_tokens_passed(self) -> None:
        fake_llm = FakeLLMClient(
            response=(
                "ANSWER:\nAnswer.\n\n"
                "CITATIONS:\n[test-source:abcdef123456789012345678]"
            )
        )
        generator = DefaultAnswerGenerator(fake_llm, temperature=0.5, max_tokens=256)
        generator.generate(self.context)

        call = fake_llm.calls[0]
        self.assertEqual(call["temperature"], 0.5)
        self.assertEqual(call["max_tokens"], 256)

    def test_system_prompt_contains_language(self) -> None:
        fake_llm = FakeLLMClient(
            response=(
                "ANSWER:\nAnswer.\n\n"
                "CITATIONS:\n[test-source:abcdef123456789012345678]"
            )
        )
        generator = DefaultAnswerGenerator(fake_llm)
        context = GenerationContext(
            message=UserMessage(text="Test", language=Language.HINDI),
            citations=self.batch,
            eligibility_decision=None,
            weather=None,
        )
        generator.generate(context)

        system_prompt = fake_llm.calls[0]["system_prompt"]
        self.assertIn("Hindi", system_prompt)

    def test_user_prompt_contains_question(self) -> None:
        fake_llm = FakeLLMClient(
            response=(
                "ANSWER:\nAnswer.\n\n"
                "CITATIONS:\n[test-source:abcdef123456789012345678]"
            )
        )
        generator = DefaultAnswerGenerator(fake_llm)
        generator.generate(self.context)

        user_prompt = fake_llm.calls[0]["user_prompt"]
        self.assertIn("Test question", user_prompt)

    def test_user_prompt_contains_citations(self) -> None:
        fake_llm = FakeLLMClient(
            response=(
                "ANSWER:\nAnswer.\n\n"
                "CITATIONS:\n[test-source:abcdef123456789012345678]"
            )
        )
        generator = DefaultAnswerGenerator(fake_llm)
        generator.generate(self.context)

        user_prompt = fake_llm.calls[0]["user_prompt"]
        self.assertIn("test-source:abcdef123456789012345678", user_prompt)

    def test_user_prompt_contains_eligibility(self) -> None:
        decision = EligibilityDecision(
            scheme="PM-KISAN",
            status=EligibilityStatus.ELIGIBLE,
            summary="All conditions satisfied.",
            evidence=(),
        )
        fake_llm = FakeLLMClient(
            response=(
                "ANSWER:\nAnswer.\n\n"
                "CITATIONS:\n[test-source:abcdef123456789012345678]"
            )
        )
        generator = DefaultAnswerGenerator(fake_llm)
        context = GenerationContext(
            message=UserMessage(text="Test", language=Language.ENGLISH),
            citations=self.batch,
            eligibility_decision=decision,
            weather=None,
        )
        generator.generate(context)

        user_prompt = fake_llm.calls[0]["user_prompt"]
        self.assertIn("ELIGIBILITY", user_prompt)
        self.assertIn("All conditions satisfied", user_prompt)

    def test_user_prompt_contains_weather(self) -> None:
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
        fake_llm = FakeLLMClient(
            response=(
                "ANSWER:\nAnswer.\n\n"
                "CITATIONS:\n[test-source:abcdef123456789012345678]"
            )
        )
        generator = DefaultAnswerGenerator(fake_llm)
        context = GenerationContext(
            message=UserMessage(text="Test", language=Language.ENGLISH),
            citations=self.batch,
            eligibility_decision=None,
            weather=weather,
        )
        generator.generate(context)

        user_prompt = fake_llm.calls[0]["user_prompt"]
        self.assertIn("WEATHER", user_prompt)
        self.assertIn("31.2", user_prompt)


class TestFakeAnswerGenerator(unittest.TestCase):
    def test_returns_canned_answer(self) -> None:
        from kisansathi.generation.fake import make_fake_generated_answer, make_fake_generation_context
        from kisansathi.domain.schemas import ResponseStatus

        expected = make_fake_generated_answer(
            text="Canned answer",
            citation_ids=("chunk-1",),
            status=ResponseStatus.ANSWERED,
        )
        fake_gen = FakeAnswerGenerator(answer=expected)
        context = make_fake_generation_context()

        result = fake_gen.generate(context)

        self.assertEqual(result.text, "Canned answer")
        self.assertEqual(result.citation_ids, ("chunk-1",))

    def test_records_calls(self) -> None:
        from kisansathi.generation.fake import make_fake_generation_context

        fake_gen = FakeAnswerGenerator()
        context1 = make_fake_generation_context(message_text="Question 1")
        context2 = make_fake_generation_context(message_text="Question 2")

        fake_gen.generate(context1)
        fake_gen.generate(context2)

        self.assertEqual(len(fake_gen.calls), 2)
        self.assertEqual(fake_gen.calls[0].message.text, "Question 1")
        self.assertEqual(fake_gen.calls[1].message.text, "Question 2")

    def test_raises_configured_exception(self) -> None:
        from kisansathi.generation.fake import make_fake_generation_context
        from kisansathi.generation.models import LLMError

        fake_gen = FakeAnswerGenerator(answer=LLMError("Simulated failure"))
        context = make_fake_generation_context()

        with self.assertRaises(LLMError):
            fake_gen.generate(context)


if __name__ == "__main__":
    unittest.main()