"""Unit tests for fake generation utilities."""

import unittest
from kisansathi.citations.models import CitationBatch
from kisansathi.domain.schemas import Language, ResponseStatus, UserMessage
from kisansathi.generation.fake import (
    FakeLLMClient,
    FakeAnswerGenerator,
    make_fake_generated_answer,
    make_fake_generation_context,
)
from kisansathi.generation.models import GeneratedAnswer, LLMError


class TestFakeLLMClient(unittest.TestCase):
    def test_returns_canned_response(self) -> None:
        client = FakeLLMClient(response="Canned response")
        result = client.generate(system_prompt="sys", user_prompt="user")
        self.assertEqual(result, "Canned response")

    def test_raises_configured_exception(self) -> None:
        client = FakeLLMClient(response=Exception("Network error"))
        with self.assertRaises(Exception) as cm:
            client.generate(system_prompt="sys", user_prompt="user")
        self.assertIn("Network error", str(cm.exception))

    def test_records_calls(self) -> None:
        client = FakeLLMClient(response="Response 1")
        client.generate(system_prompt="sys1", user_prompt="user1", temperature=0.1)
        client.generate(system_prompt="sys2", user_prompt="user2", max_tokens=100)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["system_prompt"], "sys1")
        self.assertEqual(client.calls[0]["temperature"], 0.1)
        self.assertEqual(client.calls[1]["max_tokens"], 100)


class TestFakeAnswerGenerator(unittest.TestCase):
    def test_returns_canned_generated_answer(self) -> None:
        expected = GeneratedAnswer(
            text="Fake answer",
            citation_ids=("chunk-1",),
            status=ResponseStatus.ANSWERED,
            language=Language.ENGLISH,
        )
        fake = FakeAnswerGenerator(answer=expected)

        from kisansathi.generation.fake import make_fake_generation_context
        context = make_fake_generation_context()
        result = fake.generate(context)

        self.assertEqual(result.text, "Fake answer")
        self.assertEqual(result.citation_ids, ("chunk-1",))

    def test_default_answer_when_none_provided(self) -> None:
        fake = FakeAnswerGenerator(answer=None)

        from kisansathi.generation.fake import make_fake_generation_context
        context = make_fake_generation_context()
        result = fake.generate(context)

        self.assertIsInstance(result, GeneratedAnswer)
        self.assertEqual(result.status, ResponseStatus.ANSWERED)

    def test_records_calls(self) -> None:
        fake = FakeAnswerGenerator()
        from kisansathi.generation.fake import make_fake_generation_context

        context1 = make_fake_generation_context(message_text="Q1")
        context2 = make_fake_generation_context(message_text="Q2")

        fake.generate(context1)
        fake.generate(context2)

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0].message.text, "Q1")
        self.assertEqual(fake.calls[1].message.text, "Q2")

    def test_raises_configured_exception(self) -> None:
        fake = FakeAnswerGenerator(answer=LLMError("Simulated LLM failure"))
        from kisansathi.generation.fake import make_fake_generation_context
        context = make_fake_generation_context()

        with self.assertRaises(LLMError):
            fake.generate(context)


class TestMakeFakeGeneratedAnswer(unittest.TestCase):
    def test_creates_answer_with_defaults(self) -> None:
        answer = make_fake_generated_answer()
        self.assertIsInstance(answer, GeneratedAnswer)
        self.assertEqual(answer.status, ResponseStatus.ANSWERED)
        self.assertEqual(answer.language, Language.ENGLISH)

    def test_accepts_overrides(self) -> None:
        answer = make_fake_generated_answer(
            text="Custom text",
            citation_ids=("a", "b"),
            status=ResponseStatus.NEEDS_CLARIFICATION,
            language=Language.HINDI,
        )
        self.assertEqual(answer.text, "Custom text")
        self.assertEqual(answer.citation_ids, ("a", "b"))
        self.assertEqual(answer.status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(answer.language, Language.HINDI)


class TestMakeFakeGenerationContext(unittest.TestCase):
    def test_creates_context_with_defaults(self) -> None:
        context = make_fake_generation_context()
        self.assertIsInstance(context.message, UserMessage)
        self.assertEqual(context.message.text, "Test question")
        self.assertEqual(context.message.language, Language.ENGLISH)
        self.assertIsInstance(context.citations, CitationBatch)
        self.assertIsNone(context.eligibility_decision)
        self.assertIsNone(context.weather)

    def test_accepts_overrides(self) -> None:
        from kisansathi.citations.models import CitationBatch
        custom_batch = CitationBatch()
        context = make_fake_generation_context(
            message_text="Custom question",
            citations=custom_batch,
            language=Language.HINDI,
        )
        self.assertEqual(context.message.text, "Custom question")
        self.assertEqual(context.message.language, Language.HINDI)
        self.assertIs(context.citations, custom_batch)


if __name__ == "__main__":
    unittest.main()