import unittest
from typing import Any
from unittest.mock import patch

from kisansathi.domain.schemas import (
    AssistantResponse,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.orchestration.graph import OrchestrationState, build_graph
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.weather.client import OpenMeteoClient
from kisansathi.weather.exceptions import WeatherRequestError
from kisansathi.weather.models import WeatherCurrent, WeatherRequest, WeatherResponse


class FakeRetriever:
    def __init__(self, results: tuple[SearchResult, ...] | None = None) -> None:
        self.results = results or ()
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        self.calls.append((query, top_k))
        return self.results


class FakeWeatherClient:
    def __init__(
        self,
        response: WeatherResponse | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.response = response
        self.raise_exc = raise_exc
        self.calls: list[tuple[WeatherRequest, bool, int]] = []

    def get_forecast(
        self,
        req: WeatherRequest,
        *,
        include_forecast: bool = False,
        forecast_days: int = 1,
    ) -> WeatherResponse:
        self.calls.append((req, include_forecast, forecast_days))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


OPEN_METEO_PAYLOAD: dict[str, object] = {
    "latitude": 28.6139,
    "longitude": 77.209,
    "timezone": "Asia/Kolkata",
    "current": {
        "temperature_2m": 24.6,
        "precipitation": 0.0,
        "rain": 0.0,
        "wind_speed_10m": 5.1,
        "wind_direction_10m": 210,
        "weather_code": 3,
        "time": "2026-10-06T09:00",
    },
}


class FakeTransport:
    def __init__(self, response: object = None) -> None:
        self.response = response
        self.calls: list[tuple[str, dict, float]] = []

    def get_json(self, url: str, params: dict, timeout: float) -> object:
        self.calls.append((url, dict(params), timeout))
        return self.response


def make_result(chunk_id: str) -> SearchResult:
    payload: dict[str, Any] = {
        "chunk_id": chunk_id,
        "source_id": "test-source",
        "sha256": "0" * 64,
        "scheme": "TEST",
        "jurisdiction": "IN",
        "language": "en",
        "title": "Test Title",
        "page_start": 1,
        "page_end": 1,
        "heading": None,
        "text": "Test text",
    }
    return SearchResult(score=1.0, payload=payload)


def make_weather_response(
    *,
    latitude: float = 28.6139,
    longitude: float = 77.2090,
    temperature_c: float | None = 31.2,
    precipitation_mm: float | None = 0.0,
    wind_speed_mps: float | None = 8.4,
    weather_code: int | None = 2,
) -> WeatherResponse:
    return WeatherResponse(
        latitude=latitude,
        longitude=longitude,
        timezone="Asia/Kolkata",
        current=WeatherCurrent(
            temperature_c=temperature_c,
            precipitation_mm=precipitation_mm,
            wind_speed_mps=wind_speed_mps,
            weather_code=weather_code,
            time_iso="2026-10-06T09:00",
        ),
    )


def make_state(message: UserMessage) -> OrchestrationState:
    return {
        "message": message,
        "route": "",
        "retrieved_chunks": (),
        "response": AssistantResponse(
            text="placeholder",
            language=Language.ENGLISH,
            status=ResponseStatus.ANSWERED,
        ),
    }


class OrchestrationGraphTests(unittest.TestCase):
    def test_normal_message_takes_retrieval_route(self) -> None:
        retriever = FakeRetriever((make_result("c1"),))
        graph = build_graph(retriever).compile()
        message = UserMessage(text="What documents are required for PM-KISAN enrollment?", language=Language.ENGLISH)

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0][0], message.text)
        self.assertEqual(retriever.calls[0][1], 5)
        self.assertEqual(len(result["retrieved_chunks"]), 1)
        self.assertIsInstance(result["response"], AssistantResponse)
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)

    def test_ambiguous_mixed_script_message_falls_back_gracefully(self) -> None:
        retriever = FakeRetriever((make_result("c1"),))
        graph = build_graph(retriever).compile()
        message = UserMessage(text="किसान రైతు kisan", language=None)

        result = graph.invoke(make_state(message))

        self.assertIsNone(result["detected_language"])
        self.assertEqual(retriever.calls, [(message.text, 5)])
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)

    def test_very_short_message_takes_clarification_route(self) -> None:
        retriever = FakeRetriever()
        graph = build_graph(retriever).compile()
        message = UserMessage(text="Help", language=Language.ENGLISH)

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "clarify")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(result["retrieved_chunks"], ())
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_underspecified_message_with_few_words_takes_clarification_route(self) -> None:
        retriever = FakeRetriever()
        graph = build_graph(retriever).compile()
        message = UserMessage(text="PM-KISAN", language=Language.ENGLISH)

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "clarify")
        self.assertEqual(retriever.calls, [])
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_retrieval_preserves_chunks_and_sets_answered_status(self) -> None:
        retriever = FakeRetriever((make_result("c1"), make_result("c2")))
        graph = build_graph(retriever).compile()
        message = UserMessage(text="When was PM-KISAN launched and how often are payments made?", language=Language.ENGLISH)

        result = graph.invoke(make_state(message))

        self.assertEqual(len(result["retrieved_chunks"]), 2)
        self.assertEqual(result["retrieved_chunks"][0].payload["chunk_id"], "c1")
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertIn("retrieved relevant excerpts", result["response"].text.lower())
        self.assertIn("answer generation has not yet been implemented", result["response"].text.lower())

    def test_clarification_does_not_call_retriever_and_has_needs_clarification(self) -> None:
        retriever = FakeRetriever()
        graph = build_graph(retriever).compile()
        message = UserMessage(text="details?", language=Language.HINDI)

        result = graph.invoke(make_state(message))

        self.assertEqual(retriever.calls, [])
        self.assertEqual(result["route"], "clarify")
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertEqual(result["response"].language, Language.HINDI)

    def test_graph_reaches_final_response(self) -> None:
        retriever = FakeRetriever((make_result("c1"),))
        graph = build_graph(retriever).compile()
        message = UserMessage(text="Which categories are excluded from PM-KISAN benefits?", language=Language.ENGLISH)

        result = graph.invoke(make_state(message))

        self.assertIsNotNone(result["response"])
        self.assertTrue(len(result["response"].text) > 0)
        self.assertEqual(result["response"].citations, ())


class WeatherRouteTests(unittest.TestCase):
    def test_weather_intent_routes_to_weather(self) -> None:
        retriever = FakeRetriever()
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client).compile()
        message = UserMessage(
            text="What is the weather forecast for my field today?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "weather")
        self.assertEqual(result["retrieved_chunks"], ())
        self.assertEqual(retriever.calls, [])

    def test_hindi_weather_intent_routes_to_weather(self) -> None:
        retriever = FakeRetriever()
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client).compile()
        message = UserMessage(
            text="मौसम का पूर्वानुमान क्या है",
            language=Language.HINDI,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "weather")
        self.assertEqual(len(weather_client.calls), 1)

    def test_weather_route_calls_client_once_with_message_coordinates(self) -> None:
        retriever = FakeRetriever()
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client).compile()
        message = UserMessage(
            text="Will it rain in my field tomorrow morning?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.209,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(len(weather_client.calls), 1)
        request = weather_client.calls[0][0]
        self.assertEqual(request.latitude, 28.6139)
        self.assertEqual(request.longitude, 77.209)
        self.assertIsInstance(request, WeatherRequest)
        self.assertEqual(retriever.calls, [])
        self.assertIsNotNone(result["weather"])

    def test_weather_response_text_uses_actual_weather_data(self) -> None:
        weather_data = make_weather_response(
            latitude=28.6139,
            longitude=77.2090,
            temperature_c=31.2,
            precipitation_mm=4.5,
            wind_speed_mps=8.4,
            weather_code=61,
        )
        weather_client = FakeWeatherClient(weather_data)
        graph = build_graph(FakeRetriever(), weather_client).compile()
        message = UserMessage(
            text="What is the weather like right now please?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertIs(result["weather"], weather_data)
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        text = result["response"].text
        self.assertIn("31.2", text)
        self.assertIn("4.5", text)
        self.assertIn("8.4", text)
        self.assertIn("61", text)
        self.assertIn("28.6139", text)
        self.assertIn("77.209", text)

    def test_missing_measurements_render_as_unavailable_not_zero(self) -> None:
        weather_data = make_weather_response(
            temperature_c=None,
            precipitation_mm=None,
            wind_speed_mps=None,
            weather_code=None,
        )
        weather_client = FakeWeatherClient(weather_data)
        graph = build_graph(FakeRetriever(), weather_client).compile()
        message = UserMessage(
            text="Please tell me the current temperature and rainfall.",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        text = result["response"].text
        self.assertEqual(text.count("unavailable"), 3)
        self.assertNotIn("0 °C", text)
        self.assertNotIn("weather code", text)

    def test_missing_coordinates_clarifies_without_fabricating_location(self) -> None:
        retriever = FakeRetriever()
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client).compile()
        message = UserMessage(
            text="What is the weather forecast for my field today?",
            language=Language.ENGLISH,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "weather")
        self.assertEqual(weather_client.calls, [])
        self.assertIsNone(result["weather"])
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)
        self.assertIn("latitude and longitude", result["response"].text)
        self.assertNotIn("28.6139", result["response"].text)

    def test_partial_coordinates_clarify_without_calling_client(self) -> None:
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(FakeRetriever(), weather_client).compile()
        message = UserMessage(
            text="What is the temperature in my village right now?",
            language=Language.ENGLISH,
            latitude=28.6139,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "weather")
        self.assertEqual(weather_client.calls, [])
        self.assertIsNone(result["weather"])
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_weather_route_without_injected_client_reports_unavailable(self) -> None:
        graph = build_graph(FakeRetriever()).compile()
        message = UserMessage(
            text="What is the weather forecast for my field today?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "weather")
        self.assertIsNone(result["weather"])
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)
        self.assertIn("unavailable", result["response"].text.lower())

    def test_weather_client_failure_reports_unavailable(self) -> None:
        weather_client = FakeWeatherClient(raise_exc=WeatherRequestError("fail"))
        graph = build_graph(FakeRetriever(), weather_client).compile()
        message = UserMessage(
            text="What is the weather forecast for my field today?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(len(weather_client.calls), 1)
        self.assertIsNone(result["weather"])
        self.assertEqual(result["response"].status, ResponseStatus.ABSTAINED)

    def test_graph_construction_does_not_call_weather_client(self) -> None:
        weather_client = FakeWeatherClient(make_weather_response())

        build_graph(FakeRetriever(), weather_client).compile()

        self.assertEqual(weather_client.calls, [])

    def test_weather_route_reaches_only_the_injected_client(self) -> None:
        weather_client = FakeWeatherClient(make_weather_response())
        message = UserMessage(
            text="What is the weather forecast for my field today?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        with patch(
            "kisansathi.weather.client._StdlibTransport.get_json",
            side_effect=AssertionError("network access attempted"),
        ) as transport_get_json:
            graph = build_graph(FakeRetriever(), weather_client).compile()
            result = graph.invoke(make_state(message))

        self.assertEqual(transport_get_json.call_count, 0)
        self.assertEqual(len(weather_client.calls), 1)
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)

    def test_real_client_is_usable_through_the_injected_transport(self) -> None:
        message = UserMessage(
            text="What is the weather forecast for my field today?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )
        transport = FakeTransport(response=OPEN_METEO_PAYLOAD)
        graph = build_graph(FakeRetriever(), OpenMeteoClient(transport=transport)).compile()

        result = graph.invoke(make_state(message))

        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][1]["latitude"], 28.6139)
        self.assertEqual(transport.calls[0][1]["longitude"], 77.2090)
        self.assertEqual(result["weather"].current.temperature_c, 24.6)
        self.assertIn("24.6", result["response"].text)


class CrossRouteIsolationTests(unittest.TestCase):
    def test_weather_client_not_called_on_retrieval_route(self) -> None:
        retriever = FakeRetriever((make_result("c1"),))
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client).compile()
        message = UserMessage(
            text="What documents are required for PM-KISAN enrollment?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(weather_client.calls, [])
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(result["retrieved_chunks"]), 1)

    def test_weather_client_not_called_on_clarification_route(self) -> None:
        retriever = FakeRetriever()
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client).compile()
        message = UserMessage(
            text="Help",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "clarify")
        self.assertEqual(weather_client.calls, [])
        self.assertEqual(retriever.calls, [])
        self.assertEqual(result["response"].status, ResponseStatus.NEEDS_CLARIFICATION)

    def test_underspecified_weather_text_takes_clarification_route(self) -> None:
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(FakeRetriever(), weather_client).compile()
        message = UserMessage(
            text="rain?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "clarify")
        self.assertEqual(weather_client.calls, [])

    def test_retrieval_behavior_unchanged_with_weather_client_injected(self) -> None:
        retriever = FakeRetriever((make_result("c1"), make_result("c2")))
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client).compile()
        message = UserMessage(
            text="When was PM-KISAN launched and how often are payments made?",
            language=Language.ENGLISH,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0][1], 5)
        self.assertEqual(len(result["retrieved_chunks"]), 2)
        self.assertEqual(result["response"].status, ResponseStatus.ANSWERED)
        self.assertIn("retrieved relevant excerpts", result["response"].text.lower())
        self.assertIn(
            "answer generation has not yet been implemented",
            result["response"].text.lower(),
        )
        self.assertEqual(result["response"].citations, ())

    def test_weather_keywords_do_not_match_agricultural_terms(self) -> None:
        retriever = FakeRetriever((make_result("c1"),))
        weather_client = FakeWeatherClient(make_weather_response())
        graph = build_graph(retriever, weather_client).compile()
        message = UserMessage(
            text="Does the scheme cover training for grain storage and drainage work?",
            language=Language.ENGLISH,
            latitude=28.6139,
            longitude=77.2090,
        )

        result = graph.invoke(make_state(message))

        self.assertEqual(result["route"], "retrieval")
        self.assertEqual(weather_client.calls, [])


if __name__ == "__main__":
    unittest.main()