"""Deterministic LangGraph orchestration skeleton."""

from enum import StrEnum
from typing import NotRequired, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from kisansathi.domain.schemas import (
    AssistantResponse,
    Citation,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.weather.exceptions import WeatherError
from kisansathi.weather.models import WeatherCurrent, WeatherRequest, WeatherResponse


class Route(StrEnum):
    """The routes the request may take out of ``route_request``."""

    RETRIEVAL = "retrieval"
    WEATHER = "weather"
    CLARIFY = "clarify"


_ROUTE_NODES: dict[str, str] = {
    Route.RETRIEVAL: "retrieve",
    Route.WEATHER: "weather",
    Route.CLARIFY: "clarify",
}

_TOKEN_STRIP = ".,?!:;\"'()[]{}-"

_WEATHER_KEYWORDS = frozenset(
    {
        "weather",
        "forecast",
        "rain",
        "rainfall",
        "temperature",
        "मौसम",
        "बारिश",
        "तापमान",
    }
)


class OrchestrationState(TypedDict):
    """Graph state for the orchestration skeleton."""

    message: UserMessage
    route: str
    retrieved_chunks: tuple[SearchResult, ...]
    response: AssistantResponse
    weather: NotRequired[WeatherResponse | None]


class _RetrieverProtocol:
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        raise NotImplementedError


class _WeatherClientProtocol(Protocol):
    def get_forecast(
        self,
        req: WeatherRequest,
        *,
        include_forecast: bool = False,
        forecast_days: int = 1,
    ) -> WeatherResponse:
        """Return current weather for the requested coordinates."""
        ...


def _is_underspecified(message: UserMessage) -> bool:
    """Return True if the user message is very short or underspecified."""
    text = message.text.strip()
    if len(text) < 10:
        return True
    words = [token for token in text.split() if token]
    if len(words) < 3:
        return True
    return False


def _has_weather_intent(message: UserMessage) -> bool:
    """Return True if the message matches a provisional weather keyword.

    Provisional by design: a fixed keyword set with no scoring, no stemming and no
    model. It exists so the weather route is reachable end to end, and it is expected
    to be replaced by a real intent classifier once one is available.
    """
    tokens = message.text.lower().split()
    return any(token.strip(_TOKEN_STRIP) in _WEATHER_KEYWORDS for token in tokens)


def _resolve_coordinates(message: UserMessage) -> tuple[float, float] | None:
    """Return caller-supplied coordinates, or None when they are unavailable.

    Coordinates are never inferred or defaulted here. There is no geocoding yet, and
    inventing a location would silently report another farmer's weather, so a partial
    or absent pair is reported as unavailable and routed to clarification.
    """
    latitude = message.latitude
    longitude = message.longitude
    if latitude is None or longitude is None:
        return None
    return float(latitude), float(longitude)


def _build_clarification_response(message: UserMessage) -> AssistantResponse:
    """Create a deterministic clarification response."""
    language = message.language or Language.ENGLISH
    text = "Could you please provide more specific details so I can help?"
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.NEEDS_CLARIFICATION,
        citations=(),
    )


def _build_retrieval_response(message: UserMessage, chunk_count: int) -> AssistantResponse:
    """Create a response noting that retrieval occurred but generation is not implemented."""
    language = message.language or Language.ENGLISH
    if chunk_count <= 0:
        text = (
            "I retrieved relevant excerpts from the available sources, "
            "but answer generation has not yet been implemented."
        )
    else:
        text = (
            "I retrieved relevant excerpts from the available sources, "
            "but answer generation has not yet been implemented."
        )
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.ANSWERED,
        citations=(),
    )


def _build_weather_location_request(message: UserMessage) -> AssistantResponse:
    """Ask for a location when the request carries no coordinates."""
    language = message.language or Language.ENGLISH
    text = (
        "I can share the weather once I know where you are. "
        "Please provide your latitude and longitude, or your district and village name."
    )
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.NEEDS_CLARIFICATION,
        citations=(),
    )


def _build_weather_unavailable_response(message: UserMessage) -> AssistantResponse:
    """Report that weather data could not be retrieved at all."""
    language = message.language or Language.ENGLISH
    text = "Weather data is currently unavailable. Please try again later."
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.ABSTAINED,
        citations=(),
    )


def _format_measurement(value: float | None, unit: str = "") -> str:
    """Render a measured value, or say plainly that it was not reported."""
    if value is None:
        return "unavailable"
    return f"{value:g}{unit}"


def _build_weather_response(
    message: UserMessage,
    weather: WeatherResponse,
) -> AssistantResponse:
    """Render a weather response from actual WeatherResponse data.

    Only values reported by the weather tool are shown. No agricultural advice is
    generated here, and values the tool did not report are rendered as ``unavailable``
    rather than as zero.
    """
    language = message.language or Language.ENGLISH
    current: WeatherCurrent = weather.current
    text = (
        f"Weather at {weather.latitude:g}, {weather.longitude:g}: "
        f"temperature {_format_measurement(current.temperature_c, ' °C')}, "
        f"precipitation {_format_measurement(current.precipitation_mm, ' mm')}, "
        f"wind speed {_format_measurement(current.wind_speed_mps, ' m/s')}"
    )
    if current.weather_code is not None:
        text += f", weather code {current.weather_code}"
    return AssistantResponse(
        text=text + ".",
        language=language,
        status=ResponseStatus.ANSWERED,
        citations=(),
    )


def build_graph(
    retriever: _RetrieverProtocol,
    weather_client: _WeatherClientProtocol | None = None,
) -> StateGraph:
    """Build the deterministic orchestration graph.

    The weather client is injected rather than held in graph state, so building and
    compiling the graph performs no I/O. Passing ``weather_client=None`` keeps the
    weather route reachable but answers it without calling a client.
    """

    graph = StateGraph(OrchestrationState)

    def route_request(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        if _is_underspecified(message):
            return {**state, "route": Route.CLARIFY}
        if _has_weather_intent(message):
            return {**state, "route": Route.WEATHER}
        return {**state, "route": Route.RETRIEVAL}

    def retrieve(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        results = retriever.retrieve(message.text, top_k=5)
        return {**state, "retrieved_chunks": tuple(results)}

    def weather(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        coordinates = _resolve_coordinates(message)
        if coordinates is None:
            return {
                **state,
                "response": _build_weather_location_request(message),
                "weather": None,
                "retrieved_chunks": (),
            }
        if weather_client is None:
            return {
                **state,
                "response": _build_weather_unavailable_response(message),
                "weather": None,
                "retrieved_chunks": (),
            }
        latitude, longitude = coordinates
        try:
            result = weather_client.get_forecast(
                WeatherRequest(latitude=latitude, longitude=longitude)
            )
        except (WeatherError, TypeError, ValueError):
            return {
                **state,
                "response": _build_weather_unavailable_response(message),
                "weather": None,
                "retrieved_chunks": (),
            }
        return {**state, "weather": result, "retrieved_chunks": ()}

    def clarify(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        response = _build_clarification_response(message)
        return {**state, "response": response, "retrieved_chunks": ()}

    def finalize_response(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        route = state.get("route")
        if route == Route.CLARIFY:
            existing_response = state.get("response")
            if existing_response is None:
                existing_response = _build_clarification_response(message)
            return {**state, "response": existing_response}
        if route == Route.WEATHER:
            weather_result = state.get("weather")
            if weather_result is None:
                existing_response = state.get("response")
                if existing_response is None:
                    existing_response = _build_weather_location_request(message)
                return {**state, "response": existing_response}
            return {**state, "response": _build_weather_response(message, weather_result)}
        retrieved = state.get("retrieved_chunks") or ()
        response = _build_retrieval_response(message, len(retrieved))
        return {**state, "response": response}

    graph.add_node("route_request", route_request)
    graph.add_node("retrieve", retrieve)
    graph.add_node("weather", weather)
    graph.add_node("clarify", clarify)
    graph.add_node("finalize_response", finalize_response)

    graph.add_edge(START, "route_request")
    graph.add_conditional_edges(
        "route_request",
        lambda state: state["route"],
        _ROUTE_NODES,
    )
    graph.add_edge("retrieve", "finalize_response")
    graph.add_edge("weather", "finalize_response")
    graph.add_edge("clarify", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph