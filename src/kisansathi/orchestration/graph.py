"""Deterministic LangGraph orchestration skeleton."""

from collections.abc import Mapping
from enum import StrEnum
from typing import NotRequired, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from kisansathi.citations.models import CitationBatch, CitationError, RejectedCitation
from kisansathi.citations.resolver import CitationResolver
from kisansathi.domain.schemas import (
    AssistantResponse,
    Citation,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.eligibility.evaluator import evaluate
from kisansathi.eligibility.models import (
    FACT_NAMES,
    EligibilityDecision,
    EligibilityError,
    EligibilityRequest,
    EligibilityStatus,
)
from kisansathi.retrieval.vector_store import SearchResult
from kisansathi.weather.exceptions import WeatherError
from kisansathi.weather.models import WeatherCurrent, WeatherRequest, WeatherResponse


class Route(StrEnum):
    """The routes the request may take out of ``route_request``."""

    RETRIEVAL = "retrieval"
    ELIGIBILITY = "eligibility"
    WEATHER = "weather"
    CLARIFY = "clarify"


_ROUTE_NODES: dict[str, str] = {
    Route.RETRIEVAL: "retrieve",
    Route.ELIGIBILITY: "eligibility",
    Route.WEATHER: "weather",
    Route.CLARIFY: "clarify",
}

_TOKEN_STRIP = ".,?!:;\"'()[]{}-"

# Marker for provenance that failed to resolve before it could be attributed to any
# source. It is only ever a rejection reason, never a citation.
_UNRESOLVED_SOURCE = "<unresolved>"

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

# Deliberately narrow. Terms such as "enrollment", "excluded" and "categories" are
# excluded because they also occur in document questions that must keep reaching
# retrieval rather than the eligibility evaluator.
_ELIGIBILITY_KEYWORDS = frozenset(
    {
        "eligible",
        "eligibility",
        "योग्य",
        "पात्र",
        "पात्रता",
    }
)


class OrchestrationState(TypedDict):
    """Graph state for the orchestration skeleton."""

    message: UserMessage
    route: str
    retrieved_chunks: tuple[SearchResult, ...]
    response: AssistantResponse
    weather: NotRequired[WeatherResponse | None]
    citations: NotRequired[CitationBatch | None]
    eligibility_decision: NotRequired[EligibilityDecision | None]
    eligibility_request: NotRequired[EligibilityRequest | None]


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


class _EligibilityEvaluatorProtocol(Protocol):
    def __call__(
        self, request: EligibilityRequest, *, rules: object = ...
    ) -> EligibilityDecision:
        """Evaluate one eligibility request and return its decision."""
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


def _has_eligibility_intent(message: UserMessage) -> bool:
    """Return True if the message matches a provisional eligibility keyword.

    Provisional for the same reason as ``_has_weather_intent``: a fixed keyword set with
    no scoring, no stemming and no model. Matching an intent does not imply an answer is
    available, because eligibility still needs structured facts from the caller.
    """
    tokens = message.text.lower().split()
    return any(token.strip(_TOKEN_STRIP) in _ELIGIBILITY_KEYWORDS for token in tokens)


def _coerce_eligibility_request(value: object) -> EligibilityRequest | None:
    """Return a trustworthy EligibilityRequest, or None when the input cannot supply one.

    Facts are never inferred. Only the four conditions in ``FACT_NAMES`` are read, every
    non-boolean is treated as not supplied, and the request is rebuilt through
    ``EligibilityRequest`` so its own validation still applies. A malformed request is
    reported as absent rather than partially trusted.
    """
    if isinstance(value, EligibilityRequest):
        scheme: object = value.scheme
        supplied = {fact: getattr(value, fact) for fact in FACT_NAMES}
    elif isinstance(value, Mapping):
        scheme = value.get("scheme")
        supplied = {fact: value.get(fact) for fact in FACT_NAMES}
    else:
        return None

    if not isinstance(scheme, str) or not scheme.strip():
        return None

    facts = {
        fact: item if isinstance(item, bool) else None
        for fact, item in supplied.items()
    }
    try:
        return EligibilityRequest(scheme=scheme, **facts)
    except EligibilityError:
        return None


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


def _build_retrieval_response(
    message: UserMessage,
    chunk_count: int,
    batch: CitationBatch | None = None,
) -> AssistantResponse:
    """Create a response noting that retrieval occurred but generation is not implemented.

    Provenance that resolved is attached, so the citations already point at real pages of
    registered documents. When every retrieved chunk was refused, no claim can be
    attributed to anything, so the response abstains instead of citing nothing while
    still sounding answered.
    """
    language = message.language or Language.ENGLISH
    citations: tuple[Citation, ...] = batch.citations if batch is not None else ()
    if chunk_count > 0 and not citations and batch is not None and batch.rejected:
        text = (
            "I could not verify the sources behind these excerpts, "
            "so I am not citing them."
        )
        return AssistantResponse(
            text=text,
            language=language,
            status=ResponseStatus.ABSTAINED,
            citations=(),
        )
    text = (
        "I retrieved relevant excerpts from the available sources, "
        "but answer generation has not yet been implemented."
    )
    return AssistantResponse(
        text=text,
        language=language,
        status=ResponseStatus.ANSWERED,
        citations=citations,
    )


def _build_eligibility_response(
    message: UserMessage,
    decision: EligibilityDecision | None,
    batch: CitationBatch | None,
) -> AssistantResponse:
    """Render an eligibility outcome and its citations, refusing unciteable verdicts.

    The decision summary is deterministic text from the rule set, not generated prose, so
    it is safe to state directly. A verdict that cannot cite its own evidence is withheld:
    reporting "you are not eligible" without a source would be an unauditable claim about
    a farmer's benefit.
    """
    language = message.language or Language.ENGLISH
    citations: tuple[Citation, ...] = batch.citations if batch is not None else ()

    if decision is None:
        return AssistantResponse(
            text=(
                "I cannot determine eligibility without the required facts. "
                "Please provide the following so I can check them against the official "
                f"guidelines: {', '.join(FACT_NAMES)}."
            ),
            language=language,
            status=ResponseStatus.NEEDS_CLARIFICATION,
            citations=(),
        )

    if decision.status is EligibilityStatus.UNSUPPORTED_SCHEME:
        return AssistantResponse(
            text=decision.summary,
            language=language,
            status=ResponseStatus.ABSTAINED,
            citations=(),
        )

    if decision.status is EligibilityStatus.INSUFFICIENT_INFORMATION:
        return AssistantResponse(
            text=(
                f"{decision.summary} "
                f"Please provide the following: {', '.join(decision.missing_facts)}."
            ),
            language=language,
            status=ResponseStatus.NEEDS_CLARIFICATION,
            citations=(),
        )

    if not citations:
        return AssistantResponse(
            text=(
                "I reached a determination, but I could not verify the official documents "
                "behind it, so I am withholding it rather than stating it without a source."
            ),
            language=language,
            status=ResponseStatus.ABSTAINED,
            citations=(),
        )

    return AssistantResponse(
        text=(
            f"{decision.summary} "
            "This determination is based only on the official documents cited below."
        ),
        language=language,
        status=ResponseStatus.ANSWERED,
        citations=citations,
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
    *,
    citation_resolver: CitationResolver | None = None,
    eligibility_evaluator: _EligibilityEvaluatorProtocol | None = None,
) -> StateGraph:
    """Build the deterministic orchestration graph.

    Collaborators are injected rather than held in graph state, so building and compiling
    the graph performs no I/O. Passing ``weather_client=None`` keeps the weather route
    reachable but answers it without calling a client. Passing ``citation_resolver=None``
    keeps the retrieval and eligibility routes reachable but publishes no citations, which
    is the original behaviour. Both defaults leave those routes unable to fabricate
    anything, since citations are only ever produced by the injected resolver.
    """

    evaluator = evaluate if eligibility_evaluator is None else eligibility_evaluator

    graph = StateGraph(OrchestrationState)

    def route_request(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        if _is_underspecified(message):
            return {**state, "route": Route.CLARIFY}
        if _has_weather_intent(message):
            return {**state, "route": Route.WEATHER}
        if _has_eligibility_intent(message):
            return {**state, "route": Route.ELIGIBILITY}
        return {**state, "route": Route.RETRIEVAL}

    def retrieve(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        results = retriever.retrieve(message.text, top_k=5)
        return {**state, "retrieved_chunks": tuple(results)}

    def eligibility(state: OrchestrationState) -> OrchestrationState:
        """Evaluate caller-supplied facts, or ask for them when none were supplied.

        Facts come from ``eligibility_request`` only. Nothing here parses the message
        text, because guessing a farmer's circumstances would silently produce a wrong
        verdict about a real benefit.
        """
        request = _coerce_eligibility_request(state.get("eligibility_request"))
        if request is None:
            return {
                **state,
                "eligibility_decision": None,
                "retrieved_chunks": (),
            }
        try:
            decision = evaluator(request)
        except EligibilityError:
            decision = None
        return {**state, "eligibility_decision": decision, "retrieved_chunks": ()}

    def resolve_citations(state: OrchestrationState) -> OrchestrationState:
        """Join resolved provenance into citations, or record that nothing resolved.

        Both evidence routes converge here so citation rules live in one place and are
        never duplicated per route. Only retrieval payloads and eligibility evidence are
        handed to the resolver, never source text.

        An empty batch with no rejections means citation resolution is not configured, so
        the routes keep their original un-cited behaviour. Anything the resolver refused,
        including a resolver that failed outright, is recorded as a rejection so the
        response abstains instead of claiming to have sourced something.
        """
        if citation_resolver is None:
            return {**state, "citations": CitationBatch()}

        route = state.get("route")
        try:
            if route == Route.RETRIEVAL:
                chunks = state.get("retrieved_chunks") or ()
                batch = citation_resolver.resolve_payloads(
                    result.payload for result in chunks
                )
            elif route == Route.ELIGIBILITY:
                decision = state.get("eligibility_decision")
                if decision is None:
                    batch = CitationBatch()
                else:
                    batch = citation_resolver.resolve_evidence_batch(decision.evidence)
            else:
                batch = CitationBatch()
        except CitationError as error:
            batch = CitationBatch(
                rejected=(
                    RejectedCitation(
                        source_id=_UNRESOLVED_SOURCE,
                        chunk_id=None,
                        reason=f"citation resolution failed: {error}",
                    ),
                )
            )
        return {**state, "citations": batch}

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
        if route == Route.ELIGIBILITY:
            response = _build_eligibility_response(
                message,
                state.get("eligibility_decision"),
                state.get("citations"),
            )
            return {**state, "response": response}
        retrieved = state.get("retrieved_chunks") or ()
        response = _build_retrieval_response(
            message,
            len(retrieved),
            state.get("citations"),
        )
        return {**state, "response": response}

    graph.add_node("route_request", route_request)
    graph.add_node("retrieve", retrieve)
    graph.add_node("eligibility", eligibility)
    graph.add_node("resolve_citations", resolve_citations)
    graph.add_node("weather", weather)
    graph.add_node("clarify", clarify)
    graph.add_node("finalize_response", finalize_response)

    graph.add_edge(START, "route_request")
    graph.add_conditional_edges(
        "route_request",
        lambda state: state["route"],
        _ROUTE_NODES,
    )
    graph.add_edge("retrieve", "resolve_citations")
    graph.add_edge("eligibility", "resolve_citations")
    graph.add_edge("resolve_citations", "finalize_response")
    graph.add_edge("weather", "finalize_response")
    graph.add_edge("clarify", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph
