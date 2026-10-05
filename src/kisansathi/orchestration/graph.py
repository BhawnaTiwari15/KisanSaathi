"""Deterministic LangGraph orchestration skeleton."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from kisansathi.domain.schemas import (
    AssistantResponse,
    Citation,
    Language,
    ResponseStatus,
    UserMessage,
)
from kisansathi.retrieval.vector_store import SearchResult


class OrchestrationState(TypedDict):
    """Graph state for the orchestration skeleton."""

    message: UserMessage
    route: str
    retrieved_chunks: tuple[SearchResult, ...]
    response: AssistantResponse


class _RetrieverProtocol:
    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        raise NotImplementedError


def _is_underspecified(message: UserMessage) -> bool:
    """Return True if the user message is very short or underspecified."""
    text = message.text.strip()
    if len(text) < 10:
        return True
    words = [token for token in text.split() if token]
    if len(words) < 3:
        return True
    return False


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


def build_graph(retriever: _RetrieverProtocol) -> StateGraph:
    """Build the deterministic orchestration graph."""

    graph = StateGraph(OrchestrationState)

    def route_request(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        if _is_underspecified(message):
            return {**state, "route": "clarify"}
        return {**state, "route": "retrieval"}

    def retrieve(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        results = retriever.retrieve(message.text, top_k=5)
        return {**state, "retrieved_chunks": tuple(results)}

    def clarify(state: OrchestrationState) -> OrchestrationState:
        message = state["message"]
        response = _build_clarification_response(message)
        return {**state, "response": response, "retrieved_chunks": ()}

    def finalize_response(state: OrchestrationState) -> OrchestrationState:
        route = state.get("route")
        if route == "clarify":
            existing_response = state.get("response")
            if existing_response is None:
                existing_response = _build_clarification_response(state["message"])
            return {**state, "response": existing_response}
        retrieved = state.get("retrieved_chunks") or ()
        response = _build_retrieval_response(state["message"], len(retrieved))
        return {**state, "response": response}

    graph.add_node("route_request", route_request)
    graph.add_node("retrieve", retrieve)
    graph.add_node("clarify", clarify)
    graph.add_node("finalize_response", finalize_response)

    graph.add_edge(START, "route_request")
    graph.add_conditional_edges(
        "route_request",
        lambda state: state["route"],
        {"retrieval": "retrieve", "clarify": "clarify"},
    )
    graph.add_edge("retrieve", "finalize_response")
    graph.add_edge("clarify", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph
