"""Composition root for the KisanSaathi Streamlit application.

Assembles injectable dependencies and builds the LangGraph application service.
Every dependency has a production default; each can be overridden at the call
site, including explicitly to ``None``. A sentinel distinguishes "not
provided" (use the default) from "explicitly disabled" (pass ``None``).

Typical usage in Streamlit:

    from kisansathi.ui.composition_root import build_application_service

    @st.cache_resource
    def load_app():
        return build_application_service()

    app = load_app()
    result = app.graph.invoke(state)

Building the service performs local I/O (opening the Qdrant collection), and
the embedding model loads lazily on the first query, so the UI caches one
instance per process with ``st.cache_resource``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kisansathi.citations.registry import ManifestSourceRegistry
from kisansathi.citations.resolver import CitationResolver
from kisansathi.config import Settings
from kisansathi.eligibility.evaluator import evaluate
from kisansathi.language import DeterministicLanguageDetector
from kisansathi.orchestration.graph import build_graph
from kisansathi.retrieval.embeddings import EmbeddingService
from kisansathi.retrieval.retriever import DenseRetriever
from kisansathi.retrieval.vector_store import QdrantVectorStore
from kisansathi.vision.providers import create_vision_analyzer
from kisansathi.weather.client import OpenMeteoClient

logger = logging.getLogger(__name__)

# Sentinel meaning "this dependency was not provided, so build the default".
_AUTO: Any = object()

DEFAULT_SOURCES_MANIFEST = Path(__file__).resolve().parents[3] / "data" / "sources.json"


@dataclass(frozen=True, slots=True)
class ApplicationService:
    """The assembled application: a compiled LangGraph ready to invoke."""

    graph: Any


def _default_retriever(settings: Settings) -> DenseRetriever:
    return DenseRetriever(EmbeddingService(settings), QdrantVectorStore(settings))


def _default_citation_resolver(manifest_path: Path | None = None) -> CitationResolver | None:
    path = manifest_path if manifest_path is not None else DEFAULT_SOURCES_MANIFEST
    try:
        registry = ManifestSourceRegistry.from_manifest(path)
    except Exception:
        logger.warning(
            "citation resolver disabled: source manifest %s could not be loaded",
            path,
            exc_info=True,
        )
        return None
    return CitationResolver(registry)


def build_application_service(
    settings: Settings | None = None,
    *,
    retriever: Any = _AUTO,
    weather_client: Any = _AUTO,
    citation_resolver: Any = _AUTO,
    eligibility_evaluator: Any = _AUTO,
    answer_generator: Any = _AUTO,
    language_detector: Any = _AUTO,
    speech_to_text: Any = _AUTO,
    vision_analyzer: Any = _AUTO,
) -> ApplicationService:
    """Build and return the LangGraph application service.

    Args:
        settings: Project settings. If ``None``, loads ``Settings.from_env()``.
        retriever: Required retrieval dependency. Defaults to a dense retriever
            over the configured local Qdrant collection.
        weather_client: Defaults to the Open-Meteo client. Pass ``None`` to
            keep the weather route reachable without a client (guardrails then
            report weather as unavailable).
        citation_resolver: Defaults to a resolver over ``data/sources.json``
            (overridable with ``KISANSAATHI_SOURCES_MANIFEST``). Pass ``None``
            to publish no citations (the manifest is a tracked file, so this
            only fails when the checkout is incomplete).
        eligibility_evaluator: Defaults to the deterministic PM-KISAN
            evaluator. Pass ``None`` to fall back to the graph default.
        answer_generator: Defaults to ``None`` (deterministic placeholder
            responses) because no production LLM provider is configured.
        language_detector: Defaults to the deterministic script-based detector.
        speech_to_text: Defaults to ``None`` (no production speech provider is
            configured); the graph then leaves text input unchanged.
        vision_analyzer: Defaults to the configured vision provider factory
            (``fake`` unless ``KISANSAATHI_VISION_PROVIDER`` says otherwise).

    Returns:
        An ``ApplicationService`` whose ``graph`` attribute is the compiled
        ``StateGraph``, invokable via ``graph.invoke(state)``.
    """
    if settings is None:
        settings = Settings.from_env()

    if retriever is _AUTO:
        retriever = _default_retriever(settings)
    if retriever is None:
        raise ValueError("retriever is required: the graph calls retriever.retrieve()")

    if weather_client is _AUTO:
        weather_client = OpenMeteoClient()
    if citation_resolver is _AUTO:
        manifest_path = (
            Path(settings.sources_manifest_path)
            if settings.sources_manifest_path is not None
            else None
        )
        citation_resolver = _default_citation_resolver(manifest_path)
    if eligibility_evaluator is _AUTO:
        eligibility_evaluator = evaluate
    if answer_generator is _AUTO:
        answer_generator = None
    if language_detector is _AUTO:
        language_detector = DeterministicLanguageDetector()
    if speech_to_text is _AUTO:
        speech_to_text = None
    if vision_analyzer is _AUTO:
        vision_analyzer = create_vision_analyzer(settings)

    graph = build_graph(
        retriever=retriever,
        weather_client=weather_client,
        citation_resolver=citation_resolver,
        eligibility_evaluator=eligibility_evaluator,
        answer_generator=answer_generator,
        language_detector=language_detector,
        speech_to_text=speech_to_text,
        vision_analyzer=vision_analyzer,
    ).compile()

    return ApplicationService(graph=graph)
