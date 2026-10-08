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

import atexit
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kisansathi.build_indexes import run_build
from kisansathi.citations.registry import ManifestSourceRegistry
from kisansathi.citations.resolver import CitationResolver
from kisansathi.config import LLMProvider, Settings
from kisansathi.eligibility.evaluator import evaluate
from kisansathi.generation import DefaultAnswerGenerator
from kisansathi.generation.gemini import GeminiTextClient
from kisansathi.language import DeterministicLanguageDetector
from kisansathi.orchestration.graph import build_graph
from kisansathi.retrieval.bm25_store import BM25Store
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
    retriever: DenseRetriever

    def close(self) -> None:
        """Close the retriever and its vector store.

        Safe to call multiple times. Registered with ``atexit`` to run
        before Python module shutdown, avoiding Qdrant portalocker traceback
        on Windows.
        """
        if hasattr(self.retriever, "close"):
            self.retriever.close()


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


def _default_answer_generator(settings: Settings):
    """Build the production answer generator when LLM configuration is valid.

    Returns ``None`` unless a real Gemini provider with an API key is
    configured. A missing key must NOT silently fall back to a fake/no-op
    generator: the graph's deterministic placeholder keeps the safe
    failure/ABSTAINED behavior, and the application still starts.
    """
    if settings.llm_provider != LLMProvider.GEMINI or settings.llm_api_key is None:
        return None
    client = GeminiTextClient(settings)
    return DefaultAnswerGenerator(
        client,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def _ensure_indexes_exist(settings: Settings) -> None:
    """Build Qdrant and BM25 indexes if they do not already exist.

    Checks for the Qdrant collection and BM25 database content. If either is
    missing or empty, builds both indexes from the committed processed corpus
    using the shared build_indexes logic.
    """
    # Quick check without holding connections open
    with QdrantVectorStore(settings) as qdrant_store:
        collection_missing = not qdrant_store.collection_exists()

    bm25_path = Path(settings.bm25_storage_path)
    bm25_missing = not bm25_path.exists()

    if not collection_missing and not bm25_missing:
        logger.info("Qdrant and BM25 indexes already present; skipping build.")
        return

    logger.info("Indexes missing (qdrant=%s, bm25=%s); building from corpus...",
                collection_missing, bm25_missing)

    embedding_service = EmbeddingService(settings)
    with QdrantVectorStore(settings) as vector_store:
        with BM25Store(settings) as bm25_store:
            summary = run_build(
                "data/processed",
                embedding_service=embedding_service,
                vector_store=vector_store,
                bm25_store=bm25_store,
            )

    if summary.errors:
        logger.warning("Index build completed with errors: %s", summary.errors)
    else:
        logger.info("Index build complete: %s chunks embedded, %s BM25 indexed",
                    summary.dense_upserted, summary.bm25_indexed)


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
        answer_generator: Defaults to ``None`` unless a real Gemini LLM
            provider is configured (provider ``gemini`` plus
            ``KISANSAATHI_LLM_API_KEY``); the graph then keeps its
            deterministic safe-failure placeholder. A missing key never falls
            back to a fake/no-op generator.
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

    _ensure_indexes_exist(settings)

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
        answer_generator = _default_answer_generator(settings)
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

    service = ApplicationService(graph=graph, retriever=retriever)
    atexit.register(service.close)
    return service
