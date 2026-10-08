"""Command-line entry point that builds the dense and BM25 indexes.

Both indexes are derived artifacts: they are rebuilt from the tracked
processed corpus under ``data/processed`` and are never committed. Run from
the project root after ingestion (and after any corpus change):

    python -m kisansathi.build_indexes            # dense (Qdrant) + BM25
    python -m kisansathi.build_indexes --bm25-only
    python -m kisansathi.build_indexes --dense-only

Dense indexing loads the embedding model (downloading weights on first use).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from kisansathi.config import Settings
from kisansathi.logging_setup import configure_logging

DEFAULT_CORPUS_ROOT = "data/processed"


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """Aggregate result of one index build run."""

    corpus_root: str
    files_indexed: int
    chunks_read: int
    dense_embedded: int
    dense_upserted: int
    bm25_indexed: int
    errors: tuple[str, ...]


def find_chunk_files(corpus_root: str | Path) -> tuple[Path, ...]:
    """Return the processed JSONL files under a corpus root, sorted by path."""
    root = Path(corpus_root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root does not exist: {root}")
    return tuple(sorted(root.rglob("*.jsonl")))


def run_build(
    corpus_root: str | Path,
    *,
    embedding_service: object | None = None,
    vector_store: object | None = None,
    bm25_store: object | None = None,
    batch_size: int = 32,
) -> BuildSummary:
    """Index every processed JSONL file into the provided stores.

    ``embedding_service`` and ``vector_store`` must be provided together to
    build the dense index; ``bm25_store`` builds the lexical index. At least
    one target is required. Stores are caller-owned and stay open.
    """
    if (embedding_service is None) != (vector_store is None):
        raise ValueError("dense indexing requires both an embedding service and a vector store")
    if embedding_service is None and bm25_store is None:
        raise ValueError("at least one index target is required")

    files = find_chunk_files(corpus_root)
    if not files:
        raise ValueError(f"no processed .jsonl files found under {Path(corpus_root)}")

    errors: list[str] = []
    chunks_read = 0
    dense_embedded = 0
    dense_upserted = 0
    bm25_indexed = 0

    if embedding_service is not None and vector_store is not None:
        from kisansathi.retrieval.indexer import CorpusIndexer

        indexer = CorpusIndexer(embedding_service, vector_store, batch_size=batch_size)
        for path in files:
            summary = indexer.index_file(path)
            chunks_read += summary.chunks_read
            dense_embedded += summary.embedded
            dense_upserted += summary.upserted
            errors.extend(f"{path}: {error}" for error in summary.errors)

    if bm25_store is not None:
        for path in files:
            summary = bm25_store.index_file(path)
            if embedding_service is None:
                chunks_read += summary.chunks_read
            bm25_indexed += summary.indexed
            errors.extend(f"{path}: {error}" for error in summary.errors)

    return BuildSummary(
        corpus_root=str(corpus_root),
        files_indexed=len(files),
        chunks_read=chunks_read,
        dense_embedded=dense_embedded,
        dense_upserted=dense_upserted,
        bm25_indexed=bm25_indexed,
        errors=tuple(errors),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the dense (Qdrant) and BM25 indexes from processed ingestion JSONL"
    )
    parser.add_argument(
        "--corpus-root",
        default=DEFAULT_CORPUS_ROOT,
        help=f"Directory of processed JSONL files (default: {DEFAULT_CORPUS_ROOT})",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Build only the dense Qdrant index (loads the embedding model)",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Build only the SQLite FTS5 BM25 index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size for dense indexing (default: 32)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    configure_logging()
    arguments = build_parser().parse_args(argv)
    if arguments.dense_only and arguments.bm25_only:
        print("--dense-only and --bm25-only cannot be combined", file=sys.stderr)
        return 2
    if arguments.batch_size < 1:
        print("--batch-size must be a positive integer", file=sys.stderr)
        return 2

    dense_enabled = not arguments.bm25_only
    bm25_enabled = not arguments.dense_only

    try:
        settings = Settings.from_env()
        with ExitStack() as resources:
            embedding_service = None
            vector_store = None
            bm25_store = None
            if dense_enabled:
                from kisansathi.retrieval.embeddings import EmbeddingService
                from kisansathi.retrieval.vector_store import QdrantVectorStore

                embedding_service = EmbeddingService(settings)
                vector_store = resources.enter_context(QdrantVectorStore(settings))
            if bm25_enabled:
                from kisansathi.retrieval.bm25_store import BM25Store

                bm25_store = resources.enter_context(BM25Store(settings))
            summary = run_build(
                arguments.corpus_root,
                embedding_service=embedding_service,
                vector_store=vector_store,
                bm25_store=bm25_store,
                batch_size=arguments.batch_size,
            )
    except (FileNotFoundError, ValueError) as error:
        print(f"Index build failed: {error}", file=sys.stderr)
        return 1

    print(f"corpus: {summary.corpus_root} ({summary.files_indexed} files)")
    print(f"chunks read: {summary.chunks_read}")
    if dense_enabled:
        print(f"dense: embedded={summary.dense_embedded} upserted={summary.dense_upserted}")
    if bm25_enabled:
        print(f"bm25: indexed={summary.bm25_indexed}")
    if summary.errors:
        for error in summary.errors:
            print(f"error: {error}", file=sys.stderr)
        print("Index build finished with errors", file=sys.stderr)
        return 1
    print("Index build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
