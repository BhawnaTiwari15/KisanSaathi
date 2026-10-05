"""Index extracted JSONL chunks into the dense vector store."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from kisansathi.retrieval.processed_chunks import read_chunk_payloads


class _EmbeddingService(Protocol):
    def embed_documents(self, documents: Sequence[str]) -> np.ndarray: ...


class _VectorStore(Protocol):
    def upsert(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Mapping[str, object]],
    ) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class IndexingSummary:
    chunks_read: int = 0
    embedded: int = 0
    upserted: int = 0
    skipped_records: int = 0
    errors: tuple[str, ...] = ()


class CorpusIndexer:
    """Read one ingestion JSONL file and index only its chunk records."""

    def __init__(
        self,
        embedding_service: _EmbeddingService,
        vector_store: _VectorStore,
        *,
        batch_size: int = 32,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._batch_size = batch_size

    def index_file(self, jsonl_path: str | Path) -> IndexingSummary:
        """Index the chunk records in one JSONL file and summarize the result."""
        parsed = read_chunk_payloads(jsonl_path)
        errors = list(parsed.errors)
        chunk_payloads = parsed.payloads

        embedded = 0
        upserted = 0
        for start in range(0, len(chunk_payloads), self._batch_size):
            batch = chunk_payloads[start : start + self._batch_size]
            documents = [str(payload["text"]) for payload in batch]
            try:
                vectors = np.asarray(self._embedding_service.embed_documents(documents))
                if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                    raise ValueError(
                        "embedding output must have shape "
                        f"({len(batch)}, dimension); received {vectors.shape}"
                    )
                if not np.isfinite(vectors).all():
                    raise ValueError("embedding output contains non-finite values")
            except Exception as error:
                errors.append(f"embedding batch at chunk {start + 1} failed: {error}")
                continue

            embedded += len(batch)
            try:
                point_ids = self._vector_store.upsert(vectors, batch)
            except Exception as error:
                errors.append(f"upsert batch at chunk {start + 1} failed: {error}")
                continue
            upserted += len(point_ids)
            if len(point_ids) != len(batch):
                errors.append(
                    f"upsert batch at chunk {start + 1} returned {len(point_ids)} IDs "
                    f"for {len(batch)} chunks"
                )

        return IndexingSummary(
            chunks_read=parsed.chunks_read,
            embedded=embedded,
            upserted=upserted,
            skipped_records=parsed.skipped_records,
            errors=tuple(errors),
        )
