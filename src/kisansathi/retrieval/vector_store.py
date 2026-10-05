"""Local Qdrant vector storage for retrieved document chunks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from kisansathi.config import Settings


class VectorPayload(TypedDict):
    chunk_id: str
    source_id: str
    sha256: str
    scheme: str
    jurisdiction: str
    language: str
    title: str
    page_start: int
    page_end: int
    heading: str | None
    text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    score: float
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Point:
    point_id: str
    vector: list[float]
    payload: dict[str, object]


class _VectorBackend(Protocol):
    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(self, collection_name: str, vector_size: int) -> None: ...

    def upsert(self, collection_name: str, points: Sequence[_Point]) -> None: ...

    def search(
        self, collection_name: str, query_vector: Sequence[float], limit: int
    ) -> Sequence[SearchResult]: ...

    def close(self) -> None: ...


class _LocalQdrantBackend:
    def __init__(self, storage_path: str) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(path=storage_path)

    def collection_exists(self, collection_name: str) -> bool:
        return self._client.collection_exists(collection_name=collection_name)

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(self, collection_name: str, points: Sequence[_Point]) -> None:
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=point.point_id, vector=point.vector, payload=point.payload)
                for point in points
            ],
            wait=True,
        )

    def search(
        self, collection_name: str, query_vector: Sequence[float], limit: int
    ) -> Sequence[SearchResult]:
        response = self._client.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            limit=limit,
            with_payload=True,
        )
        return tuple(
            SearchResult(score=float(point.score), payload=dict(point.payload or {}))
            for point in response.points
        )

    def close(self) -> None:
        self._client.close()


class QdrantVectorStore:
    """Store and search dense vectors in a local Qdrant collection."""

    _PAYLOAD_FIELDS = frozenset(VectorPayload.__annotations__)

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: _VectorBackend | None = None,
    ) -> None:
        self._settings = settings if settings is not None else Settings.from_env()
        self._backend = backend or _LocalQdrantBackend(self._settings.qdrant_storage_path)
        self.ensure_collection()

    @property
    def collection_name(self) -> str:
        return self._settings.qdrant_collection_name

    @property
    def vector_dimension(self) -> int:
        return self._settings.embedding_dimension

    def collection_exists(self) -> bool:
        return self._backend.collection_exists(self.collection_name)

    def ensure_collection(self) -> bool:
        """Create the configured cosine collection if it is absent."""
        if self.collection_exists():
            return False
        self._backend.create_collection(self.collection_name, self.vector_dimension)
        return True

    @staticmethod
    def point_id_for_chunk(chunk_id: str) -> str:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        return str(uuid5(NAMESPACE_URL, f"kisansathi:chunk:{chunk_id}"))

    def upsert(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Mapping[str, object]],
    ) -> tuple[str, ...]:
        """Upsert vectors and their chunk metadata, returning deterministic point IDs."""
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same number of items")
        if len(vectors) == 0:
            return ()

        points: list[_Point] = []
        for vector, payload in zip(vectors, payloads, strict=True):
            valid_vector = self._validate_vector(vector)
            valid_payload = self._validate_payload(payload)
            points.append(
                _Point(
                    point_id=self.point_id_for_chunk(str(valid_payload["chunk_id"])),
                    vector=valid_vector,
                    payload=valid_payload,
                )
            )
        self._backend.upsert(self.collection_name, points)
        return tuple(point.point_id for point in points)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> tuple[SearchResult, ...]:
        """Return the closest points with their similarity score and payload."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        vector = self._validate_vector(query_vector)
        return tuple(self._backend.search(self.collection_name, vector, limit))

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "QdrantVectorStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        try:
            array = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("vector must contain numeric values") from error
        if array.shape != (self.vector_dimension,):
            raise ValueError(
                f"Vector dimension {array.shape} does not match configured dimension "
                f"{self.vector_dimension}"
            )
        if not np.isfinite(array).all():
            raise ValueError("vector must contain only finite values")
        return array.tolist()

    @classmethod
    def _validate_payload(cls, payload: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        missing = cls._PAYLOAD_FIELDS - payload.keys()
        if missing:
            raise ValueError(f"payload is missing fields: {', '.join(sorted(missing))}")

        normalized = dict(payload)
        for field_name in (
            "chunk_id",
            "source_id",
            "sha256",
            "scheme",
            "jurisdiction",
            "language",
            "title",
            "text",
        ):
            value = normalized[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"payload field {field_name} must be a non-empty string")
        for field_name in ("page_start", "page_end"):
            value = normalized[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"payload field {field_name} must be a positive integer")
        if normalized["page_end"] < normalized["page_start"]:
            raise ValueError("payload page_end must not be before page_start")
        if normalized["heading"] is not None and not isinstance(normalized["heading"], str):
            raise ValueError("payload field heading must be a string or null")
        return normalized