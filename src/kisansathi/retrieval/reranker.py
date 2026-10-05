"""Second-stage query-document reranking with a CrossEncoder."""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from kisansathi.config import Settings
from kisansathi.retrieval.vector_store import SearchResult


class _CrossEncoderModel(Protocol):
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        convert_to_numpy: bool,
    ) -> object: ...


ModelLoader = Callable[[str], _CrossEncoderModel]


def _load_cross_encoder(model_name: str) -> _CrossEncoderModel:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class Reranker:
    """Rescore retrieved candidates jointly with a query using a CrossEncoder."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        candidate_depth: int = 10,
        default_top_k: int = 5,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._validate_positive_int(candidate_depth, "candidate_depth")
        self._validate_positive_int(default_top_k, "top_k")
        self._settings = settings if settings is not None else Settings.from_env()
        self._candidate_depth = candidate_depth
        self._default_top_k = default_top_k
        self._model_loader = model_loader
        self._model: _CrossEncoderModel | None = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int | None = None,
    ) -> tuple[SearchResult, ...]:
        """Rescore candidates and return up to ``top_k`` by relevance."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be blank")
        result_limit = self._default_top_k if top_k is None else top_k
        self._validate_positive_int(result_limit, "top_k")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError("candidates must be a sequence of SearchResult objects")
        if len(candidates) > self._candidate_depth:
            raise ValueError(
                f"candidate count must not exceed candidate_depth ({self._candidate_depth})"
            )
        if not candidates:
            return ()

        pairs: list[tuple[str, str]] = []
        for candidate in candidates:
            if not isinstance(candidate, SearchResult):
                raise TypeError("candidates must contain only SearchResult objects")
            text = candidate.payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("candidate payload text must be a non-empty string")
            pairs.append((query, text))

        try:
            scores = np.asarray(
                self._get_model().predict(pairs, convert_to_numpy=True), dtype=np.float64
            ).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError("CrossEncoder returned invalid relevance scores") from error
        if scores.size != len(candidates):
            raise ValueError(
                f"CrossEncoder returned {scores.size} scores for {len(candidates)} candidates"
            )
        if not np.isfinite(scores).all():
            raise ValueError("CrossEncoder returned non-finite relevance scores")

        ranked = sorted(
            enumerate(zip(candidates, scores, strict=True)),
            key=lambda item: (-float(item[1][1]), item[0]),
        )
        return tuple(
            SearchResult(score=float(score), payload=candidate.payload)
            for _, (candidate, score) in ranked[:result_limit]
        )

    def _get_model(self) -> _CrossEncoderModel:
        if self._model is None:
            loader = self._model_loader or _load_cross_encoder
            self._model = loader(self._settings.reranker_model_name)
        return self._model

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")