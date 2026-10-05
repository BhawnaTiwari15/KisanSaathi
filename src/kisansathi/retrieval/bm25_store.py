"""Persistent local BM25 indexing and search using SQLite FTS5."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3

from kisansathi.config import Settings
from kisansathi.retrieval.processed_chunks import read_chunk_payloads, validate_retrieval_payload
from kisansathi.retrieval.vector_store import SearchResult


@dataclass(frozen=True, slots=True)
class BM25IndexingSummary:
    chunks_read: int = 0
    indexed: int = 0
    skipped_records: int = 0
    errors: tuple[str, ...] = ()


class BM25Store:
    """Persistent FTS5 store that returns the shared retrieval result type."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        database_path: str | Path | None = None,
    ) -> None:
        self._settings = settings if settings is not None else Settings.from_env()
        configured_path = (
            self._settings.bm25_storage_path if database_path is None else database_path
        )
        self._database_path = str(configured_path)
        if not self._database_path.strip():
            raise ValueError("BM25 database path must not be empty")
        if self._database_path != ":memory:":
            Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._database_path)
        self._connection.row_factory = sqlite3.Row
        self._check_fts5()
        self._connection.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS bm25_chunks USING fts5(
                chunk_id UNINDEXED,
                payload_json UNINDEXED,
                title,
                heading,
                text,
                tokenize='unicode61'
            )"""
        )
        self._connection.commit()

    def _check_fts5(self) -> None:
        try:
            self._connection.execute(
                "CREATE VIRTUAL TABLE temp.bm25_fts5_probe USING fts5(content)"
            )
            self._connection.execute("DROP TABLE temp.bm25_fts5_probe")
        except sqlite3.Error as error:
            self._connection.close()
            raise RuntimeError(
                "SQLite FTS5 is required for BM25Store but is unavailable in this Python build"
            ) from error

    def index_file(self, jsonl_path: str | Path) -> BM25IndexingSummary:
        """Index chunk records from one processed JSONL file without embedding them."""
        parsed = read_chunk_payloads(jsonl_path)
        errors = list(parsed.errors)
        indexed = 0
        if parsed.payloads:
            try:
                indexed = self.upsert(parsed.payloads)
            except (TypeError, ValueError, sqlite3.Error) as error:
                errors.append(f"BM25 upsert failed: {error}")
        return BM25IndexingSummary(
            chunks_read=parsed.chunks_read,
            indexed=indexed,
            skipped_records=parsed.skipped_records,
            errors=tuple(errors),
        )

    def upsert(self, payloads: Sequence[Mapping[str, object]]) -> int:
        """Insert or replace complete retrieval payloads by chunk ID transactionally."""
        rows: list[tuple[str, str, str, str, str]] = []
        for payload in payloads:
            normalized = validate_retrieval_payload(payload)
            rows.append(
                (
                    str(normalized["chunk_id"]),
                    json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                    str(normalized["title"]),
                    normalized["heading"] or "",
                    str(normalized["text"]),
                )
            )

        with self._connection:
            for chunk_id, payload_json, title, heading, text in rows:
                self._connection.execute(
                    "DELETE FROM bm25_chunks WHERE chunk_id = ?", (chunk_id,)
                )
                self._connection.execute(
                    """INSERT INTO bm25_chunks
                    (chunk_id, payload_json, title, heading, text)
                    VALUES (?, ?, ?, ?, ?)""",
                    (chunk_id, payload_json, title, heading, text),
                )
        return len(rows)

    def search(self, query: str, *, limit: int = 5) -> tuple[SearchResult, ...]:
        """Search with FTS5 BM25 and return higher-is-better similarity scores."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be blank")
        self._validate_limit(limit)
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return ()
        match_query = " OR ".join(f'"{term}"' for term in terms)
        rows = self._connection.execute(
            """SELECT payload_json,
                      bm25(bm25_chunks, 0.0, 0.0, 3.0, 2.0, 1.0) AS rank
               FROM bm25_chunks
               WHERE bm25_chunks MATCH ?
               ORDER BY rank ASC
               LIMIT ?""",
            (match_query, limit),
        )
        return tuple(
            SearchResult(
                score=-float(row["rank"]),
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "BM25Store":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")