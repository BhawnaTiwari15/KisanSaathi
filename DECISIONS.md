# Architecture Decisions

## 2026-10-05: Start as a Python 3.12 modular package

**Decision:** Use a `src/` package layout and Python 3.12-compatible project metadata. Keep feature areas in separate modules as they are implemented.

**Reason:** The layout makes imports behave like an installed package and gives each capability a clear ownership boundary without introducing service infrastructure prematurely.

**Trade-off:** Contributors need an editable install for normal development imports. The foundation can still be tested directly with the standard library.

## 2026-10-05: Keep the foundation dependency-light

**Decision:** Use standard-library dataclasses, enums, environment access, and `unittest` for the initial contracts and tests. Declare pytest and Ruff as optional development tools.

**Reason:** The contracts are small, testable without external services, and can be validated in a clean Python 3.12 environment.

**Trade-off:** Dataclasses provide type hints and basic invariant checks, but not automatic parsing/coercion or JSON-schema generation. Revisit a validation library if external input contracts require those features.

## 2026-10-05: Keep secrets out of source and avoid implicit dotenv loading

**Decision:** Read settings from process environment variables, provide `.env.example` with non-secret defaults, and do not add a dotenv dependency or custom parser in the foundation.

**Reason:** This keeps configuration behavior explicit and avoids a runtime dependency before any integration needs secrets.

**Trade-off:** Developers must load `.env` into their process environment themselves if they use a local env file.

## 2026-10-05: Define only cross-cutting conversation contracts

**Decision:** Define user message, assistant response, language, response status, and citation schemas only. Add feature-specific types alongside their owning modules when those milestones begin.

**Reason:** These contracts support the planned bilingual and citation-aware interaction without inventing speculative schemas for unimplemented capabilities.

**Trade-off:** The schemas will evolve when ingestion, eligibility, and tool boundaries are implemented; changes should be documented and tested then.

## 2026-10-05: Keep official document ingestion local and curated

**Decision:** Use a version-controlled source manifest as an allowlist. Developers provide reviewed PDFs locally; ingestion does not crawl or download URLs. Store source files and generated records under git-ignored data directories.

**Reason:** Source authority and reuse rights need human review, and ingestion must be reproducible without a network dependency or a committed document corpus.

**Trade-off:** A maintainer must obtain, review, and stage each source document manually.

## 2026-10-05: Use PyMuPDF for the first text-PDF parser

**Decision:** Pin PyMuPDF as the runtime PDF dependency. Extract page text and block font/style details; preserve one-based page numbers, keep empty-page records, and flag documents with no extracted text. Do not add OCR.

**Reason:** PyMuPDF exposes page-level text and layout signals needed for source references and heading-aware chunks with a focused dependency.

**Trade-off:** Font/style-based heading detection is heuristic, and scanned PDFs require later manual processing or a separately approved OCR capability.

## 2026-10-05: Identify immutable document versions by SHA-256

**Decision:** `source_id` identifies a logical document; SHA-256 identifies exact file bytes. Keep each raw version under a checksum-addressed path and refuse to overwrite mismatching existing raw or processed artifacts.

**Reason:** Re-ingestion is idempotent, changed files remain auditable, and checksum comparisons detect exact duplicates without confusing them with semantic similarity.

**Trade-off:** Byte-level hashes do not detect re-encoded or near-identical documents. Such deduplication is deferred.

## 2026-10-05: Preserve pages and make chunk output deterministic

**Decision:** Convert PDF page indices to one-based numbers at extraction, retain a record for every page, and attach one-based start/end pages to each chunk. Use font/style heading heuristics with paragraph fallback and stable chunk IDs/JSONL serialization.

**Reason:** Later citation and evaluation layers need traceable page references, while stable output makes ingestion changes testable.

**Trade-off:** Layout heuristics cannot reliably infer semantic headings from every PDF. The pipeline records its version and chunk size so later comparisons can be explicit.

## 2026-10-05: Add lazy dense embeddings with BGE-M3

**Decision:** Use the installed `sentence-transformers` 6.1.0 release with `BAAI/bge-m3` as the configurable default model. Load it only when an embedding is requested, use the model's query/document encoding methods, and normalize vectors for cosine or dot-product retrieval.

**Reason:** Dense embeddings establish the retrieval representation while keeping model loading and any model-weight download out of module imports and unit-test collection.

**Trade-off:** The first real embedding request may require model weights to be available locally or downloaded. Automatic indexing and sparse retrieval remain separate milestones.

## 2026-10-05: Add a local Qdrant vector store

**Decision:** Use `qdrant-client` 1.19.1 in local persistent mode, with configurable storage path, collection name, and embedding dimension. Create cosine-distance collections on demand, derive point UUIDs deterministically from `chunk_id`, and preserve chunk/source metadata as payload.

**Reason:** A local client gives development and tests the Qdrant API without requiring a running server, while stable IDs make repeated chunk upserts idempotent.

**Trade-off:** This layer only stores and searches dense vectors; PDF ingestion does not automatically index data. Hybrid search and reranking remain deferred.

## 2026-10-05: Index processed chunks explicitly

**Decision:** Add a corpus indexer that reads one processed JSONL file at a time, ignores document/page records, joins document metadata onto chunk payloads, batches text through the existing embedding service, and upserts through the existing vector store.

**Reason:** This connects the existing offline ingestion artifacts to dense retrieval without changing the PDF pipeline or automatically indexing files.

**Trade-off:** Indexing is an explicit caller action; there is no CLI, watcher, automatic re-embedding, hybrid search, or reranking in this milestone.

## 2026-10-05: Add a local SQLite FTS5 BM25 index

**Decision:** Keep lexical retrieval in a separate persistent SQLite FTS5 index. Reuse the processed-chunk JSONL parser, key rows by `chunk_id`, index title/heading/text, preserve the full retrieval payload, and use FTS5's built-in `bm25()` ranking through a separate `BM25Retriever`.

**Reason:** FTS5 is available in the current Python SQLite build, needs no additional package or model download, and leaves the existing dense Qdrant collection and `DenseRetriever` unchanged.

**Trade-off:** FTS5 is a single-node lexical index and does not stem terms by language. Hybrid fusion remains a later layer; a larger distributed deployment may move sparse BM25 vectors into Qdrant.

## 2026-10-05: Add lazy multilingual cross-encoder reranking

**Decision:** Use `sentence_transformers.CrossEncoder` with `BAAI/bge-reranker-v2-m3`, loading it lazily and allowing model injection for tests. Rerank at most 10 supplied candidates and return 5 by default, preserving each candidate payload and replacing only its score with the cross-encoder relevance score.

**Reason:** Dense/BM25/RRF stages efficiently produce candidates, while a cross-encoder jointly evaluates each query-document pair for finer relevance ordering. BGE Reranker v2 M3 provides multilingual coverage without introducing another dependency; keeping it second-stage bounds local inference cost.

**Trade-off:** First use may download a roughly 568M-parameter model and is slower than first-stage retrieval, especially on CPU. Scores are ranking signals, not probabilities, and no evaluation or score fusion with RRF is included yet.

## 2026-10-05: Add a hand-labeled retrieval evaluation baseline

**Decision:** Store a versioned English query/relevant-chunk set under `data/evaluation/` and evaluate Dense, BM25, Hybrid, and Hybrid+Reranker at top five. Report per-query binary Hit Rate@5 and macro-averaged true Recall@5; keep model execution behind a runnable CLI and test the evaluator with fakes.

**Reason:** Human labels ground evaluation in the source text and avoid circularly deriving relevance from retriever outputs. Reporting both metrics distinguishes the project's "at least one relevant result" hit criterion from standard multi-relevant-document recall.

**Trade-off:** Relevant IDs are specific to the processed corpus version. Any re-ingestion or chunking change that changes IDs requires human review of the associated labels; the English baseline does not establish multilingual quality.