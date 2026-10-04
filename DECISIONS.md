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