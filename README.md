# KisanSaathi

KisanSaathi is a multilingual, multimodal AI assistant for farmers. The current implementation includes the Python foundation, an offline ingestion pipeline for manually curated official PDF sources, dense, BM25, and hybrid retrieval, a lazy second-stage reranker, a local Qdrant vector-store abstraction, and explicit JSONL indexing. Automatic PDF downloading/indexing, RAG, eligibility, and user interfaces are not implemented.

## Requirements

- Windows
- Python 3.12

## Set up

From PowerShell, create the virtual environment and install the project with its development tools:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

From Command Prompt, activate with `.venv\Scripts\activate.bat` instead. Runtime dependencies include PyMuPDF for extracting text and layout information from local PDFs, Sentence Transformers for embeddings and cross-encoder reranking, and Qdrant for local vector storage. The embedding service defaults to `BAAI/bge-m3`; the reranker defaults to multilingual `BAAI/bge-reranker-v2-m3`. Configure them through `KISANSAATHI_EMBEDDING_MODEL_NAME` and `KISANSAATHI_RERANKER_MODEL_NAME`. Both models load lazily when first used; their weights may need to be downloaded then. BM25 uses the Python standard library's SQLite FTS5 support and persists at `data/bm25.sqlite3` by default; configure the path with `KISANSAATHI_BM25_STORAGE_PATH`. Qdrant uses local persistent storage at `data/qdrant`, collection `kisansathi_chunks`, and vector dimension 1024 by default. Configure these with `KISANSAATHI_QDRANT_STORAGE_PATH`, `KISANSAATHI_QDRANT_COLLECTION_NAME`, and `KISANSAATHI_EMBEDDING_DIMENSION`. Stores do not automatically index PDFs; explicitly index processed JSONL chunks.

Copy `.env.example` to `.env` when configuring local settings. Do not put secrets in `.env.example` or commit `.env`. Settings are read from process environment variables; this foundation does not parse `.env` files automatically.

## Official source ingestion

Only add documents after manually verifying the issuing authority and official source URL. Record one source per logical document in `data/sources.json`; an HTTPS URL or government-looking hostname alone does not establish authority. Unknown publication or effective dates should remain null.

Place each reviewed PDF under `data/incoming/` at the relative path in its manifest entry, then run:

```powershell
python -m kisansathi.ingestion
```

The command reads only the local manifest and files; it does not download documents. Originals are retained under `data/raw/` using their SHA-256 checksum, and extracted page/chunk records are written under `data/processed/`. These three data directories are ignored by git. Re-running unchanged inputs preserves existing versions and produces deterministic processed output.

The parser handles text-based PDFs only. Empty or scanned documents are reported for manual review; OCR is not included. Heading detection uses PDF font/style signals and falls back to paragraph-based chunks when no headings are detected. The source registry starts empty, and no government PDFs are included in this repository.

## Run checks

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

After installing the development extra, `python -m pytest` and `ruff check .` are also available.

## Retrieval evaluation

The hand-labeled English baseline is `data/evaluation/retrieval_en_v1.json`, tied to the exact processed corpus checksums in its `corpus_version`. Run it from the project root with `python -m kisansathi.evaluation` after explicitly populating the dense and BM25 stores. The evaluator compares DenseRetriever, BM25Retriever, HybridRetriever, and HybridRetriever followed by Reranker, reporting per-query results and aggregate metrics.

The project's binary `recall_at_5` is Hit Rate@5: a query scores 1 if at least one relevant chunk is in the top five, otherwise 0. True Recall@5 is also reported as the per-query fraction of relevant chunks found in the top five, macro-averaged across queries. Labels are hand-reviewed against the source documents to avoid circularly judging systems by their own results. Since labels point to chunk IDs, re-ingestion or changed chunking requires reviewing and updating the affected labels.

## Current scope

`HybridRetriever` combines dense and BM25 rankings with Reciprocal Rank Fusion. `Reranker` can then rescore a bounded candidate set with query-document CrossEncoder relevance scores; this second-stage operation is separate from dense, BM25, and RRF scoring. A deterministic LangGraph orchestration skeleton (`kisansathi.orchestration`) routes messages to clarification, retrieval, eligibility, or weather and returns a final `AssistantResponse`; it injects a retriever, a weather client, a citation resolver, and an eligibility evaluator as dependencies and intentionally omits LLM, vision, Whisper, translation, TTS, tools, checkpointer, and external services. Both evidence routes converge on one `resolve_citations` node, so the graph selects evidence and attaches whatever the resolver returns while the resolver alone decides what is citable; the graph never constructs a `Citation` itself and never reads chunk text. The weather route is deliberately left independent from the citation layer and does not reach the resolver, because weather carries no document evidence. A weather tool (`kisansathi.weather`) provides typed Open-Meteo integration (forecast API, no API key) with transport isolation and network-free tests; it is independent from LangGraph for this milestone. A deterministic eligibility evaluator (`kisansathi.eligibility`) answers PM-KISAN questions from an explicit, versioned rule set whose every condition cites a verbatim excerpt of an ingested corpus chunk; it is a pure function and is injected into the graph rather than imported by it. Its facts are read from a caller-supplied `eligibility_request` and are never parsed out of message text, because inferring a farmer's landholding or tax status by keyword would silently produce a wrong verdict about a real benefit; an absent or malformed request yields insufficient information or a clarification instead of a guess. Conditions the corpus does not support (including the commonly assumed two-hectare land cap, which the FAQ contradicts) are recorded as unsupported rather than approximated. A citation and provenance layer (`kisansathi.citations`) resolves retrieval payloads and eligibility evidence into validated `Citation` objects carrying source id, title, page range, registered URL, and verbatim issuing authority. Document metadata is joined from the tracked `data/sources.json` manifest at citation time rather than duplicated into stored chunk payloads, and resolution is fail-closed: unregistered sources, foreign chunks, malformed checksums, invalid pages, and metadata contradictions are refused rather than approximated, with refusals collected in a `CitationBatch`. Two gates guard the not-yet-built answer generator: pre-generation evidence resolution, and post-generation `validate_referenced_citations`, which discards any answer citing an id that was never resolved. The resolver never reads chunk text, so OCR damage is never repaired on the way to a citation. Fail-closed behaviour extends into the graph: an eligibility verdict that cannot cite its own evidence is withheld as `ABSTAINED`, and retrieval whose every chunk was refused abstains rather than returning an answered response with no sources. Building the graph without a `citation_resolver` leaves both routes exactly as they were, and publishes no citations. RAG, external tools beyond retrieval/weather/eligibility/citations, and user interface have not been implemented. `CorpusIndexer` explicitly indexes one processed JSONL file at a time into the dense store; `BM25Store.index_file()` separately builds the local lexical index from the same chunks. Ingestion does not automatically embed or index PDFs.