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

### Vision Provider Configuration (new)

The vision pipeline uses an injectable `VisionAnalyzer` protocol. The current concrete implementation is **Google Gemini** (via `kisansathi.vision.providers.GeminiVisionAnalyzer`). A `FakeVisionAnalyzer` is used for all normal tests.

Configure the vision provider through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `KISANSAATHI_VISION_PROVIDER` | `fake` | Provider to use: `fake` or `gemini` |
| `KISANSAATHI_VISION_API_KEY` | (required for gemini) | Gemini API key from Google AI Studio |
| `KISANSAATHI_VISION_MODEL_NAME` | `gemini-1.5-flash-latest` | Gemini model name |
| `KISANSAATHI_VISION_TIMEOUT_SECONDS` | `15.0` | Total request timeout |
| `KISANSAATHI_VISION_CONNECT_TIMEOUT_SECONDS` | `5.0` | Connection timeout |
| `KISANSAATHI_VISION_MAX_RETRIES` | `2` | Max retry attempts for transient failures |
| `KISANSAATHI_VISION_RETRY_BACKOFF_BASE` | `1.0` | Exponential backoff base (seconds) |
| `KISANSAATHI_VISION_MAX_IMAGE_BYTES` | `10485760` | Max image size (10 MB) |
| `KISANSAATHI_VISION_TEMPERATURE` | `0.0` | Generation temperature (0 = deterministic) |
| `KISANSAATHI_VISION_MAX_OUTPUT_TOKENS` | `1024` | Max output tokens |

**Never commit API keys.** Use a secret manager or `.env` file (gitignored).

To use the real Gemini provider:

```powershell
$env:KISANSAATHI_VISION_PROVIDER="gemini"
$env:KISANSAATHI_VISION_API_KEY="your-gemini-api-key"
python -m pytest tests/ -q  # Uses fake provider (no API key in CI)
```

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

`HybridRetriever` combines dense and BM25 rankings with Reciprocal Rank Fusion. `Reranker` can then rescore a bounded candidate set with query-document CrossEncoder relevance scores; this second-stage operation is separate from dense, BM25, and RRF scoring. A deterministic LangGraph orchestration skeleton (`kisansathi.orchestration`) routes messages to clarification, retrieval, eligibility, or weather and returns a final `AssistantResponse`; it injects a retriever, a weather client, a citation resolver, and an eligibility evaluator as dependencies and intentionally omits LLM, vision, Whisper, translation, TTS, tools, checkpointer, and external services. Both evidence routes converge on one `resolve_citations` node, so the graph selects evidence and attaches whatever the resolver returns while the resolver alone decides what is citable; the graph never constructs a `Citation` itself and never reads chunk text. The weather route is deliberately left independent from the citation layer and does not reach the resolver, because weather carries no document evidence. A weather tool (`kisansathi.weather`) provides typed Open-Meteo integration (forecast API, no API key) with transport isolation and network-free tests; it is independent from LangGraph for this milestone. A deterministic eligibility evaluator (`kisansathi.eligibility`) answers PM-KISAN questions from an explicit, versioned rule set whose every condition cites a verbatim excerpt of an ingested corpus chunk; it is a pure function and is injected into the graph rather than imported by it. Its facts are read from a caller-supplied `eligibility_request` and are never parsed out of message text, because inferring a farmer's landholding or tax status by keyword would silently produce a wrong verdict about a real benefit; an absent or malformed request yields insufficient information or a clarification instead of a guess. Conditions the corpus does not support (including the commonly assumed two-hectare land cap, whic...

## Vision Analysis (new)

A `kisansathi.vision` package provides an injectable `VisionAnalyzer` protocol. The domain layer depends only on this protocol; concrete providers live in `kisansathi.vision.providers`.

**Current provider**: Google Gemini (multimodal LLM) via `GeminiVisionAnalyzer`. All normal tests use `FakeVisionAnalyzer` — no network calls, no API keys required.

**Architecture**:
- `VisionAnalyzer.analyze(image_bytes, content_type, ...)` → `VisionResult`
- `VisionResult` contains structured `VisualObservation` objects with explicit uncertainty
- Observations are NEVER definitive diagnoses — they use "symptoms consistent with", "possibly", confidence scores
- Vision output is an OBSERVATION, not evidence. It augments retrieval queries but never bypasses document grounding
- Provider failures map to domain exceptions: `AnalyzerUnavailableError`, `AnalyzerTimeoutError`, `ImageValidationError`, etc.

**Provider behavior**:
- Validates image (MIME, magic bytes, size, dimensions) BEFORE provider call
- Sends tightly constrained prompt demanding structured JSON output
- Validates provider response against `VisionResult` schema before returning
- Retries ONLY transient failures (429, 5xx, timeout); never retries auth, 400, or malformed output
- Structured logging: provider, model, latency, status, retry count, confidence distribution
- NEVER logs: API keys, raw image bytes, full request/response payloads

**Privacy**:
- Image bytes held in memory only during request; discarded immediately after
- No disk persistence of images
- Provider retention policy documented (Gemini API: no training on API data)

**Testing**:
- Normal CI: `FakeVisionAnalyzer` only, 600+ tests, zero network calls
- Contract tests (manual): `tests/contract/test_gemini_vision_contract.py` skipped unless `RUN_REAL_VISION_TESTS=1` and `GEMINI_API_KEY` set

## Answer generation (new)

A `kisansathi.generation` package provides an injectable `AnswerGenerator` protocol with a `DefaultAnswerGenerator` implementation. The LangGraph orchestration adds a `generate_answer` node (after `resolve_citations`) and a `validate_and_attach_citations` node on the retrieval and eligibility paths. `build_graph` accepts a keyword-only `answer_generator` parameter; when absent, routes behave exactly as before with deterministic placeholder responses. The generator receives a `GenerationContext` (message, resolved `CitationBatch`, eligibility decision, weather, vision_result) and returns a structured `GeneratedAnswer`. The prompt explicitly forbids external knowledge, requires inline citation IDs from an allowlist, and demands the requested language. Model output is parsed as `ANSWER:` / `CITATIONS:` sections; cited IDs are validated against the resolved batch, then re-validated post-generation via `validate_referenced_citations`. LLM failures yield `ABSTAINED`; malformed output or hallucinated citation IDs yield `NEEDS_CLARIFICATION`. Weather and clarification routes bypass generation entirely. No LLM provider is hard-coded; tests use a `FakeLLMClient` and `FakeAnswerGenerator`.

## Answer quality evaluation (new)

A `kisansathi.evaluation.answer_quality` module provides **semantic answer quality metrics** using an LLM judge (RAGAS-compatible adapter). This is **separate from deterministic checks** in `kisansathi.evaluation.answer` and **opt-in only** — it does not run as part of normal pytest.

### Metrics

- **Faithfulness**: Does the generated answer stay faithful to retrieved contexts?
- **Answer Relevance**: Is the answer relevant to the query?
- **Context Precision**: Are retrieved contexts relevant (vs reference contexts)?
- **Context Recall**: Do retrieved contexts cover reference contexts? (Only when reference contexts exist)

### Architecture

- **Provider-agnostic `LLMJudge` protocol** — inject any LLM provider; `FakeLLMJudge` for deterministic tests
- **Prompt-injection safe** — evaluation prompts explicitly forbid following instructions in retrieved/generated content
- **Missing-input handling** — metrics unavailable when required inputs missing; no fake scores substituted
- **Reproducibility metadata** — records benchmark version, judge provider/model, temperature, timestamp, config
- **No API keys in logs/reports**

### Benchmark

Hand-authored English dataset: `data/evaluation/answer_quality/answer_quality_en_v1.json` (5 cases, tied to corpus checksums). No multilingual answer-quality labels fabricated — Hindi/Kannada/Telugu not reported without real labels.

### CLI (opt-in)

```powershell
python -m kisansathi.evaluation.main answer-quality ^
  --dataset data/evaluation/answer_quality/answer_quality_en_v1.json ^
  --output-jsonl output/answer_quality.jsonl ^
  --output-markdown output/answer_quality.md
```

Uses `FakeLLMJudge` by default (no API calls). Real judges implemented programmatically via `LLMJudge` protocol.

### Reports

- **Machine-readable JSONL** — one line per metric with judge metadata
- **Human-readable Markdown** — summary table, per-case breakdown, limitations section

### Limitations

- Judge bias/variance affects scores
- No confidence intervals or statistical significance
- Results not comparable across different judge providers/models/configurations
- Context precision/recall require reference contexts (unavailable when missing)