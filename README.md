# KisanSaathi

KisanSaathi is a multilingual, multimodal AI assistant for farmers. The current implementation includes the Python foundation, an offline ingestion pipeline for manually curated official PDF sources, dense, BM25, and hybrid retrieval, a lazy second-stage reranker, a local Qdrant vector-store abstraction, explicit JSONL indexing, a deterministic LangGraph orchestration with citations, eligibility, weather, vision, voice, and generation boundaries, and a mobile-friendly Streamlit UI. Automatic PDF downloading, OCR, and production LLM/speech providers are not implemented.

## Requirements

- Python 3.12
- A computer that can run the offline models locally (development), or a small virtual private server (deployment, see [Deployment](#deployment))

## Set up

From PowerShell, create the virtual environment and install the project with its development tools:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

From Command Prompt, activate with `.venv\Scripts\activate.bat` instead. Install the optional UI extra (`streamlit`) with `python -m pip install -e ".[ui]"`.

Runtime dependencies: PyMuPDF for extracting text and layout information from local PDFs, Sentence Transformers for embeddings and cross-encoder reranking, Qdrant for local vector storage, and `httpx` (direct project dependency; used by the vision provider).

## Configuration

Copy `.env.example` to `.env` when configuring local settings, or export the `KISANSAATHI_*` variables in your shell. Do not put secrets in `.env.example` or commit `.env`. Settings are read from process environment variables; this project does not parse `.env` files automatically.

| Variable | Default | Description |
|----------|---------|-------------|
| `KISANSAATHI_ENV` | `development` | Application environment: `development`, `test`, `production` |
| `KISANSAATHI_DEFAULT_LANGUAGE` | `en` | Default UI language: `en` or `hi` |
| `KISANSAATHI_EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Sentence-transformer embedding model |
| `KISANSAATHI_RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker model |
| `KISANSAATHI_EMBEDDING_DIMENSION` | `1024` | Vector dimension for the Qdrant collection |
| `KISANSAATHI_QDRANT_STORAGE_PATH` | `data/qdrant` | Local Qdrant persistent storage directory |
| `KISANSAATHI_QDRANT_COLLECTION_NAME` | `kisansathi_chunks` | Qdrant collection name |
| `KISANSAATHI_BM25_STORAGE_PATH` | `data/bm25.sqlite3` | SQLite FTS5 BM25 database |
| `KISANSAATHI_SOURCES_MANIFEST` | `data/sources.json` | Source manifest used for citations; override when the deployed layout differs from a repository checkout |
| `KISANSAATHI_VISION_PROVIDER` | `fake` | Vision provider: `fake` or `gemini` |
| `KISANSAATHI_VISION_API_KEY` | (required for gemini) | Gemini API key from Google AI Studio |
| `KISANSAATHI_VISION_MODEL_NAME` | `gemini-1.5-flash-latest` | Gemini model name |
| `KISANSAATHI_VISION_TIMEOUT_SECONDS` | `15.0` | Total request timeout |
| `KISANSAATHI_VISION_CONNECT_TIMEOUT_SECONDS` | `5.0` | Connection timeout |
| `KISANSAATHI_VISION_MAX_RETRIES` | `2` | Max retry attempts for transient failures |
| `KISANSAATHI_VISION_RETRY_BACKOFF_BASE` | `1.0` | Exponential backoff base (seconds) |
| `KISANSAATHI_VISION_MAX_IMAGE_BYTES` | `10485760` | Max image size (10 MB) |
| `KISANSAATHI_VISION_TEMPERATURE` | `0.0` | Generation temperature (0 = deterministic) |
| `KISANSAATHI_VISION_MAX_OUTPUT_TOKENS` | `1024` | Max output tokens |
| `KISANSAATHI_LOG_LEVEL` | `INFO` | Logging level for entry points |

**Never commit API keys.** Use a secret manager or a gitignored `.env` file.

Both models load lazily when first used; their weights may need to be downloaded then. BM25 uses the Python standard library's SQLite FTS5 support; Qdrant uses local persistent storage. Neither store indexes PDFs automatically; build the indexes explicitly (see [Indexes](#indexes)).

## Official source ingestion

Only add documents after manually verifying the issuing authority and official source URL. Record one source per logical document in `data/sources.json`; an HTTPS URL or government-looking hostname alone does not establish authority. Unknown publication or effective dates should remain null.

Place each reviewed PDF under `data/incoming/` at the relative path in its manifest entry, then run:

```powershell
python -m kisansathi.ingestion
```

The command reads only the local manifest and files; it does not download documents. Originals are retained under `data/raw/` using their SHA-256 checksum, and extracted page/chunk records are written under `data/processed/`. The processed corpus is small (~77 KB) and **is committed to the repository**; the raw PDFs and your `data/incoming/` files are gitignored. Re-running unchanged inputs preserves existing versions and produces deterministic processed output.

The parser handles text-based PDFs only. Empty or scanned documents are reported for manual review; OCR is not included. Heading detection uses PDF font/style signals and falls back to paragraph-based chunks when no headings are detected.

## Indexes

Both retrieval indexes are derived artifacts: they rebuild from the committed processed corpus and are never committed themselves.

```powershell
python -m kisansathi.build_indexes                # dense (Qdrant) + BM25
python -m kisansathi.build_indexes --bm25-only    # lexical index only
python -m kisansathi.build_indexes --dense-only   # embedding index only
```

Run from the project root after ingestion (and after any corpus change). Dense indexing loads the embedding model on first run. Per-file errors are reported on stderr without aborting the run; the command exits non-zero if any file failed.

## Streamlit UI

The project ships a mobile-friendly web UI as an optional extra:

```powershell
python -m pip install -e ".[ui]"
python -m streamlit run src/kisansathi/ui/streamlit_app.py
```

The UI is a thin presentation layer over the existing application boundaries:

- `kisansathi.ui.streamlit_app` renders inputs and responses; it never calls Qdrant, Gemini, Whisper, or eligibility rules directly.
- `kisansathi.ui.helpers` holds pure, Streamlit-free presentation helpers (request-state building, citation formatting, status labels/footers, location and eligibility-fact parsing), covered by `tests/test_ui_helpers.py`.
- `kisansathi.ui.composition_root.build_application_service` assembles the real application: a dense retriever over the local Qdrant collection, the Open-Meteo weather client, a citation resolver over the tracked `data/sources.json` manifest (overridable with `KISANSAATHI_SOURCES_MANIFEST`), the deterministic PM-KISAN eligibility evaluator, the configured vision provider, and the deterministic language detector. Every dependency is overridable at the call site, and the service is cached once per process with `st.cache_resource`.

Inputs: question text, language (English/Hindi/Kannada/Telugu), an optional crop image, optional audio, explicit latitude/longitude for weather, and optional PM-KISAN facts as tri-state selections ("Not sure" / "Yes" / "No"). The UI never infers a location, never parses facts out of free text, and never applies eligibility rules itself; facts the farmer does not know reach the eligibility engine as missing. Responses render by status — answered, needs clarification, abstained — with sources in an expander and status-specific footers; unexpected graph failures surface one fixed safe message while details are logged server-side only.

## Deployment

The deployment target is a **single-process virtual private server**: a Linux venv running the Streamlit app under systemd. No containerization is used in this milestone.

Why not Streamlit Community Cloud or a free HF Space? The bundled CPU-only torch and the fp32 `BAAI/bge-m3` weights (2.3 GB) comfortably exceed Community Cloud's ~2.7 GB memory budget. A ~2 vCPU / 8 GB RAM / 40 GB disk VPS (~$5–10/month) gives the app headroom (estimated peak 3.5–4.5 GB) plus room for the ~6.4 GB of model weights already cached locally.

### Recommended steps

1. Provision a VM (2 vCPU, 8 GB RAM, 40 GB disk is comfortable).
2. Clone the repository and create the venv:
   ```bash
   python3.12 -m venv .venv
   ./.venv/bin/activate
   # Install CPU-only torch first so the default GPU wheel is not pulled.
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -e ".[ui]"
   ```
3. Create `.env` with the `KISANSAATHI_*` values you need (see [Configuration](#configuration)) and restrict permissions:
   ```bash
   chmod 600 .env
   ```
4. Build the retrieval indexes (downloads model weights on first run):
   ```bash
   python -m kisansathi.build_indexes
   ```
5. Run the app under systemd. Example unit `/etc/systemd/system/kisansathi.service`:
   ```ini
   [Unit]
   Description=KisanSaathi Streamlit app
   After=network-online.target

   [Service]
   User=kisansathi
   WorkingDirectory=/opt/kisansathi
   EnvironmentFile=/opt/kisansathi/.env
   ExecStart=/opt/kisansathi/.venv/bin/python -m streamlit run \
       src/kisansathi/ui/streamlit_app.py
   Restart=always
   RestartSec=5
   Environment=PYTHONUNBUFFERED=1

   [Install]
   WantedBy=multi-user.target
   ```
   Then `systemctl enable --now kisansathi`. Health is available at `/_stcore/health` (returns `"ok"`); per-run failures are logged server-side only.
6. To update: `git pull`, re-run the index build if the corpus changed, then `systemctl restart kisansathi`.
7. Optional pinning: generate a full lockfile with `pip freeze > requirements.lock` on the server. It is documented here rather than committed because the project's direct pins in `pyproject.toml` are the source of truth.

### Deployment notes

- Single-user development focus: no authentication, reverse proxy, or TLS is configured in this milestone.
- Use an absolute `KISANSAATHI_SOURCES_MANIFEST` in the unit's `.env` if the checkout layout differs from the repo default.
- Logging is governed by `KISANSAATHI_LOG_LEVEL` and goes to stderr, which systemd collects via `journalctl -u kisansathi`.

## Demo workflow

1. Start the app (see [Streamlit UI](#streamlit-ui)) and open `http://localhost:8501`.
2. Ask an eligibility question such as *"Which farmer families are eligible for PM-KISAN benefits?"* and verify the answer cites the tracked `data/sources.json` sources in the expander.
3. Upload a crop image and confirm the response reports the vision observation (with the fake provider, a deterministic placeholder) instead of a diagnosis.
4. Ask a weather question with explicit latitude/longitude and confirm the weather route returns without document citations.
5. Prove restart safety: kill the process, restart it, and confirm repeat questions answer identically (the corpus and indexes are unchanged).
6. On the VPS: `curl http://<host>:8501/_stcore/health` returns `ok` and `systemctl status kisansathi` shows the service active.

## Run checks

```powershell
python -m pytest -q
python -m compileall -q src tests
```

The full automated suite passes (over 750 tests) and makes zero network calls. `ruff check .` is available after installing the development extra.

## Retrieval evaluation

The hand-labeled English baseline is `data/evaluation/retrieval_en_v1.json`, tied to the exact processed corpus checksums in its `corpus_version`. Run it from the project root with `python -m kisansathi.evaluation` after explicitly populating the dense and BM25 stores. The evaluator compares DenseRetriever, BM25Retriever, HybridRetriever, and HybridRetriever followed by Reranker, reporting per-query results and aggregate metrics.

The project's primary `recall_at_5` is Hit Rate@5: a query scores 1 if at least one relevant chunk is in the top five, otherwise 0. True Recall@5 is also reported as the per-query fraction of relevant chunks found in the top five, macro-averaged across queries. Labels are hand-reviewed against the source documents to avoid circularly judging systems by their own results. Since labels point to chunk IDs, re-ingestion or changed chunking requires reviewing and updating the affected labels.

Measured on the current 10-query English benchmark (dense + BM25 stores built from this corpus): Dense Hit@5 0.80, BM25 Hit@5 0.90, Hybrid Hit@5 0.90, Hybrid+Reranker Hit@5 0.90, Recall@5 0.90. This is a small benchmark, not a statistically significant result.

## Multilingual retrieval evaluation

A `kisansathi.evaluation.multilingual` module provides **cross-language retrieval evaluation** with per-language Hit@K and Recall@K metrics. This is **separate from answer-quality evaluation** — it measures "Did we retrieve the right evidence?" not "Did the answer use evidence correctly?"

### Current Language Coverage

| Language | Source Documents | Genuine Labels | Status |
|----------|-----------------|----------------|--------|
| English | PM-KISAN FAQ, MH PDMC | 10 queries | **Available** |
| Hindi | None | None | **Unavailable** |
| Kannada | None | None | **Unavailable** |
| Telugu | None | None | **Unavailable** |
| Marathi | MH PDMC (bilingual) | None | **Excluded** — not a target evaluation language |

**No multilingual retrieval metric is reported until genuine relevance labels exist for that language.**

### Metrics

- **Hit@5**: Fraction of queries with ≥1 relevant chunk in top 5
- **Recall@5**: Macro-averaged fraction of relevant chunks found in top 5 (primary metric)
- **Retrieval Gap**: English Recall@5 − Target Language Recall@5 (only when both have genuine labels)

### Architecture

- **Per-language versioned benchmarks**: `data/evaluation/retrieval/retrieval_{lang}_v{n}.json`
- **Unavailable language handling**: Explicit `unavailable=true` with reason; never reported as 0.0
- **Reproducibility metadata**: Benchmark version, corpus checksum, retrieval config, K, timestamp, evaluator version
- **Native-speaker annotation workflow**: Queries authored by native speakers; relevance judged against source documents
- **No mechanical translation**: Translated English queries are NOT valid benchmarks

### CLI

```powershell
python -m kisansathi.evaluation.main multilingual ^
  --base-path data/evaluation/retrieval ^
  --output-jsonl output/multilingual.jsonl ^
  --output-markdown output/multilingual.md ^
  --language en  # optional: filter to specific language (en/hi/kn/te/all)
```

### Native Annotation Requirements

Before reporting a Hindi/Kannada/Telugu retrieval score:

1. Add source documents in that language to `data/sources.json` and `data/incoming/`
2. Ingest documents (`python -m kisansathi.ingestion`)
3. Native speaker formulates real farmer queries in that language
4. Native speaker identifies relevant chunks from the corpus for each query
5. Second native speaker validates (target Cohen's kappa ≥ 0.8)
6. Write benchmark JSON with all metadata
7. Run evaluation via CLI

## Answer generation

A `kisansathi.generation` package provides an injectable `AnswerGenerator` protocol with a `DefaultAnswerGenerator` implementation. The LangGraph orchestration adds a `generate_answer` node (after `resolve_citations`) and a `validate_and_attach_citations` node on the retrieval and eligibility paths. `build_graph` accepts a keyword-only `answer_generator` parameter; when absent, routes behave exactly as before with deterministic placeholder responses. The generator receives a `GenerationContext` (message, resolved `CitationBatch`, eligibility decision, weather, vision_result) and returns a structured `GeneratedAnswer`. The prompt explicitly forbids external knowledge, requires inline citation IDs from an allowlist, and demands the requested language. Model output is parsed as `ANSWER:` / `CITATIONS:` sections; cited IDs are validated against the resolved batch, then re-validated post-generation via `validate_referenced_citations`. LLM failures yield `ABSTAINED`; malformed output or hallucinated citation IDs yield `NEEDS_CLARIFICATION`. Weather and clarification routes bypass generation entirely. No LLM provider is hard-coded; tests use a `FakeLLMClient` and `FakeAnswerGenerator`.

## Answer quality evaluation

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

Uses `FakeLLMJudge` by default (no API calls). Real judges are implemented programmatically via the `LLMJudge` protocol.

### Limitations

- Judge bias/variance affects scores
- No confidence intervals or statistical significance
- Results not comparable across different judge providers/models/configurations
- Context precision/recall require reference contexts (unavailable when missing)

## Vision analysis

A `kisansathi.vision` package provides an injectable `VisionAnalyzer` protocol. The domain layer depends only on this protocol; concrete providers live in `kisansathi.vision.providers`.

**Current provider**: Google Gemini (multimodal LLM) via `GeminiVisionAnalyzer`. The API key travels in the `x-goog-api-key` header, never in the URL. All normal tests use `FakeVisionAnalyzer` — no network calls, no API keys required.

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
- Normal CI: `FakeVisionAnalyzer` only, zero network calls
- Contract tests (manual): `tests/contract/test_gemini_vision_contract.py` skipped unless `RUN_REAL_VISION_TESTS=1` and `GEMINI_API_KEY` set

## Current scope

`HybridRetriever` combines dense and BM25 rankings with Reciprocal Rank Fusion. `Reranker` can then rescore a bounded candidate set with query-document CrossEncoder relevance scores; this second-stage operation is separate from dense, BM25, and RRF scoring. A deterministic LangGraph orchestration skeleton (`kisansathi.orchestration`) routes messages to clarification, retrieval, eligibility, or weather and returns a final `AssistantResponse`; it injects a retriever, a weather client, a citation resolver, and an eligibility evaluator as dependencies and intentionally omits LLM, vision, Whisper, translation, TTS, tools, checkpointer, and external services. Both evidence routes converge on one `resolve_citations` node, so the graph selects evidence and attaches whatever the resolver returns while the resolver alone decides what is citable; the graph never constructs a `Citation` itself and never reads chunk text. The weather route is deliberately left independent from the citation layer and does not reach the resolver, because weather carries no document evidence. A weather tool (`kisansathi.weather`) provides typed Open-Meteo integration (forecast API, no API key) with transport isolation and network-free tests; it is independent from LangGraph for this milestone. A deterministic eligibility evaluator (`kisansathi.eligibility`) answers PM-KISAN questions from an explicit, versioned rule set whose every condition cites a verbatim excerpt of an ingested corpus chunk; it is a pure function and is injected into the graph rather than imported by it. Its facts are read from a caller-supplied `eligibility_request` and are never parsed out of message text, because inferring a farmer's landholding or tax status by keyword would silently produce a wrong verdict about a real benefit; an absent or malformed request yields insufficient information or a clarification instead of a guess. Conditions the corpus does not support (including the commonly assumed two-hectare land cap) are not encoded in the rules.

## Known limitations

- **No production answer generator or speech-to-text is configured**, so retrieval answers are deterministic placeholders and uploaded audio reaches the speech boundary without transcription. The UI help text states this rather than implying Whisper runs.
- **The vision provider defaults to `fake`.** Real image observations require the Gemini provider and a key; the contract test is opt-in and disabled in CI.
- **Multilingual retrieval is measured for English only** (10 queries). Hindi, Kannada, and Telugu show no score until real labels exist.
- **No OCR.** Empty or scanned PDFs are flagged for manual review, not transcribed.
- **No automatic PDF downloads.** Documents are added by hand after verifying the issuing authority.
- **First use downloads model weights** (BGE-M3 ~2.3 GB + reranker ~2.1 GB), so the app needs network access on first run.
- **One server, one process.** There is no load balancing, no auth, and no TLS in this milestone.

## Data, evaluation, and portfolio claims

- The processed corpus (`data/processed/`, ~77 KB) is committed; raw PDFs, `data/incoming/`, model weights, and generated indexes are not. Model caches (if ever pointed inside the repo) live under `hf_cache/`, which is ignored.
- All retrieval and answer-quality metrics are tied to versioned benchmark files whose `corpus_version` pins the processed corpus checksums; labels are hand-reviewed against the source documents.
- No multilingual metric is invented. Where labels do not exist, the benchmarks record the language as explicitly unavailable rather than reporting 0.0 or translated-query scores.
- Every claim that would appear in a portfolio is reproducible from this repository: the committed corpus, the pinned dependencies in `pyproject.toml`, the index-build CLI, and the deterministic test suite.

## Interview talking points

- **Why the architecture is trustworthy**: evidence code paths converge on a single resolve-citations node, and the resolver alone decides what is citable; the orchestration graph never constructs a citation itself.
- **Why retrieval is honest**: hybrid dense/BM25 retrieval with RRF plus a lazy reranker is measured on hand-labeled English queries (Hit@5 0.90 on the current small benchmark), with the number and vocabulary of labels openly stated.
- **Why the UI does not pretend**: no location inference, no free-text fact parsing, no fake speech transcription, no fake vision — capabilities and boundaries are stated in the UI itself.
- **Why deployment is a single VPS process**: the model weight and library footprint (bundled CPU torch ~1.4 GB, fp32 embeddings, ~3.5–4.5 GB peak working set) is too large for free single-app tiers, and a systemd-managed venv is restart-safe, observable, and cheap.
- **Why secrets are handled defensively**: API keys travel in headers, not URLs; `.env` is gitignored; nothing secret is ever logged.