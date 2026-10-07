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

## 2026-10-05: Add deterministic LangGraph orchestration skeleton

**Decision:** Introduce `kisansathi.orchestration` with `build_graph(retriever)` using `StateGraph`, `START`, `END` from `langgraph.graph`, and a plain `TypedDict` state. The graph injects the retrieval dependency (existing interface), performs explicit conditional routing (`route_request` -> `retrieval` or `clarify`), calls `retriever.retrieve()` only on the retrieval path, and produces `AssistantResponse` objects in `finalize_response`. No checkpointer, no LLM, vision, Whisper, translation, TTS, tools, or external services.

**Reason:** This establishes the first orchestration boundary without modifying retrieval or domain schemas, keeps routing deterministic and testable, and isolates the provisional rule so it can be replaced later.

**Trade-off:** The routing logic is intentionally simple (length/word-count heuristic) and not robust NLP; answer generation remains absent and responses explicitly state that limitation.

## 2026-10-05: Add Open-Meteo weather tool with transport isolation

**Decision:** Add `kisansathi.weather` with typed models (`WeatherRequest`, `WeatherCurrent`, `WeatherForecastPoint`, `WeatherResponse`), explicit exceptions (`WeatherError` and specific subclasses), and `OpenMeteoClient` using a `Transport` Protocol. Use the official Open-Meteo `/v1/forecast` endpoint with no API key, stdlib HTTP (`urllib`) with explicit timeout, and strict validation (latitude [-90,90], longitude [-180,180]). Keep HTTP isolated behind the transport and the tool independent from LangGraph; weather results remain outside graph state until the graph actually needs them.

**Reason:** Isolates I/O for testability (fake transports), matches existing domain/model style (frozen dataclasses + validation), provides explicit error taxonomy (invalid coordinates, timeout, HTTP/API failure, malformed response, unavailable data), and avoids adding external HTTP dependencies.

**Trade-off:** Uses stdlib HTTP (no retries/caching); kept minimal (current fields only by default, optional forecast). No geocoding, LLM, or answer generation included.

## 2026-10-06: Add deterministic PM-KISAN eligibility evaluation

**Decision:** Add kisansathi.eligibility with frozen dataclasses (EligibilityRequest, EvidenceRef, EligibilityRule, RuleSet, EligibilityDecision), a two-member RuleKind vocabulary (must_be_true, must_be_false), and evaluate(request, *, rules=PM_KISAN) as a pure function with no retrieval, I/O, or model calls. Implement four PM-KISAN conditions, each carrying a verbatim excerpt of a real ingested chunk plus an EvidenceRef (source_id, sha256, chunk_id, page range, locator) pinned to the corpus checksum. Rules live as versioned Python literals, not runtime JSON. Request facts are tri-state (ool | None), so an unknown fact is reported rather than defaulted. Precedence is unsupported scheme > ineligible > insufficient information > eligible. Conditions the corpus does not support are recorded in UNSUPPORTED_CONDITIONS with a reason and locator instead of being approximated.

**Reason:** Eligibility must stay deterministic and auditable, so every rule is justified by exact source text rather than general knowledge, and OCR-damaged, date-dependent, or heading-only conditions are documented as unsupported rather than reconstructed. Keeping the evaluator free of LangGraph and retrieval dependencies preserves this milestone's boundary and lets the evaluator be injected into the graph later as a plain dependency.

**Trade-off:** Only four conditions are evaluable, so most real-world PM-KISAN exclusions return insufficient_information rather than a verdict. Excerpt provenance cannot be re-verified in CI because data/processed is gitignored, so substring verification is an opt-in test that skips when the corpus is absent; structural provenance is asserted unconditionally.

## 2026-10-06: Add citation and provenance layer with two validation gates

**Decision:** Add kisansathi.citations with a ManifestSourceRegistry (a source_id to ManifestEntry index built from the tracked data/sources.json), a CitationResolver, and alidate_referenced_citations. Extend domain.Citation with optional page_end, issuing_authority, and chunk_id. Document metadata is joined from the registered source manifest at citation time rather than duplicated into every stored chunk payload, so retrieval contracts stay untouched. Resolution is fail-closed: an unregistered source_id, a chunk_id belonging to another source, a malformed sha256, an invalid or reversed page range, or a title contradicting registered metadata is refused rather than approximated. 
esolve_payloads and 
esolve_evidence_batch collect refusals in a CitationBatch instead of raising, so one bad chunk cannot destroy an answer.

**Reason:** Citations must be traceable to source_id, title, page, URL, and issuing authority without fabricating any of them. The manifest is the only canonical, version-controlled home of document metadata, because /data/processed/ is gitignored. Eligibility EvidenceRef resolves through a structural EvidenceLike Protocol, so retrieval and eligibility converge on one Citation type without either module importing the other. Two gates guard the future LLM stage: pre-generation resolution means the generator can only be shown real evidence, and post-generation alidate_referenced_citations means any citation id it invents causes the whole answer to be discarded rather than published with a fabricated reference.

**Trade-off:** The resolver never reads chunk text, so OCR damage and the chunker heading-leak defect pass through to citations verbatim rather than being masked; a citation can therefore point at a chunk whose heading field is known to be wrong, which is the honest record of what was stored. No page count exists in the manifest or processed metadata, so page validity is structural (one-based, ascending) and cannot detect a page beyond the end of the real document. sha256 format is validated but not cross-checked against a manifest checksum, since the manifest has none. The resolver is not yet wired into the graph, and no answer generator exists.

## 2026-10-06: Wire citations and eligibility into LangGraph

**Decision:** Add an ELIGIBILITY route to the orchestration graph and a single resolve_citations node that both evidence routes converge on. build_graph now accepts keyword-only citation_resolver and eligibility_evaluator dependencies; OrchestrationState gains three NotRequired data fields (citations, eligibility_decision, eligibility_request) and holds no service objects. The citation resolver and the deterministic evaluator are both injected rather than constructed in the graph, so building and compiling still performs no I/O. Retrieval hands result.payload to resolve_payloads and eligibility hands decision.evidence to resolve_evidence_batch; the graph passes provenance only, never chunk text, and never builds a Citation itself. Passing citation_resolver=None leaves both routes exactly as they were, which is why no existing graph test needed to change.

**Reason:** Two routes produce evidence and one resolver decides what is citable, so converging on a single node keeps citation rules in one place instead of duplicating them per route. Eligibility facts are read from a caller-supplied eligibility_request and are never parsed out of message text: inferring a farmer's landholding or tax status by keyword would silently produce a wrong verdict about a real benefit, so an absent or malformed request yields insufficient_information or a clarification instead of a guess. A verdict that cannot cite its own evidence is withheld as ABSTAINED, and retrieval whose every chunk was refused abstains rather than sounding answered. Eligibility summaries come from the versioned rule set, not from generated prose, so stating them directly is deterministic and auditable. Weather and clarification keep their previous behaviour and never reach the resolver, because neither carries document evidence.

**Trade-off:** The eligibility route cannot answer from free text, so a keyword-matched question with no structured facts always asks for clarification instead of producing a verdict; regex-parsing facts out of the message was rejected as unsafe. Eligibility intent is a provisional keyword set on the same footing as the existing weather keywords, so phrasings outside that set still route to retrieval. The scheme defaults to the single implemented rule set, since there is no scheme classifier yet. All-refused provenance yields ABSTAINED, so a stale or corrupt index looks like an unhelpful assistant rather than a wrongly-citing one, and CitationBatch.rejected is currently computed and then discarded. An empty batch with no rejections is ambiguous between "resolution not configured" and "nothing to cite"; the graph reads it as unconfigured so the original un-cited behaviour is preserved, which means a genuinely empty-but-refused retrieval would not abstain. Gate 2 (validate_referenced_citations) still has no caller because no answer generator exists, and citations remain page-level with no claim-level linkage. validate_referenced_citations and the resolver were already covered by unit tests, so this milestone adds graph-level integration tests only.

## 2026-10-07: Add injectable vision analysis with Gemini provider

**Decision:** Add `kisansathi.vision` with an injectable `VisionAnalyzer` protocol, `VisionResult`/`VisualObservation` domain models, image validation, and a `GeminiVisionAnalyzer` implementation. All normal tests use `FakeVisionAnalyzer`; the real provider is in `kisansathi.vision.providers`.

**Architecture:**
- Domain layer (`vision.models`, `vision.image`) depends only on the `VisionAnalyzer` protocol
- Infrastructure layer (`vision.providers`) implements `GeminiVisionAnalyzer` behind the protocol
- Factory `create_vision_analyzer(settings)` returns configured provider (fake or Gemini)
- LangGraph `build_graph` accepts optional `vision_analyzer` keyword argument; when provided, image input is analyzed before routing

**Provider behavior:**
- Image validation (MIME, magic bytes, size ≤10MB, dimensions ≤8192px, ≤50MP) runs BEFORE any provider call
- Tightly constrained system prompt demands structured JSON: observations with label, category (crop/disease/pest/stress/healthy/other), confidence (0-1), description
- Explicit uncertainty language required: "symptoms consistent with", "possibly", confidence scores
- Provider response validated against `VisionResult` schema before returning downstream
- Provider errors mapped to domain exceptions: `AnalyzerUnavailableError`, `AnalyzerTimeoutError`, `ImageValidationError`, `UnsupportedFormatError`, `ImageTooLargeError`
- Retry ONLY transient failures (429 rate limit, 5xx server errors, timeout); NEVER retry auth (401/403), bad request (400), or malformed output
- Exponential backoff (base 1s, max 2 retries) with jitter

**Observability (no secrets):**
- Structured logs: provider, model, latency_ms, status, observations_count, max_confidence, retry_count
- NEVER logged: API keys, raw image bytes, full request/response payloads

**Privacy:**
- Image bytes held in-memory only during request; discarded immediately after parsing response
- No disk persistence of images
- Gemini API policy: "Google does not use your API data to train models"

**Retrieval interaction:**
- Vision observations (confidence ≥0.5) augment retrieval query via `format_for_retrieval()`
- Existing hybrid retrieval (RRF) and reranker unchanged
- Vision output is an OBSERVATION, not evidence — never bypasses document grounding
- Final answers cite only retrieved documents; vision observations formatted with uncertainty markers in generation context

**Testing:**
- Normal CI: `FakeVisionAnalyzer` only, 600+ tests, zero network calls, no API keys
- Contract tests (manual): `tests/contract/test_gemini_vision_contract.py` skipped unless `RUN_REAL_VISION_TESTS=1` and `GEMINI_API_KEY` set
- Unit tests for: response parsing, malformed JSON, schema validation, confidence clamping, auth failure, rate limit, timeout, retry logic, oversized image, format rejection

**Configuration:**
- `KISANSAATHI_VISION_PROVIDER` (fake|gemini, default fake)
- `KISANSAATHI_VISION_API_KEY` (required for gemini)
- `KISANSAATHI_VISION_MODEL_NAME` (default gemini-1.5-flash-latest)
- `KISANSAATHI_VISION_TIMEOUT_SECONDS` (default 15.0)
- `KISANSAATHI_VISION_CONNECT_TIMEOUT_SECONDS` (default 5.0)
- `KISANSAATHI_VISION_MAX_RETRIES` (default 2)
- `KISANSAATHI_VISION_RETRY_BACKOFF_BASE` (default 1.0)
- `KISANSAATHI_VISION_MAX_IMAGE_BYTES` (default 10MB)
- `KISANSAATHI_VISION_TEMPERATURE` (default 0.0)
- `KISANSAATHI_VISION_MAX_OUTPUT_TOKENS` (default 1024)

**Reason:** Vision must be a replaceable infrastructure dependency. The protocol keeps domain pure and testable. Hosted multimodal LLM (Gemini) is the pragmatic first choice: zero model ops, multilingual out of box, structured output support, pay-per-use. Specialized classifier or local VLM can be swapped later without domain changes. Vision output is explicitly uncertain and never used as authoritative evidence — it only augments retrieval queries.

**Trade-off:** Generalist model accuracy on rare crops/diseases is moderate; mitigated by uncertainty framing, confidence thresholds, and mandatory document grounding. Cost: ~$0.0001/image (Gemini 1.5 Flash). Latency: ~1.5s. No offline mode in MVP (fake provider for tests only). No custom crop-disease dataset or training in this milestone.

## 2026-10-07: Add LLM-judge answer quality evaluation (RAGAS-compatible adapter)

**Decision:** Add `kisansathi.evaluation.answer_quality` with semantic metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall) via a provider-agnostic `LLMJudge` protocol. A `FakeLLMJudge` enables deterministic testing without API calls. Evaluation cases (`AnswerQualityEvaluationCase`) are versioned, strongly validated, and separate from deterministic answer checks (`AnswerExample`). An explicit opt-in CLI command (`answer-quality`) runs the evaluation; it is NOT part of normal pytest.

**Architecture:**
- **Separation of concerns**: Deterministic checks (status, language, citations, refusals) in `answer.py`; semantic LLM-judge metrics in `answer_quality.py`
- **RAGAS-compatible adapter** rather than direct RAGAS dependency — avoids unnecessary coupling, same metric semantics
- **Injectable judge**: `LLMJudge` protocol with `FakeLLMJudge` for tests; real providers implemented programmatically
- **Prompt-injection safety**: Evaluation prompts explicitly tell judge not to follow instructions in retrieved/generated content
- **Missing-input handling**: Metrics marked unavailable with reason when required inputs missing (no fake scores)
- **Reproducibility metadata**: Benchmark version, judge provider/model, temperature, timestamp, config recorded in every report
- **No API keys in logs/reports**

**Benchmarks:**
- Hand-authored English dataset: `data/evaluation/answer_quality/answer_quality_en_v1.json` (5 cases, corpus-versioned)
- No multilingual answer-quality labels fabricated — Hindi/Kannada/Telugu not reported without real labels

**Reports:**
- Machine-readable JSONL (one line per metric with judge metadata)
- Human-readable Markdown (summary table, per-case breakdown, limitations section)

**Reason:** Semantic answer quality requires LLM judgment and is fundamentally different from deterministic citation/status checks. Keeping it separate preserves pytest determinism (no LLM calls in CI) while providing opt-in RAGAS-style metrics. The adapter approach avoids adding RAGAS as a hard dependency (extra weight, version coupling) while maintaining metric compatibility.

**Trade-off:** Judge bias/variance affects scores; no confidence intervals or statistical significance. Results not comparable across different judge providers/models/configurations. Context precision/recall require reference contexts (unavailable when missing). Cost/latency of real judge calls. No offline mode for semantic metrics (fake judge for tests only).

## 2026-10-07: Add multilingual retrieval evaluation

**Decision:** Add `kisansathi.evaluation.multilingual` with per-language Hit@K and Recall@K metrics, explicit unavailable language handling, retrieval gap calculation, and native-speaker annotation workflow. The existing English benchmark (`retrieval-en-v1.json`, 10 queries) is preserved unchanged. Hindi, Kannada, and Telugu datasets exist as unavailable placeholders (`unavailable=true` with reason). Marathi has a source document but is not a target evaluation language.

**Architecture:**
- **Per-language versioned benchmarks**: `data/evaluation/retrieval/retrieval_{lang}_v{n}.json` with `unavailable` flag and `annotation_metadata`
- **Extended schema**: `RetrievalExample` adds `relevance_grades` (0-3), `annotator`, `annotation_timestamp_utc`; `RetrievalDataset` adds `unavailable`, `unavailable_reason`, `annotation_metadata`
- **Unavailable language handling**: Languages without genuine labels are reported as `unavailable` with reason; never as 0.0 scores
- **Retrieval gap**: `English Recall@5 - Target Language Recall@5` — only calculated when BOTH have genuine labels
- **No mechanical translation**: Queries must be authored by native speakers; translated English queries are invalid benchmarks
- **Native annotation workflow**: Requires source docs in target language, native speaker query formulation, chunk relevance identification, inter-annotator agreement (kappa ≥ 0.8)
- **Separation from answer-quality**: Retrieval evaluation = "Did we retrieve the right evidence?" (deterministic); Answer-quality = "Did the answer use evidence correctly?" (LLM judge)
- **Reproducibility**: Benchmark version, corpus checksum, retrieval config, K, timestamp, evaluator version in every report

**Benchmarks:**
- English: `retrieval-en-v1.json` (10 queries, corpus-versioned, measured: Dense Hit@5=0.80, BM25 Hit@5=0.90, Hybrid Hit@5=0.90, Hybrid+Reranker Hit@5=0.90, Hybrid+Reranker Recall@5=0.90)
- Hindi: `retrieval_hi_v1.json` (unavailable — no source docs, no labels)
- Kannada: `retrieval_kn_v1.json` (unavailable — no source docs, no labels)
- Telugu: `retrieval_te_v1.json` (unavailable — no source docs, no labels)
- Marathi: Source exists (MH PDMC bilingual) but excluded as non-target language

**Reports:**
- Machine-readable JSONL: per language, per system, per metric; unavailable languages marked with `unavailable=true` and reason
- Human-readable Markdown: summary tables, gap analysis, unavailable languages listed, limitations section

**Reason:** Multilingual retrieval quality cannot be measured without genuine relevance labels. The current corpus has no Hindi/Kannada/Telugu source documents. The design makes unavailable languages explicit rather than fabricating scores, and provides a clear path for native-speaker annotation when source documents are added. Keeping retrieval evaluation separate from answer-quality preserves the deterministic vs. semantic distinction.

**Trade-off:** English baseline is only 10 queries — not statistically significant. Gap analysis requires labels for both languages. No multilingual retrieval metric is reported until genuine relevance labels exist for that language. Marathi source document exists but is not evaluated unless added as a target language. Mechanical query translation is explicitly prohibited.
