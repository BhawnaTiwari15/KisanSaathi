# KisanSaathi

## Overview

KisanSaathi is a multimodal, multilingual farm advisory and government-scheme assistant for farmers. It combines text, voice, and image inputs with grounded retrieval, deterministic eligibility rules, weather information, citations, and fail-closed response handling.

The assistant answers questions about official Indian agricultural scheme documents — currently the PM-KISAN revised FAQ and the Maharashtra PM-KISAN Crop Damage Incentive (PDMC) 2026–27 notification. Answers are grounded in chunks retrieved from that corpus at query time; every claim carries a citation; and when evidence is empty, invalid, or insufficient, the system abstains or asks for clarification instead of guessing.

This repository is the complete Python implementation: an offline ingestion pipeline, dense/BM25/hybrid retrieval with an optional reranker, a LangGraph state machine, deterministic eligibility evaluation, weather and vision providers, an evaluation harness, and a mobile-friendly Streamlit UI. It runs end-to-end on CPU without any API keys (a real vision provider is opt-in; real answer-generation and speech-to-text providers are not yet configured).

## Features

- **Hybrid retrieval** — dense embeddings (BGE-M3 / Qdrant), lexical BM25 (SQLite FTS5), fused with Reciprocal Rank Fusion.
- **Second-stage reranking** — lazy cross-encoder (BGE-reranker-v2-M3) over a bounded candidate set.
- **Deterministic orchestration** — a compiled LangGraph routes requests to retrieval, eligibility, weather, or clarification.
- **Grounding with citations** — every answer is built from retrieved evidence and validated against a tracked source manifest.
- **Fail-closed guardrails** — abstention and clarification on missing, malformed, or invalid evidence.
- **Deterministic eligibility** — PM-KISAN rules whose conditions quote verbatim corpus excerpts.
- **Optional multimodal inputs** — image analysis (Gemini, opt-in), weather (Open-Meteo), audio upload (transcription boundary only).
- **Honest evaluation** — hand-labeled benchmarks pinned to corpus checksums; languages without labels are reported as unavailable, never scored by proxy.

## Architecture

The application is a thin Streamlit presentation layer over a compiled LangGraph. A composition root assembles the real service once per process and injects every external dependency (retriever, weather client, citation resolver, eligibility evaluator, language detector, and optional vision/speech providers).

```
User (text / image / audio / location / eligibility facts)
  │
  ▼
Streamlit UI
  │
  ▼
Composition root (ui.composition_root.build_application_service)
  │
  ▼
LangGraph StateGraph (orchestration.graph.build_graph)
  │
  START → speech_to_text → vision → route_request
  │
  ├── Retrieval : detect_language → retrieve → resolve_citations
  │               → generate_answer → validate_and_attach_citations
  ├── Eligibility: detect_language → eligibility → resolve_citations → same spine
  ├── Weather   : weather (requires explicit coordinates)
  └── Clarify   : underspecified or missing-fact requests
  │
  ▼
apply_guardrails → finalize_response
  │
  ▼
AssistantResponse (answered / needs clarification / abstained + citations)
  │
  ▼
Streamlit UI (status, sources expander, footers)
```

All evidence-bearing routes converge on one `resolve_citations` node; the citation resolver — not the graph — decides what is citable. The graph never constructs a `Citation` itself and never reads chunk text, which prevents citation fabrication. The weather route bypasses the citation layer because weather carries no document evidence. A deterministic script-based language detector runs on the retrieval and eligibility paths; ambiguous script mixes fall back gracefully without crashing.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Embeddings / reranker | Sentence-Transformers, `BAAI/bge-m3` (1024-d), `BAAI/bge-reranker-v2-m3` |
| Vector store | Qdrant (local persistent storage) |
| Lexical index | SQLite FTS5 BM25 (standard library) |
| Orchestration | LangGraph `StateGraph` |
| UI | Streamlit (optional extra) |
| PDF parsing | PyMuPDF (text-based PDFs only) |
| Weather | Open-Meteo forecast API (no key) |
| Vision | httpx-based Gemini provider (opt-in) |
| Tests | pytest (771 passed, 11 skipped, zero network) |

## Data Sources & Provenance

Documents are added by hand only after the issuing authority and official source URL have been verified. `data/sources.json` records one entry per logical document (authority, official URL, publication/effective dates); an HTTPS URL or a government-looking hostname alone is not treated as proof of authority. Unknown dates remain null.

- Raw PDFs are retained under `data/raw/`, keyed by SHA-256 checksum (gitignored).
- Extracted, chunked records are written to versioned JSONL under `data/processed/` (~77 KB) and committed to the repository. Ingestion is deterministic: re-running unchanged inputs changes nothing.
- Every evaluation benchmark records `corpus_version` checksums that tie labels to the exact committed chunks.

Current corpus (2 documents):

| Document | Jurisdiction |
|---|---|
| PM-KISAN revised FAQ | India (central scheme) |
| Maharashtra PM-KISAN Crop Damage Incentive (PDMC) 2026–27 notification | Maharashtra |

## Retrieval Pipeline

Three retrievers are implemented over the same processed corpus:

- **Dense** — `DenseRetriever` + `EmbeddingService` (BGE-M3) over the local Qdrant collection.
- **BM25** — `BM25Retriever` over a SQLite FTS5 index.
- **Hybrid** — `HybridRetriever` fuses dense and BM25 rankings with Reciprocal Rank Fusion.

A `Reranker` rescores a bounded candidate set (default `candidate_depth=10`, `top_k=5`) with query–document cross-encoder relevance; it is a second stage used by the evaluation harness (`RerankedHybridRetriever`) and is separate from dense, BM25, and RRF scoring.

The current production graph uses the dense retriever for the retrieval route. Both indexes are derived artifacts — rebuilt from the committed processed corpus, never committed themselves:

```
python -m kisansathi.build_indexes                # dense + BM25
python -m kisansathi.build_indexes --dense-only   # embeddings only
python -m kisansathi.build_indexes --bm25-only    # lexical only
```

Per-file errors are reported on stderr without aborting the run; the command exits non-zero if any file failed.

## Multimodal Capabilities

**Vision.** `VisionAnalyzer` is an injectable protocol; domain code depends only on it. Results are structured `VisualObservation` objects with explicit uncertainty ("symptoms consistent with", confidence scores) — observations, never diagnoses. A successful observation can augment the retrieval query, but it never bypasses document grounding. The default provider is a deterministic fake; `GeminiVisionAnalyzer` is opt-in via `KISANSAATHI_VISION_PROVIDER=gemini` plus an API key. It sends the key in the `x-goog-api-key` header (never in the URL), validates image MIME/magic bytes/size/dimensions before the provider call, demands and schema-validates structured JSON output, retries only transient failures (429, 5xx, timeout), and never logs keys, image bytes, or full payloads. A real-provider contract test exists but is opt-in and skipped in CI. When an image is uploaded and the analyzer is unavailable, the request abstains; text-only requests never invoke vision and are unaffected.

**Weather.** `OpenMeteoClient` returns typed forecast data (no API key) and requires explicit latitude/longitude — the UI never infers a location. The weather route bypasses citations (weather has no document evidence) and reports "unavailable" on client failure or missing measurements instead of failing the app.

**Voice.** The UI accepts audio upload, but no production speech-to-text provider is configured. The `speech_to_text` boundary validates audio (`validate_audio_input`) and, when a provider is injected, transcribes and validates the result before routing; with no provider, audio is left untranscribed and the UI states that transcription is unavailable.

## Eligibility

The PM-KISAN evaluator (`kisansathi.eligibility`) answers eligibility questions from an explicit, versioned rule set in which every condition cites a verbatim excerpt of an ingested chunk. Facts arrive through a caller-supplied tri-state request ("Not sure" / "Yes" / "No"); facts are never parsed out of free text, because inferring a farmer's landholding or income-tax status from keywords would silently produce a wrong verdict about a real benefit. Absent or malformed requests yield clarification, and conditions the corpus does not support (including the commonly assumed two-hectare land cap) return insufficient information rather than a guess.

## Grounded Responses & Citations

The citation layer records *where an answer came from*; it does not assert that the source is factually correct. The flow is retrieval/eligibility → `resolve_citations` → generation → `validate_and_attach_citations` → guardrails → response.

- `ManifestSourceRegistry` loads `data/sources.json`; `CitationResolver` turns requested chunk IDs into validated citations.
- Citation IDs are structured URIs (`source_id:chunk_id`); malformed IDs, unknown sources, invalid page ranges, and unsupported schemes are rejected.
- Post-generation, every citation referenced by the answer is re-validated against the resolved batch; a hallucinated ID forces clarification or abstention.
- If the manifest cannot be loaded, the resolver is disabled and responses cite nothing rather than fabricate sources.

When no answer generator is configured (the current default), the retrieval route returns the retrieved evidence with its resolved citations and a deterministic placeholder note; eligibility and weather responses are fully formed.

## Guardrails

A single deterministic layer (`kisansathi.guardrails`) runs after citation/generation processing, before the response is finalized:

| Situation | Behavior |
|---|---|
| Successful grounded answer | Answered with validated citations |
| Empty retrieval / no relevant chunks | **ABSTAINED** |
| Missing or insufficient eligibility facts | **NEEDS_CLARIFICATION** (never a guess) |
| Invalid / unregistered citation | Safe failure (abstain, no fabrication) |
| Malformed generation or hallucinated citation ID | Clarify or abstain |
| LLM/provider failure | **ABSTAINED** |
| Vision unavailable | Does not block independent text routes |
| Weather client failure | Weather reports unavailable; app unaffected |
| Internal graph error | UI shows one fixed safe message; details logged server-side only |

## Evaluation

The automated suite runs **771 passed tests (11 skipped)** with zero network calls:

```
python -m pytest -q
python -m compileall -q src tests
```

Versioned, hand-labeled benchmarks live under `data/evaluation/` and are pinned to the processed-corpus checksums. Evaluation is split into deliberately separate tiers:

1. **Deterministic checks (CI)** — retrieval hit/recall, citation validity, eligibility verdicts, answer-language matching, safety/refusal behavior, guardrail statuses.
2. **Deterministic retrieval benchmark** — `python -m kisansathi.evaluation`, a hand-labeled 10-query English set over the built dense and BM25 stores (no LLM involved).
3. **Semantic answer-quality (opt-in)** — `python -m kisansathi.evaluation.main answer-quality` scores faithfulness, answer relevance, context precision, and context recall against a RAGAS-style adapter over the `LLMJudge` protocol. It runs only when invoked explicitly and uses `FakeLLMJudge` (no API calls) by default; the report records `judge_provider`/`judge_model` metadata so a fake judge's scores are never mistaken for a real model's. This command is evaluation infrastructure, not a statement about answer quality today.

### Multilingual evaluation

`python -m kisansathi.evaluation.main multilingual` reports per-language Hit@K and Recall@K (`data/evaluation/retrieval/retrieval_{lang}_v{n}.json`):

| Language | Genuine relevance labels | Status |
|---|---|---|
| English | 10 hand-labeled queries | Evaluated |
| Hindi | None yet | Unavailable — genuine labels do not yet exist |
| Kannada | None yet | Unavailable — genuine labels do not yet exist |
| Telugu | None yet | Unavailable — genuine labels do not yet exist |
| Marathi | None (source exists, bilingual) | Not a target evaluation language |

No multilingual retrieval metric is reported until genuine relevance labels exist for that language; mechanical translation of English queries is explicitly not an acceptable substitute. Unavailable languages are reported as unavailable, never as 0.0.

## Results

Measured on the current English benchmark (`data/evaluation/retrieval_en_v1.json`, 10 queries, labels hand-reviewed against the source documents, dense + BM25 stores built from the committed corpus):

| System | Hit@5 | Recall@5 |
|---|---|---|
| Dense | 0.80 | 0.80 |
| BM25 | 0.90 | 0.8333 |
| Hybrid (RRF) | 0.90 | 0.85 |
| Hybrid + Reranker | **0.90** | **0.90** |

These numbers are reproducible from this repository. The benchmark is English-only, 10 queries, small and directional, and **not statistically representative**. It is not a claim about production-scale performance.

## Current Limitations

- **Small English benchmark** — single 10-query set; no confidence intervals.
- **Small corpus** — two official documents; the assistant does not cover all Indian schemes.
- **Multilingual labels unavailable** — Hindi, Kannada, and Telugu have no genuine relevance labels; Marathi is not a target evaluation language. No multilingual retrieval score exists.
- **No production answer generator** — retrieval answers are deterministic placeholders describing what was retrieved; a real LLM provider is a planned integration (only investment: implement the `LLMClient` protocol and inject a `DefaultAnswerGenerator`).
- **No production speech-to-text** — audio upload is accepted but not transcribed.
- **Vision provider dependency** — defaults to a deterministic fake; real analysis requires the opt-in Gemini provider and an API key. Vision output is observation-level and uncertain, never a diagnosis.
- **Answer-quality judge limitations** — scores carry judge bias/variance, no significance testing, and are not comparable across judge providers.
- **Model / resource requirements** — BGE-M3 (~2.3 GB) and reranker (~2.1 GB) weights download lazily on first use; dense retrieval needs ~8 GB RAM for comfortable headroom.
- **Deployment limitations** — single-process VPS deployment; no load balancing, authentication, reverse proxy, or TLS.
- **Extraction limitations** — text-based PDFs only; no OCR and no automatic PDF downloading; documents are added by hand after authority verification.

## Local Setup

**Prerequisites:** Python 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"          # + pytest, ruff, compileall
python -m pip install -e ".[ui]"           # + Streamlit UI
```

**Environment variables.** Copy `.env.example` to `.env` for local values, or export the `KISANSAATHI_*` variables. Settings are read from process environment variables only (`.env` files are not parsed automatically). Key variables:

| Variable | Default | Description |
|---|---|---|
| `KISANSAATHI_ENV` | `development` | `development`, `test`, `production` |
| `KISANSAATHI_DEFAULT_LANGUAGE` | `en` | Default UI language: `en` or `hi` |
| `KISANSAATHI_EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Embedding model |
| `KISANSAATHI_RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `KISANSAATHI_QDRANT_STORAGE_PATH` | `data/qdrant` | Local Qdrant storage directory |
| `KISANSAATHI_BM25_STORAGE_PATH` | `data/bm25.sqlite3` | SQLite FTS5 BM25 database |
| `KISANSAATHI_SOURCES_MANIFEST` | `data/sources.json` | Source manifest used for citations |
| `KISANSAATHI_VISION_PROVIDER` | `fake` | `fake` or `gemini` |
| `KISANSAATHI_VISION_API_KEY` | (empty) | Required only for the Gemini provider |
| `KISANSAATHI_LOG_LEVEL` | `INFO` | Logging level for entry points |

**Never commit `.env` or real API keys.** The full variable inventory is documented in `.env.example`.

## Running the Application

**Prepare the corpus and indexes.** The processed corpus is committed; build the derived indexes before starting (first dense build downloads the embedding weights):

```
python -m kisansathi.build_indexes
```

Ingestion is only needed when adding new documents: place reviewed PDFs under `data/incoming/` per their `data/sources.json` entry, then run `python -m kisansathi.ingestion`.

**Start the UI:**

```
python -m streamlit run src/kisansathi/ui/streamlit_app.py
```

The UI accepts a free-text question, a language selector, optional image, optional audio, explicit latitude/longitude for weather, and optional PM-KISAN facts as tri-state selections. Responses render by status — answered / needs clarification / abstained — with sources in an expander and status-specific footers. Example flows that work today: ask *"Which farmer families are eligible for PM-KISAN benefits?"* (cites the tracked sources), upload a crop image (an observation from the fake provider, not a diagnosis), or ask a weather question with explicit coordinates (returns without document citations).

## Deployment

The current target is a **single-process virtual private server**: a Linux venv running the Streamlit app under systemd. No containerization, authentication, reverse proxy, or TLS is configured.

1. Provision a VM (~2 vCPU / 8 GB RAM / 40 GB disk).
2. Clone the repository and create the venv; install CPU-only torch first so the default GPU wheel is not pulled, then `pip install -e ".[ui]"`.
3. Create a `chmod 600` `.env` with the needed `KISANSAATHI_*` values (use absolute index and storage paths).
4. Build the retrieval indexes: `python -m kisansathi.build_indexes` (downloads model weights on first run).
5. Run under systemd (`EnvironmentFile=/opt/kisansathi/.env`, `ExecStart=/opt/kisansathi/.venv/bin/python -m streamlit run src/kisansathi/ui/streamlit_app.py`, `Restart=always`). Health check: `curl http://<host>:8501/_stcore/health` → `"ok"`. Logs go to stderr, collected via `journalctl -u kisansathi`.
6. Update by `git pull`, rebuilding indexes if the corpus changed, then `systemctl restart kisansathi`.

The bundled CPU torch and fp32 BGE-M3 weights exceed free managed tiers (e.g. Streamlit Community Cloud's ~2.7 GB memory budget), so a ~2 vCPU / 8 GB RAM VPS is used to give the estimated 3.5–4.5 GB peak working set headroom.

## Project Structure

```
src/kisansathi/
├── ingestion/            # PDF -> chunked JSONL pipeline (data/processed)
├── retrieval/            # embeddings, dense/BM25/hybrid retrievers, reranker, stores
├── orchestration/        # LangGraph graph + nodes (graph.py)
├── eligibility/          # deterministic PM-KISAN rule evaluator
├── citations/            # source manifest registry + citation resolver
├── generation/           # AnswerGenerator + LLMClient protocol
├── guardrails/           # final abstention / clarification policy
├── language/             # deterministic script-based language detector
├── weather/              # Open-Meteo client
├── voice/                # SpeechToText contract + audio validation
├── vision/               # VisionAnalyzer protocol + fake/Gemini providers
├── ui/                   # Streamlit app, presentation helpers, composition root
├── evaluation/           # retrieval / answer / answer-quality / multilingual harness
├── domain/               # shared schemas (responses, citations, requests)
├── config.py             # environment-backed settings
├── build_indexes.py      # dense + BM25 index-build CLI
└── logging_setup.py      # entry-point logging configuration
data/
├── sources.json          # tracked source manifest
├── processed/            # committed chunk corpus (v1)
├── evaluation/           # versioned retrieval/answer/answer-quality benchmarks
└── (incoming/, raw/, qdrant/, bm25.sqlite3)   # gitignored working artifacts
tests/
├── test_*.py             # 38 modules of unit + integration tests
└── contract/             # opt-in real-provider tests (skipped in CI)
```