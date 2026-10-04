# KisanSaathi

KisanSaathi is a multilingual, multimodal AI assistant for farmers. The current implementation includes the Python foundation and an offline ingestion pipeline for manually curated, official PDF sources. Retrieval, RAG, eligibility, model integrations, and user interfaces are not implemented.

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

From Command Prompt, activate with `.venv\Scripts\activate.bat` instead. The runtime dependency is PyMuPDF, used to extract text and layout information from local PDFs. The optional development install provides pytest and Ruff; tests use Python's standard library test runner.

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

## Current scope

No retrieval, embeddings, vector database, BM25, reranking, RAG, eligibility rules, model integrations, external tools, orchestration, or user interface have been implemented. The shared conversation schemas remain separate from ingestion-owned document models.