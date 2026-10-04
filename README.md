# KisanSaathi

KisanSaathi is a multilingual, multimodal AI assistant for farmers. This repository currently contains only the Python project foundation: environment-backed settings, shared domain contracts, and their tests. Feature implementations are intentionally not included yet.

## Requirements

- Windows
- Python 3.12

## Set up

From PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

From Command Prompt, activate with `.venv\Scripts\activate.bat` instead. The optional development install provides pytest and Ruff; the initial test suite itself uses Python's standard library.

Copy `.env.example` to `.env` when configuring local settings. Do not put secrets in `.env.example` or commit `.env`. Settings are read from process environment variables; this foundation does not parse `.env` files automatically.

## Run checks

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

After installing the development extra, `python -m pytest` and `ruff check .` are also available.

## Current scope

No ingestion, retrieval, RAG, eligibility rules, model integrations, external tools, orchestration, or user interface have been implemented. The shared schemas are intentionally limited to user messages, assistant responses, language/status values, and source citations.