"""KisanSaathi Streamlit user-interface package.

Streamlit is a thin presentation layer only. All business logic (retrieval,
eligibility, citations, weather, vision, voice, guardrails) lives in the
respective packages below the ``src/kisansathi/`` boundary.

Modules:

- ``composition_root`` assembles injectable dependencies and builds the
  LangGraph application service that the UI invokes.
- ``helpers`` holds the pure, Streamlit-free presentation helpers that shape
  input into graph state and graph output into display strings.
- ``streamlit_app`` is the ``streamlit run`` entry point.

This package deliberately performs no imports at package-init time: importing
``kisansathi.ui`` must not pull in the orchestration graph or the guardrails
layer, so the helpers stay importable without the optional ``ui`` extra.
"""
