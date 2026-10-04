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