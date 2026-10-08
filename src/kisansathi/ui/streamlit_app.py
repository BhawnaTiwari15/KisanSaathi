"""KisanSaathi Streamlit presentation layer.

**Thin presentation layer only.** All business logic (retrieval, eligibility,
citations, weather, vision, voice, guardrails) lives in the respective
packages below ``src/kisansathi/``.

This module must NOT:
- call Qdrant, BM25, or reranker directly
- call Gemini or Whisper directly
- implement eligibility rules
- construct citations manually
- implement weather logic
- contain RAG prompts
- contain business decision logic

Run locally from the project root:

    .venv\\Scripts\\python.exe -m streamlit run src/kisansathi/ui/streamlit_app.py
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from kisansathi.domain.schemas import Language, ResponseStatus
from kisansathi.eligibility.models import FACT_NAMES, EligibilityRequest
from kisansathi.logging_setup import configure_logging
from kisansathi.ui import helpers as ui
from kisansathi.ui.composition_root import ApplicationService, build_application_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application boundary (cached per process)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Starting the assistant...")
def _load_application() -> ApplicationService:
    """Build the application service once per process and reuse it.

    Building opens the local Qdrant collection, so one shared instance also
    keeps a single store handle for the whole server process.
    """
    return build_application_service()


# ---------------------------------------------------------------------------
# Clear callbacks (registered via on_click so widget state changes happen
# before the widget is instantiated on the next run)
# ---------------------------------------------------------------------------


def _clear_question() -> None:
    st.session_state["question_text"] = ""


def _clear_eligibility_facts() -> None:
    for fact in FACT_NAMES:
        st.session_state[fact] = "Not sure"


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def _render_text_input() -> str:
    """Render the primary text input and return the farmer's question."""
    st.markdown("### 💬 Your question")
    question = st.text_input(
        "Ask KisanSaathi:",
        placeholder="e.g. What scheme am I eligible for?",
        key="question_text",
    )
    st.button("Clear", key="clear_question", on_click=_clear_question)
    return question


def _render_language_selector() -> Language:
    """Render the language selector and return the selected Language."""
    st.markdown("### 🌐 Language")
    return st.radio(
        "Select language:",
        options=list(ui.LANGUAGE_LABELS),
        format_func=lambda lang: ui.LANGUAGE_LABELS[lang],
        key="language_selector",
        horizontal=True,
    )


def _render_image_input() -> tuple[bytes, str, str] | None:
    """Render the optional image uploader and return (bytes, content_type, name)."""
    st.markdown("### 📷 Crop image (optional)")
    uploaded = st.file_uploader(
        "Upload a crop image:",
        type=["jpg", "jpeg", "png"],
        help="Supported: JPG, PNG. Max 10 MB (vision layer enforcement).",
        key="image_uploader",
    )
    if uploaded is None:
        return None
    st.image(uploaded, caption="Preview", use_container_width=True)
    image_bytes = uploaded.getvalue()
    if not image_bytes:
        return None
    content_type = ui.image_content_type_for(getattr(uploaded, "type", None))
    filename = getattr(uploaded, "name", None) or "upload"
    return image_bytes, content_type, filename


def _render_audio_input() -> tuple[bytes, str] | None:
    """Render the optional audio uploader and return (bytes, content_type).

    Audio bytes are passed to the application's speech boundary; no Whisper is
    implemented in Streamlit, and transcription only happens when the
    composition root has a speech provider configured.
    """
    st.markdown("### 🎤 Audio input (optional)")
    uploaded = st.file_uploader(
        "Upload audio (WAV, MP3, M4A):",
        type=["wav", "mp3", "m4a"],
        help="Audio bytes are passed to the application boundary; transcription "
             "requires a configured speech provider.",
        key="audio_uploader",
    )
    if uploaded is None:
        return None
    content_type = getattr(uploaded, "type", None) or "audio/wav"
    st.audio(uploaded, format=content_type)
    audio_bytes = uploaded.getvalue()
    if not audio_bytes:
        return None
    return audio_bytes, content_type


def _render_location_input() -> tuple[float, float] | None:
    """Render explicit latitude/longitude inputs for the weather route.

    Returns ``(latitude, longitude)`` or None. The UI never infers or
    defaults a location.
    """
    st.markdown("### 📍 Location (for weather)")
    left, right = st.columns(2)
    with left:
        latitude_text = st.text_input(
            "Latitude:",
            placeholder="-90 to 90",
            key="lat_input",
            help="Required for weather. Example: 28.6139",
        )
    with right:
        longitude_text = st.text_input(
            "Longitude:",
            placeholder="-180 to 180",
            key="lon_input",
            help="Required for weather. Example: 77.2090",
        )

    location = ui.parse_location(latitude_text, longitude_text)
    if location is None:
        if latitude_text.strip() or longitude_text.strip():
            st.caption("Enter both values as numbers within range to use weather.")
        else:
            st.caption("Leave blank if your question does not require weather.")
    return location


def _render_eligibility_inputs() -> EligibilityRequest | None:
    """Render optional PM-KISAN facts as tri-state selections.

    The UI only collects facts. The eligibility engine decides the result;
    NO eligibility conditions or rules are duplicated here.
    """
    st.markdown("### ✅ PM-KISAN Eligibility facts (optional)")
    st.caption(
        "Select the facts you know. Leave “Not sure” if you do not know the "
        "value; the eligibility engine will report missing facts."
    )

    selections: dict[str, str] = {}
    left, right = st.columns(2)
    with left:
        for fact in FACT_NAMES[:2]:
            selections[fact] = st.selectbox(
                ui.FACT_LABELS[fact], ui.FACT_SELECTIONS, key=fact
            )
    with right:
        for fact in FACT_NAMES[2:]:
            selections[fact] = st.selectbox(
                ui.FACT_LABELS[fact], ui.FACT_SELECTIONS, key=fact
            )

    st.button(
        "Clear eligibility facts",
        key="clear_eligibility",
        on_click=_clear_eligibility_facts,
    )
    return ui.build_eligibility_request(selections)


# ---------------------------------------------------------------------------
# Response rendering
# ---------------------------------------------------------------------------


def _render_response(response: Any) -> None:
    """Render one AssistantResponse with its status-specific presentation."""
    if response is None:
        return

    st.markdown(f"**{ui.status_label(response.status)}**")

    status = response.status
    if status is ResponseStatus.ANSWERED:
        st.success(response.text)
    elif status is ResponseStatus.NEEDS_CLARIFICATION:
        st.warning(response.text)
    elif status is ResponseStatus.ABSTAINED:
        st.error(response.text)
    else:
        st.info(response.text)

    if response.citations:
        with st.expander("Sources", expanded=False):
            for index, citation in enumerate(response.citations, start=1):
                st.caption(f"[{index}] {ui.render_citation_text(citation)}")

    footer = ui.status_footer(status)
    if footer:
        st.caption(footer)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def _submit(
    question: str,
    language: Language,
    image: tuple[bytes, str, str] | None,
    audio: tuple[bytes, str] | None,
    location: tuple[float, float] | None,
    eligibility: EligibilityRequest | None,
) -> None:
    """Shape inputs into graph state, invoke the boundary, keep the response."""
    if image is not None:
        image_bytes, image_content_type, image_filename = image
    else:
        image_bytes, image_content_type, image_filename = None, None, None
    audio_bytes, audio_content_type = audio if audio is not None else (None, None)

    state = ui.build_request_state(
        question,
        language,
        image_bytes=image_bytes,
        image_content_type=image_content_type,
        image_filename=image_filename,
        audio_bytes=audio_bytes,
        audio_content_type=audio_content_type,
        location=location,
        eligibility=eligibility,
    )

    try:
        app = _load_application()
    except Exception:
        logger.exception("failed to build the application service")
        st.error(ui.SAFE_ERROR_MESSAGE)
        return

    response, error = ui.invoke_graph_safely(app.graph, state)
    if error is not None:
        st.error(error)
        return
    st.session_state["last_response"] = response


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``streamlit run``."""
    configure_logging()
    st.set_page_config(
        page_title="KisanSaathi — Farmer Assistant",
        page_icon="🌾",
        layout="centered",
    )

    st.title("🌾 KisanSaathi")
    st.caption("Multilingual AI assistant for farmers")

    question = _render_text_input()
    language = _render_language_selector()
    image = _render_image_input()
    audio = _render_audio_input()
    location = _render_location_input()
    eligibility = _render_eligibility_inputs()

    st.markdown("---")
    if st.button("Get Advice", type="primary", use_container_width=True):
        if not ui.validate_text(question):
            st.warning("Please enter a question before requesting advice.")
        else:
            _submit(question, language, image, audio, location, eligibility)

    response = st.session_state.get("last_response")
    if response is not None:
        st.markdown("### 📋 Response")
        _render_response(response)

    st.markdown("---")
    st.caption(
        "KisanSaathi is a deterministic foundation for farmer assistance. "
        "Answers are grounded in official document citations. "
        "Optional modalities (image, weather) may be used but "
        "their failure does not block an independently answerable text request."
    )


if __name__ == "__main__":
    main()
