"""Vision provider factory and public exports."""

from kisansathi.config import Settings, VisionProvider
from kisansathi.vision.models import VisionAnalyzer
from kisansathi.vision.fake import FakeVisionAnalyzer, make_fake_vision_analyzer
from kisansathi.vision.providers.gemini import GeminiVisionAnalyzer


def _normalize_provider(value: str | VisionProvider) -> str:
    """Normalize provider value to lowercase string."""
    if isinstance(value, VisionProvider):
        return value.value
    return str(value).strip().lower()


def create_vision_analyzer(settings: Settings | None = None) -> VisionAnalyzer:
    """Create a VisionAnalyzer based on configuration.

    Args:
        settings: Application settings. If None, loads from environment.

    Returns:
        VisionAnalyzer implementation (Fake, Gemini, etc.)

    Note:
        Tests should inject FakeVisionAnalyzer directly rather than using this factory.
        This factory is intended for production composition root.
    """
    from kisansathi.config import Settings as SettingsClass

    if settings is None:
        settings = SettingsClass.from_env()

    provider = _normalize_provider(settings.vision_provider)

    if provider == "fake" or settings.vision_api_key is None:
        return make_fake_vision_analyzer()

    if provider == "gemini":
        return GeminiVisionAnalyzer(settings)

    raise ValueError(f"Unknown vision provider: {provider}")


__all__ = [
    "create_vision_analyzer",
    "GeminiVisionAnalyzer",
    "FakeVisionAnalyzer",
    "make_fake_vision_analyzer",
]