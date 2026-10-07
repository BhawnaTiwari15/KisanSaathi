"""Multilingual evaluation infrastructure."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kisansathi.evaluation.retrieval import (
    RetrievalDataset,
    RetrievalEvaluationResult,
    evaluate_systems,
)
from kisansathi.evaluation.schemas import Language, RetrievalCategory, Difficulty, load_retrieval_dataset


@dataclass(frozen=True, slots=True)
class LanguageMetrics:
    """Retrieval metrics for a single language."""

    language: Language
    dataset_version: str
    corpus_version: str
    query_count: int
    systems: dict[str, dict[str, float]]  # system_name -> {hit_rate, recall}

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language.value,
            "dataset_version": self.dataset_version,
            "corpus_version": self.corpus_version,
            "query_count": self.query_count,
            "systems": self.systems,
        }


@dataclass(frozen=True, slots=True)
class MultilingualReport:
    """Cross-language comparison report."""

    evaluation_timestamp_utc: str
    corpus_version: str
    system_version: str
    top_k: int
    languages: tuple[LanguageMetrics, ...]
    english_baseline: LanguageMetrics | None = None

    def get_gap(self, language: Language, metric: str) -> float | None:
        """Get the gap between English and the given language for a metric."""
        if self.english_baseline is None:
            return None
        lang_metrics = next((lm for lm in self.languages if lm.language == language), None)
        if lang_metrics is None:
            return None
        # Average across systems
        eng_val = sum(s.get(metric, 0.0) for s in self.english_baseline.systems.values()) / max(1, len(self.english_baseline.systems))
        lang_val = sum(s.get(metric, 0.0) for s in lang_metrics.systems.values()) / max(1, len(lang_metrics.systems))
        return eng_val - lang_val

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_timestamp_utc": self.evaluation_timestamp_utc,
            "corpus_version": self.corpus_version,
            "system_version": self.system_version,
            "top_k": self.top_k,
            "languages": [lm.to_dict() for lm in self.languages],
            "english_baseline": self.english_baseline.to_dict() if self.english_baseline else None,
        }


def find_retrieval_datasets(base_path: str | Path) -> dict[Language, Path]:
    """Find available retrieval evaluation datasets by language.

    Looks for files matching: retrieval_{lang}_v{version}.json
    """
    base = Path(base_path)
    datasets: dict[Language, Path] = {}

    if not base.exists():
        return datasets

    for lang in Language:
        pattern = f"retrieval_{lang.value}_v*.json"
        matches = list(base.glob(pattern))
        if matches:
            # Use the latest version (highest version number)
            latest = max(matches, key=lambda p: int(p.stem.split("_v")[-1]) if "_v" in p.stem else 0)
            datasets[lang] = latest

    return datasets


def load_available_retrieval_datasets(base_path: str | Path) -> dict[Language, RetrievalDataset]:
    """Load all available retrieval datasets."""
    datasets = find_retrieval_datasets(base_path)
    loaded: dict[Language, RetrievalDataset] = {}

    for lang, path in datasets.items():
        try:
            loaded[lang] = load_retrieval_dataset(path)
        except Exception as e:
            # Log warning but continue with other languages
            print(f"Warning: Failed to load {lang.value} dataset from {path}: {e}")

    return loaded


def run_multilingual_retrieval_evaluation(
    systems: dict[str, Any],  # retriever objects
    datasets: dict[Language, RetrievalDataset],
    k: int = 5,
    system_version: str = "",
) -> MultilingualReport:
    """Run retrieval evaluation across all available languages."""
    from datetime import datetime

    language_results: list[LanguageMetrics] = []
    english_baseline: LanguageMetrics | None = None

    for lang, dataset in datasets.items():
        if not dataset.examples:
            continue

        results = evaluate_systems(systems, dataset, k=k, system_version=system_version)

        systems_dict = {}
        for sys in results.systems:
            systems_dict[sys.system_name] = {
                "hit_rate_at_k": sys.hit_rate_at_k,
                "recall_at_k": sys.recall_at_k,
            }

        lm = LanguageMetrics(
            language=lang,
            dataset_version=dataset.evaluation_set_version,
            corpus_version=dataset.corpus_version,
            query_count=len(dataset.examples),
            systems=systems_dict,
        )
        language_results.append(lm)

        if lang == Language.ENGLISH:
            english_baseline = lm

    return MultilingualReport(
        evaluation_timestamp_utc=datetime.utcnow().isoformat() + "Z",
        corpus_version=next(iter(datasets.values())).corpus_version if datasets else "",
        system_version=system_version,
        top_k=k,
        languages=tuple(language_results),
        english_baseline=english_baseline,
    )


def print_multilingual_summary(report: MultilingualReport) -> str:
    """Generate a human-readable multilingual summary."""
    lines = [
        "# Multilingual Retrieval Evaluation Report",
        f"**Timestamp:** {report.evaluation_timestamp_utc}",
        f"**Corpus:** {report.corpus_version}",
        f"**System:** {report.system_version}",
        f"**Top-K:** {report.top_k}",
        "",
        "## Per-Language Metrics",
        "",
    ]

    # Table header
    systems = set()
    for lm in report.languages:
        systems.update(lm.systems.keys())
    systems = sorted(systems)

    if not systems:
        lines.append("No systems evaluated.")
        return "\n".join(lines)

    # Print table for each metric
    for metric in ("hit_rate_at_k", "recall_at_k"):
        metric_name = "Hit@K" if metric == "hit_rate_at_k" else "Recall@K"
        lines.append(f"### {metric_name}")
        lines.append("")
        header = ["Language", "Queries"] + systems
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for lm in report.languages:
            row = [lm.language.value, str(lm.query_count)]
            for sys_name in systems:
                val = lm.systems.get(sys_name, {}).get(metric, 0.0)
                row.append(f"{val:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Gap analysis
    if report.english_baseline:
        lines.append("## English vs Indic Language Gap (Recall@K)")
        lines.append("")
        lines.append("| Language | Gap vs English |")
        lines.append("| --- | --- |")
        for lm in report.languages:
            if lm.language == Language.ENGLISH:
                continue
            gap = report.get_gap(lm.language, "recall_at_k")
            if gap is not None:
                lines.append(f"| {lm.language.value} | {gap:+.3f} |")
        lines.append("")

    return "\n".join(lines)