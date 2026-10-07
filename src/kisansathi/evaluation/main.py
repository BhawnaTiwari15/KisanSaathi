"""Evaluation command-line interface."""

import argparse
import asyncio
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from kisansathi.config import Settings
from kisansathi.evaluation.answer_quality import (
    AnswerQualityEvaluationSummary,
    FakeLLMJudge,
    JudgeConfig,
    evaluate_answer_quality_dataset,
)
from kisansathi.evaluation.report import (
    generate_answer_quality_jsonl,
    generate_answer_quality_markdown,
    generate_retrieval_jsonl,
    generate_retrieval_markdown,
    write_jsonl,
)
from kisansathi.evaluation.retrieval import RetrievalEvaluationResult, evaluate_systems
from kisansathi.evaluation.schemas import Language, load_answer_quality_dataset, load_retrieval_dataset
from kisansathi.retrieval.bm25_retriever import BM25Retriever
from kisansathi.retrieval.bm25_store import BM25Store
from kisansathi.retrieval.embeddings import EmbeddingService
from kisansathi.retrieval.hybrid_retriever import HybridRetriever
from kisansathi.retrieval.reranker import Reranker
from kisansathi.retrieval.retriever import DenseRetriever
from kisansathi.retrieval.vector_store import QdrantVectorStore


def _build_retrieval_systems(settings: Settings):
    """Build all retrieval systems for evaluation."""
    vector_store = QdrantVectorStore(settings)
    bm25_store = BM25Store(settings)
    dense = DenseRetriever(EmbeddingService(settings), vector_store)
    bm25 = BM25Retriever(bm25_store)
    hybrid = HybridRetriever(dense, bm25)
    reranker = Reranker(settings, candidate_depth=10, default_top_k=5)

    from kisansathi.evaluation.recall import RerankedHybridRetriever
    return {
        "DenseRetriever": dense,
        "BM25Retriever": bm25,
        "HybridRetriever": hybrid,
        "HybridRetriever+Reranker": RerankedHybridRetriever(
            hybrid, reranker, candidate_depth=10
        ),
    }, vector_store, bm25_store


def run_retrieval_evaluation(
    dataset_path: str,
    output_jsonl: str | None = None,
    output_markdown: str | None = None,
    k: int = 5,
    system_version: str = "",
) -> RetrievalEvaluationResult:
    """Run retrieval evaluation and optionally write reports."""
    dataset = load_retrieval_dataset(dataset_path)
    settings = Settings.from_env()

    with ExitStack() as resources:
        systems, vector_store, bm25_store = _build_retrieval_systems(settings)
        # Keep stores alive during evaluation
        resources.enter_context(vector_store)
        resources.enter_context(bm25_store)

        result = evaluate_systems(systems, dataset, k=k, system_version=system_version)

    # Write reports
    if output_jsonl:
        write_jsonl(generate_retrieval_jsonl(result), output_jsonl)
    if output_markdown:
        with open(output_markdown, "w", encoding="utf-8") as f:
            f.write(generate_retrieval_markdown(result))

    # Print summary to stdout
    print(json.dumps({
        "evaluation_set_version": result.evaluation_set_version,
        "corpus_version": result.corpus_version,
        "system_version": result.system_version,
        "top_k": result.top_k,
        "timestamp_utc": result.timestamp_utc,
        "systems": [
            {
                "system_name": sys.system_name,
                "query_count": sys.query_count,
                "hit_rate_at_k": sys.hit_rate_at_k,
                "recall_at_k": sys.recall_at_k,
            }
            for sys in result.systems
        ],
    }, ensure_ascii=False, indent=2))

    return result


def run_multilingual_evaluation(
    base_path: str,
    output_jsonl: str | None = None,
    output_markdown: str | None = None,
    k: int = 5,
    system_version: str = "",
    language_filter: str = "all",
) -> Any:
    """Run multilingual retrieval evaluation."""
    from kisansathi.evaluation.multilingual import (
        load_available_retrieval_datasets,
        run_multilingual_retrieval_evaluation,
        print_multilingual_summary,
    )
    from kisansathi.evaluation.report import generate_multilingual_jsonl, generate_multilingual_markdown, write_jsonl
    from kisansathi.evaluation.schemas import Language

    settings = Settings.from_env()

    with ExitStack() as resources:
        systems, vector_store, bm25_store = _build_retrieval_systems(settings)
        resources.enter_context(vector_store)
        resources.enter_context(bm25_store)

        datasets = load_available_retrieval_datasets(base_path)
        if not datasets:
            print("No retrieval datasets found.", file=sys.stderr)
            return None

        # Apply language filter
        if language_filter != "all":
            try:
                target_lang = Language(language_filter)
                datasets = {lang: ds for lang, ds in datasets.items() if lang == target_lang}
                if not datasets:
                    print(f"No dataset found for language: {language_filter}", file=sys.stderr)
                    return None
            except ValueError:
                print(f"Invalid language: {language_filter}", file=sys.stderr)
                return None

        report = run_multilingual_retrieval_evaluation(systems, datasets, k=k, system_version=system_version)

    # Write reports
    if output_jsonl:
        write_jsonl(generate_multilingual_jsonl(report), output_jsonl)
    if output_markdown:
        with open(output_markdown, "w", encoding="utf-8") as f:
            f.write(generate_multilingual_markdown(report))

    # Print summary
    print(print_multilingual_summary(report))
    return report


def run_answer_quality_evaluation(
    dataset_path: str,
    output_jsonl: str | None = None,
    output_markdown: str | None = None,
    system_version: str = "",
    judge_provider: str = "fake",
    judge_model: str = "fake-model",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> AnswerQualityEvaluationSummary:
    """Run semantic answer quality evaluation (LLM judge) and optionally write reports.

    This is an OPT-IN command that requires an LLM judge.
    By default, it uses a FakeLLMJudge for testing without API calls.
    To use a real judge, implement the LLMJudge protocol and pass it programmatically.

    Args:
        dataset_path: Path to answer quality evaluation dataset
        output_jsonl: Optional path to write JSONL results
        output_markdown: Optional path to write Markdown report
        system_version: System version identifier
        judge_provider: Judge provider name (default: "fake" for testing)
        judge_model: Judge model name (default: "fake-model")
        temperature: Judge temperature (default: 0.0)
        max_tokens: Judge max tokens (default: 1024)

    Returns:
        AnswerQualityEvaluationSummary with all metric results
    """
    dataset = load_answer_quality_dataset(dataset_path)

    # Use fake judge by default - real judge implementations should be injected programmatically
    if judge_provider == "fake":
        judge = FakeLLMJudge()
        print("NOTICE: Using FakeLLMJudge for testing. No real LLM calls made.", file=sys.stderr)
        print("To use a real judge, implement LLMJudge protocol and call evaluate_answer_quality_dataset programmatically.", file=sys.stderr)
    else:
        raise ValueError(
            f"Unknown judge provider: {judge_provider}. "
            "Real judge providers must be implemented programmatically via the LLMJudge protocol. "
            "This CLI only supports 'fake' for deterministic testing."
        )

    config = JudgeConfig(
        provider_name=judge_provider,
        model_name=judge_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Run evaluation
    summary = asyncio.run(evaluate_answer_quality_dataset(
        dataset=dataset,
        judge=judge,
        config=config,
        system_version=system_version,
    ))

    # Write reports
    if output_jsonl:
        write_jsonl(generate_answer_quality_jsonl(summary), output_jsonl)
    if output_markdown:
        with open(output_markdown, "w", encoding="utf-8") as f:
            f.write(generate_answer_quality_markdown(summary))

    # Print summary to stdout
    print(json.dumps({
        "benchmark_version": summary.benchmark_version,
        "corpus_version": summary.corpus_version,
        "system_version": summary.system_version,
        "total_cases": summary.total_cases,
        "metrics_summary": summary.metrics_summary,
        "judge_metadata": summary.judge_metadata.to_dict(),
        "timestamp_utc": summary.timestamp_utc,
    }, ensure_ascii=False, indent=2))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="KisanSaathi Evaluation Framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Retrieval evaluation
    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval evaluation")
    retrieval_parser.add_argument(
        "--dataset",
        default="data/evaluation/retrieval_en_v1.json",
        help="Path to retrieval evaluation dataset",
    )
    retrieval_parser.add_argument(
        "--output-jsonl",
        help="Path to write JSONL results",
    )
    retrieval_parser.add_argument(
        "--output-markdown",
        help="Path to write Markdown report",
    )
    retrieval_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K for evaluation",
    )
    retrieval_parser.add_argument(
        "--system-version",
        default="",
        help="System version identifier",
    )

    # Multilingual evaluation
    multi_parser = subparsers.add_parser("multilingual", help="Run multilingual retrieval evaluation")
    multi_parser.add_argument(
        "--base-path",
        default="data/evaluation/retrieval",
        help="Base path for retrieval datasets",
    )
    multi_parser.add_argument(
        "--output-jsonl",
        help="Path to write JSONL results",
    )
    multi_parser.add_argument(
        "--output-markdown",
        help="Path to write Markdown report",
    )
    multi_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K for evaluation",
    )
    multi_parser.add_argument(
        "--system-version",
        default="",
        help="System version identifier",
    )
    multi_parser.add_argument(
        "--language",
        choices=["en", "hi", "kn", "te", "all"],
        default="all",
        help="Filter evaluation to specific language (default: all)",
    )

    # Answer evaluation (deterministic)
    answer_parser = subparsers.add_parser("answer", help="Run deterministic answer quality evaluation")
    answer_parser.add_argument(
        "--dataset",
        help="Path to answer evaluation dataset",
    )
    answer_parser.add_argument(
        "--output-jsonl",
        help="Path to write JSONL results",
    )
    answer_parser.add_argument(
        "--output-markdown",
        help="Path to write Markdown report",
    )
    answer_parser.add_argument(
        "--system-version",
        default="",
        help="System version identifier",
    )
    answer_parser.add_argument(
        "--note",
        default="Requires running the full pipeline to generate responses. Not implemented in this CLI.",
        help="Note about this command",
    )

    # Safety/refusal evaluation
    safety_parser = subparsers.add_parser("safety", help="Run safety/refusal evaluation")
    safety_parser.add_argument(
        "--dataset",
        help="Path to safety evaluation dataset",
    )
    safety_parser.add_argument(
        "--output-jsonl",
        help="Path to write JSONL results",
    )
    safety_parser.add_argument(
        "--output-markdown",
        help="Path to write Markdown report",
    )
    safety_parser.add_argument(
        "--system-version",
        default="",
        help="System version identifier",
    )
    safety_parser.add_argument(
        "--note",
        default="Requires running the full pipeline to generate responses. Not implemented in this CLI.",
        help="Note about this command",
    )

    # Citation evaluation
    citation_parser = subparsers.add_parser("citation", help="Run citation evaluation")
    citation_parser.add_argument(
        "--dataset",
        help="Path to citation evaluation dataset (or use --responses)",
    )
    citation_parser.add_argument(
        "--output-jsonl",
        help="Path to write JSONL results",
    )
    citation_parser.add_argument(
        "--output-markdown",
        help="Path to write Markdown report",
    )
    citation_parser.add_argument(
        "--system-version",
        default="",
        help="System version identifier",
    )
    citation_parser.add_argument(
        "--note",
        default="Requires running the full pipeline to generate responses with citations. Not implemented in this CLI.",
        help="Note about this command",
    )

    # Answer quality evaluation (semantic/LLM-judge) - OPT-IN
    answer_quality_parser = subparsers.add_parser(
        "answer-quality",
        help="Run semantic answer quality evaluation (LLM judge) - OPT-IN, requires judge provider"
    )
    answer_quality_parser.add_argument(
        "--dataset",
        default="data/evaluation/answer_quality/answer_quality_en_v1.json",
        help="Path to answer quality evaluation dataset",
    )
    answer_quality_parser.add_argument(
        "--output-jsonl",
        help="Path to write JSONL results",
    )
    answer_quality_parser.add_argument(
        "--output-markdown",
        help="Path to write Markdown report",
    )
    answer_quality_parser.add_argument(
        "--system-version",
        default="",
        help="System version identifier",
    )
    answer_quality_parser.add_argument(
        "--judge-provider",
        default="fake",
        choices=["fake"],
        help="Judge provider (only 'fake' supported in CLI; real providers via programmatic API)",
    )
    answer_quality_parser.add_argument(
        "--judge-model",
        default="fake-model",
        help="Judge model name",
    )
    answer_quality_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature",
    )
    answer_quality_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Judge max tokens",
    )
    answer_quality_parser.add_argument(
        "--note",
        default="OPT-IN: This command runs LLM-judge semantic evaluation. Default uses FakeLLMJudge (no API calls). Real judges require programmatic implementation.",
        help="Note about this command",
    )

    args = parser.parse_args()

    try:
        if args.command == "retrieval":
            run_retrieval_evaluation(
                dataset_path=args.dataset,
                output_jsonl=args.output_jsonl,
                output_markdown=args.output_markdown,
                k=args.k,
                system_version=args.system_version,
            )
            return 0

        elif args.command == "multilingual":
            run_multilingual_evaluation(
                base_path=args.base_path,
                output_jsonl=args.output_jsonl,
                output_markdown=args.output_markdown,
                k=args.k,
                system_version=args.system_version,
                language_filter=args.language,
            )
            return 0

        elif args.command == "answer-quality":
            run_answer_quality_evaluation(
                dataset_path=args.dataset,
                output_jsonl=args.output_jsonl,
                output_markdown=args.output_markdown,
                system_version=args.system_version,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return 0

        elif args.command in ("answer", "safety", "citation"):
            print(f"Command '{args.command}' requires running the full orchestration pipeline to generate responses.")
            print("This is not implemented as a standalone CLI command.")
            print("Use the evaluation framework programmatically in tests or notebooks.")
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())