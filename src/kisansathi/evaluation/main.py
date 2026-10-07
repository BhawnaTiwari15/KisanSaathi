"""Evaluation command-line interface."""

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from kisansathi.config import Settings
from kisansathi.evaluation.retrieval import RetrievalEvaluationResult, evaluate_systems
from kisansathi.evaluation.schemas import load_retrieval_dataset
from kisansathi.evaluation.report import (
    generate_retrieval_jsonl,
    generate_retrieval_markdown,
    write_jsonl,
)
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
) -> Any:
    """Run multilingual retrieval evaluation."""
    from kisansathi.evaluation.multilingual import (
        load_available_retrieval_datasets,
        run_multilingual_retrieval_evaluation,
        print_multilingual_summary,
    )
    from kisansathi.evaluation.report import generate_multilingual_jsonl, generate_multilingual_markdown, write_jsonl

    settings = Settings.from_env()

    with ExitStack() as resources:
        systems, vector_store, bm25_store = _build_retrieval_systems(settings)
        resources.enter_context(vector_store)
        resources.enter_context(bm25_store)

        datasets = load_available_retrieval_datasets(base_path)
        if not datasets:
            print("No retrieval datasets found.", file=sys.stderr)
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