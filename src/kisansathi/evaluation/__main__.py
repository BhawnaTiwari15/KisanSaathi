"""Run the versioned local retrieval evaluation set."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json

from kisansathi.config import Settings
from kisansathi.evaluation.recall import (
    RerankedHybridRetriever,
    evaluate_systems,
    load_evaluation_dataset,
)
from kisansathi.logging_setup import configure_logging
from kisansathi.retrieval.bm25_retriever import BM25Retriever
from kisansathi.retrieval.bm25_store import BM25Store
from kisansathi.retrieval.embeddings import EmbeddingService
from kisansathi.retrieval.hybrid_retriever import HybridRetriever
from kisansathi.retrieval.reranker import Reranker
from kisansathi.retrieval.retriever import DenseRetriever
from kisansathi.retrieval.vector_store import QdrantVectorStore


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate retrieval against hand-labeled queries")
    parser.add_argument(
        "--dataset",
        default="data/evaluation/retrieval_en_v1.json",
        help="Path to the versioned hand-labeled evaluation JSON",
    )
    arguments = parser.parse_args()
    dataset = load_evaluation_dataset(arguments.dataset)
    settings = Settings.from_env()

    with ExitStack() as resources:
        vector_store = resources.enter_context(QdrantVectorStore(settings))
        bm25_store = resources.enter_context(BM25Store(settings))
        dense = DenseRetriever(EmbeddingService(settings), vector_store)
        bm25 = BM25Retriever(bm25_store)
        hybrid = HybridRetriever(dense, bm25)
        reranker = Reranker(settings, candidate_depth=10, default_top_k=5)
        systems = {
            "DenseRetriever": dense,
            "BM25Retriever": bm25,
            "HybridRetriever": hybrid,
            "HybridRetriever+Reranker": RerankedHybridRetriever(
                hybrid, reranker, candidate_depth=10
            ),
        }
        summary = evaluate_systems(systems, dataset)

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())