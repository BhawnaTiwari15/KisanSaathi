"""Evaluation framework for KisanSaathi."""

from kisansathi.evaluation.recall import (
    EvaluationDataset,
    EvaluationExample,
    EvaluationSummary,
    RerankedHybridRetriever,
    evaluate_systems,
    load_evaluation_dataset,
)
from kisansathi.evaluation.schemas import (
    Language,
    RetrievalCategory,
    Difficulty,
    AnswerStatus,
    RefusalType,
    RetrievalExample,
    RetrievalDataset,
    AnswerExample,
    AnswerDataset,
    SafetyExample,
    SafetyDataset,
    load_retrieval_dataset,
    load_answer_dataset,
    load_safety_dataset,
)
from kisansathi.evaluation.retrieval import (
    RetrievalEvaluationResult,
    SystemMetrics,
    QueryMetrics,
    evaluate_retriever,
    evaluate_systems as evaluate_retrieval_systems,
)
from kisansathi.evaluation.answer import (
    AnswerQualityResult,
    AnswerQualityCheck,
    AnswerEvaluationSummary,
    evaluate_answer_quality,
    evaluate_answer_dataset,
)
from kisansathi.evaluation.refusal import (
    RefusalResult,
    RefusalEvaluationSummary,
    evaluate_safety_dataset,
)
from kisansathi.evaluation.citation import (
    CitationEvaluationResult,
    CitationCheck,
    CitationEvaluationSummary,
    evaluate_citations,
    evaluate_citation_dataset,
)
from kisansathi.evaluation.multilingual import (
    LanguageMetrics,
    MultilingualReport,
    find_retrieval_datasets,
    load_available_retrieval_datasets,
    run_multilingual_retrieval_evaluation,
    print_multilingual_summary,
)
from kisansathi.evaluation.report import (
    generate_retrieval_jsonl,
    generate_retrieval_markdown,
    generate_answer_jsonl,
    generate_answer_markdown,
    generate_refusal_jsonl,
    generate_refusal_markdown,
    generate_citation_jsonl,
    generate_citation_markdown,
    generate_multilingual_jsonl,
    generate_multilingual_markdown,
    write_jsonl,
)

__all__ = [
    # Recall (legacy)
    "EvaluationDataset",
    "EvaluationExample",
    "EvaluationSummary",
    "RerankedHybridRetriever",
    "evaluate_systems",
    "load_evaluation_dataset",
    # Schemas
    "Language",
    "RetrievalCategory",
    "Difficulty",
    "AnswerStatus",
    "RefusalType",
    "RetrievalExample",
    "RetrievalDataset",
    "AnswerExample",
    "AnswerDataset",
    "SafetyExample",
    "SafetyDataset",
    "load_retrieval_dataset",
    "load_answer_dataset",
    "load_safety_dataset",
    # Retrieval
    "RetrievalEvaluationResult",
    "SystemMetrics",
    "QueryMetrics",
    "evaluate_retriever",
    "evaluate_retrieval_systems",
    # Answer quality
    "AnswerQualityResult",
    "AnswerQualityCheck",
    "AnswerEvaluationSummary",
    "evaluate_answer_quality",
    "evaluate_answer_dataset",
    # Refusal
    "RefusalResult",
    "RefusalEvaluationSummary",
    "evaluate_safety_dataset",
    # Citation
    "CitationEvaluationResult",
    "CitationCheck",
    "CitationEvaluationSummary",
    "evaluate_citations",
    "evaluate_citation_dataset",
    # Multilingual
    "LanguageMetrics",
    "MultilingualReport",
    "find_retrieval_datasets",
    "load_available_retrieval_datasets",
    "run_multilingual_retrieval_evaluation",
    "print_multilingual_summary",
    # Report
    "generate_retrieval_jsonl",
    "generate_retrieval_markdown",
    "generate_answer_jsonl",
    "generate_answer_markdown",
    "generate_refusal_jsonl",
    "generate_refusal_markdown",
    "generate_citation_jsonl",
    "generate_citation_markdown",
    "generate_multilingual_jsonl",
    "generate_multilingual_markdown",
    "write_jsonl",
]