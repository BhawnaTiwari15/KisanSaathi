"""Evaluation report generation: machine-readable JSONL and human-readable Markdown."""

from dataclasses import asdict
from datetime import datetime
from typing import Any
import json

from kisansathi.evaluation.retrieval import RetrievalEvaluationResult
from kisansathi.evaluation.multilingual import MultilingualReport
from kisansathi.evaluation.answer import AnswerEvaluationSummary
from kisansathi.evaluation.refusal import RefusalEvaluationSummary
from kisansathi.evaluation.citation import CitationEvaluationSummary


def generate_retrieval_jsonl(result: RetrievalEvaluationResult) -> list[str]:
    """Generate JSONL lines for retrieval evaluation."""
    lines = []
    for sys in result.systems:
        lines.append(json_dumps({
            "benchmark_version": result.evaluation_set_version,
            "corpus_version": result.corpus_version,
            "system_version": result.system_version,
            "metric": "hit_rate_at_k",
            "language": "unknown",  # Would be set from dataset
            "system": sys.system_name,
            "value": sys.hit_rate_at_k,
            "k": result.top_k,
            "timestamp_utc": result.timestamp_utc,
            "query_count": sys.query_count,
        }))
        lines.append(json_dumps({
            "benchmark_version": result.evaluation_set_version,
            "corpus_version": result.corpus_version,
            "system_version": result.system_version,
            "metric": "recall_at_k",
            "language": "unknown",
            "system": sys.system_name,
            "value": sys.recall_at_k,
            "k": result.top_k,
            "timestamp_utc": result.timestamp_utc,
            "query_count": sys.query_count,
        }))
    return lines


def generate_answer_jsonl(summary: AnswerEvaluationSummary) -> list[str]:
    """Generate JSONL lines for answer evaluation."""
    lines = []
    for check_name, rate in summary.per_check_pass_rates.items():
        lines.append(json_dumps({
            "benchmark_version": summary.evaluation_set_version,
            "corpus_version": summary.corpus_version,
            "system_version": summary.system_version,
            "metric": f"answer_{check_name}_pass_rate",
            "language": summary.language.value,
            "value": rate,
            "timestamp_utc": summary.timestamp_utc,
            "total_examples": summary.total_examples,
        }))
    lines.append(json_dumps({
        "benchmark_version": summary.evaluation_set_version,
        "corpus_version": summary.corpus_version,
        "system_version": summary.system_version,
        "metric": "answer_overall_pass_rate",
        "language": summary.language.value,
        "value": summary.pass_rate,
        "timestamp_utc": summary.timestamp_utc,
        "total_examples": summary.total_examples,
    }))
    return lines


def generate_refusal_jsonl(summary: RefusalEvaluationSummary) -> list[str]:
    """Generate JSONL lines for refusal evaluation."""
    lines = []
    lines.append(json_dumps({
        "benchmark_version": summary.evaluation_set_version,
        "corpus_version": "",
        "system_version": summary.system_version,
        "metric": "correct_answer_rate",
        "language": "mixed",
        "value": summary.correct_answer_rate,
        "timestamp_utc": summary.timestamp_utc,
        "total_examples": summary.total_examples,
    }))
    lines.append(json_dumps({
        "benchmark_version": summary.evaluation_set_version,
        "corpus_version": "",
        "system_version": summary.system_version,
        "metric": "correct_refusal_rate",
        "language": "mixed",
        "value": summary.correct_refusal_rate,
        "timestamp_utc": summary.timestamp_utc,
        "total_examples": summary.total_examples,
    }))
    lines.append(json_dumps({
        "benchmark_version": summary.evaluation_set_version,
        "corpus_version": "",
        "system_version": summary.system_version,
        "metric": "false_refusal_rate",
        "language": "mixed",
        "value": summary.false_refusal_rate,
        "timestamp_utc": summary.timestamp_utc,
        "total_examples": summary.total_examples,
    }))
    lines.append(json_dumps({
        "benchmark_version": summary.evaluation_set_version,
        "corpus_version": "",
        "system_version": summary.system_version,
        "metric": "unsafe_answer_rate",
        "language": "mixed",
        "value": summary.unsafe_answer_rate,
        "timestamp_utc": summary.timestamp_utc,
        "total_examples": summary.total_examples,
    }))
    lines.append(json_dumps({
        "benchmark_version": summary.evaluation_set_version,
        "corpus_version": "",
        "system_version": summary.system_version,
        "metric": "safety_score",
        "language": "mixed",
        "value": summary.safety_score,
        "timestamp_utc": summary.timestamp_utc,
        "total_examples": summary.total_examples,
    }))
    return lines


def generate_citation_jsonl(summary: CitationEvaluationSummary) -> list[str]:
    """Generate JSONL lines for citation evaluation."""
    lines = []
    for check_name, rate in summary.per_check_pass_rates.items():
        lines.append(json_dumps({
            "benchmark_version": "citation",
            "corpus_version": summary.corpus_version,
            "system_version": summary.system_version,
            "metric": f"citation_{check_name}_pass_rate",
            "language": "mixed",
            "value": rate,
            "timestamp_utc": summary.timestamp_utc,
            "total_responses": summary.total_responses,
        }))
    return lines


def generate_multilingual_jsonl(report: MultilingualReport) -> list[str]:
    """Generate JSONL lines for multilingual report."""
    lines = []
    for lm in report.languages:
        for sys_name, metrics in lm.systems.items():
            for metric_name, value in metrics.items():
                lines.append(json_dumps({
                    "benchmark_version": lm.dataset_version,
                    "corpus_version": lm.corpus_version,
                    "system_version": report.system_version,
                    "metric": metric_name,
                    "language": lm.language.value,
                    "system": sys_name,
                    "value": value,
                    "k": report.top_k,
                    "timestamp_utc": report.evaluation_timestamp_utc,
                    "query_count": lm.query_count,
                }))
    return lines


def write_jsonl(lines: list[str], output_path: str) -> None:
    """Write JSONL lines to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def json_dumps(obj: Any) -> str:
    """JSON dumps with consistent settings."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def generate_retrieval_markdown(result: RetrievalEvaluationResult) -> str:
    """Generate human-readable Markdown report for retrieval evaluation."""
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"**Evaluation Set:** {result.evaluation_set_version}",
        f"**Corpus Version:** {result.corpus_version}",
        f"**System Version:** {result.system_version or 'unknown'}",
        f"**Top-K:** {result.top_k}",
        f"**Timestamp (UTC):** {result.timestamp_utc}",
        "",
        "## Summary",
        "",
        "| System | Queries | Hit@K | Recall@K |",
        "|--------|---------|-------|----------|",
    ]

    for sys in result.systems:
        lines.append(f"| {sys.system_name} | {sys.query_count} | {sys.hit_rate_at_k:.3f} | {sys.recall_at_k:.3f} |")

    lines.extend([
        "",
        "## Per-Query Details",
        "",
    ])

    for sys in result.systems:
        lines.append(f"### {sys.system_name}")
        lines.append("")
        lines.append("| Query | Hit | Recall | Matched | Missed | Extra |")
        lines.append("|-------|-----|--------|---------|--------|-------|")
        for q in sys.per_query:
            query_short = q.query[:60] + ("..." if len(q.query) > 60 else "")
            lines.append(f"| {query_short} | {q.hit_at_k} | {q.recall_at_k:.3f} | "
                         f"{len(q.matched_chunk_ids)} | {len(q.missed_chunk_ids)} | {len(q.extra_chunk_ids)} |")
        lines.append("")

    return "\n".join(lines)


def generate_answer_markdown(summary: AnswerEvaluationSummary) -> str:
    """Generate human-readable Markdown report for answer evaluation."""
    lines = [
        "# Answer Generation Evaluation Report",
        "",
        f"**Evaluation Set:** {summary.evaluation_set_version}",
        f"**Corpus Version:** {summary.corpus_version}",
        f"**System Version:** {summary.system_version or 'unknown'}",
        f"**Language:** {summary.language.value}",
        f"**Timestamp (UTC):** {summary.timestamp_utc}",
        "",
        "## Summary",
        "",
        f"- **Total Examples:** {summary.total_examples}",
        f"- **Passed:** {summary.passed_examples}",
        f"- **Failed:** {summary.failed_examples}",
        f"- **Overall Pass Rate:** {summary.pass_rate:.1%}",
        "",
        "## Per-Check Pass Rates",
        "",
        "| Check | Pass Rate |",
        "|-------|-----------|",
    ]

    for check_name, rate in sorted(summary.per_check_pass_rates.items()):
        lines.append(f"| {check_name} | {rate:.1%} |")

    lines.extend([
        "",
        "## Failed Examples",
        "",
    ])

    for ex in summary.per_example:
        if not ex.overall_passed:
            lines.append(f"### {ex.example_id}: {ex.query[:80]}")
            lines.append("")
            for check in ex.failed_checks:
                lines.append(f"- **{check.check_name}**: {check.details}")
                lines.append(f"  - Expected: {check.expected}")
                lines.append(f"  - Actual: {check.actual}")
            lines.append("")

    return "\n".join(lines)


def generate_refusal_markdown(summary: RefusalEvaluationSummary) -> str:
    """Generate human-readable Markdown report for refusal evaluation."""
    lines = [
        "# Safety/Refusal Evaluation Report",
        "",
        f"**Evaluation Set:** {summary.evaluation_set_version}",
        f"**System Version:** {summary.system_version or 'unknown'}",
        f"**Timestamp (UTC):** {summary.timestamp_utc}",
        "",
        "## Summary",
        "",
        f"- **Total Examples:** {summary.total_examples}",
        f"- **Correct Answers:** {summary.correct_answers}",
        f"- **Correct Clarifications:** {summary.correct_clarifications}",
        f"- **Correct Abstentions:** {summary.correct_abstentions}",
        f"- **False Refusals:** {summary.false_refusals}",
        f"- **Unsafe Answers:** {summary.unsafe_answers}",
        f"- **Correct Answer Rate:** {summary.correct_answer_rate:.1%}",
        f"- **Correct Refusal Rate:** {summary.correct_refusal_rate:.1%}",
        f"- **False Refusal Rate:** {summary.false_refusal_rate:.1%}",
        f"- **Unsafe Answer Rate:** {summary.unsafe_answer_rate:.1%}",
        f"- **Safety Score:** {summary.safety_score:.1%}",
        "",
        "## Per-Example Results",
        "",
        "| Example | Query | Expected | Actual | Category | Correct |",
        "|---------|-------|----------|--------|----------|---------|",
    ]

    for ex in summary.per_example:
        query_short = ex.query[:50] + ("..." if len(ex.query) > 50 else "")
        expected = ex.expected_status.value
        if ex.expected_refusal_type:
            expected += f" ({ex.expected_refusal_type.value})"
        actual = ex.actual_status.value
        if ex.actual_refusal_type:
            actual += f" ({ex.actual_refusal_type})"
        correct = "✓" if ex.correct else "✗"
        lines.append(f"| {ex.example_id} | {query_short} | {expected} | {actual} | {ex.category} | {correct} |")

    return "\n".join(lines)


def generate_citation_markdown(summary: CitationEvaluationSummary) -> str:
    """Generate human-readable Markdown report for citation evaluation."""
    lines = [
        "# Citation Evaluation Report",
        "",
        f"**Corpus Version:** {summary.corpus_version}",
        f"**System Version:** {summary.system_version or 'unknown'}",
        f"**Timestamp (UTC):** {summary.timestamp_utc}",
        "",
        "## Summary",
        "",
        f"- **Total Responses:** {summary.total_responses}",
        f"- **Responses with Citations:** {summary.responses_with_citations}",
        f"- **Citation Validity Rate:** {summary.citation_validity_rate:.1%}",
        f"- **Citation Coverage Rate:** {summary.citation_coverage_rate:.1%}",
        f"- **Provenance Validity Rate:** {summary.provenance_validity_rate:.1%}",
        "",
        "## Per-Check Pass Rates",
        "",
        "| Check | Pass Rate |",
        "|-------|-----------|",
    ]

    for check_name, rate in sorted(summary.per_check_pass_rates.items()):
        lines.append(f"| {check_name} | {rate:.1%} |")

    return "\n".join(lines)


def generate_multilingual_markdown(report: MultilingualReport) -> str:
    """Generate human-readable Markdown report for multilingual evaluation."""
    from kisansathi.evaluation.multilingual import print_multilingual_summary
    return print_multilingual_summary(report)