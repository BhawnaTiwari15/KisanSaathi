"""Command-line entry point for local source ingestion."""

import argparse
from pathlib import Path
import sys

from kisansathi.ingestion.manifest import ManifestError
from kisansathi.ingestion.models import IngestionError
from kisansathi.ingestion.pipeline import ingest_manifest
from kisansathi.logging_setup import configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest reviewed local PDFs from a source manifest")
    parser.add_argument("--manifest", default="data/sources.json")
    parser.add_argument("--root", default=".", help="Project root containing data/incoming")
    arguments = parser.parse_args()

    try:
        report = ingest_manifest(arguments.manifest, arguments.root)
    except IngestionError as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1

    if not report.sources:
        print(f"No sources declared in {Path(arguments.manifest)}; nothing ingested.")
        return 0
    for source in report.sources:
        status = "unchanged" if source.unchanged else "updated" if source.updated else "ingested"
        print(f"{source.source_id}: {status} ({source.sha256}) -> {source.processed_path}")
        for warning in source.warnings:
            print(f"  warning: {warning}")
    for source_ids in report.duplicate_checksums:
        print(f"Duplicate checksum shared by source IDs: {', '.join(source_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())