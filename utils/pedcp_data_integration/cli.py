"""Command-line interface for the fixed PedCP ingestion recipe."""

import argparse

from utils.pedcp_data_integration import pipeline
from utils.pedcp_data_integration.recipe import COHORTS
from utils.project_config import load_project_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {
        "download": pipeline.download_metadata,
        "prepare": pipeline.prepare,
        "ingest": pipeline.ingest_prepared,
        "report": pipeline.report,
        "run": pipeline.run,
    }
    parser.add_argument("command", choices=stages, nargs="?", default="run")
    parser.add_argument("--config", help="Project YAML")
    parser.add_argument("--cohort", choices=COHORTS, default="all", help="Collections to process")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.command != "ingest":
        parser.error("--dry-run is only supported for ingest")
    try:
        options = {} if args.command == "download" else {"cohort": args.cohort}
        if args.command == "ingest":
            options["dry_run"] = args.dry_run
        stages[args.command](load_project_config(args.config), **options)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"pedcp {args.command}: {exc}\n")
    return 0
