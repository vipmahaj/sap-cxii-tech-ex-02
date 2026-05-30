"""ETL CLI entrypoint.

Usage:
    python etl.py load data/orders.csv

The `load` subcommand reads the CSV, applies the transformations defined in
orders.etl_core, upserts into SQLite, and rebuilds the FAISS index. The
load summary is printed to stdout.

This file is intentionally a thin argparse shell. The orchestration lives
in orders/load.py so it's importable from tests without argparse plumbing.
"""

from __future__ import annotations

import argparse
import sys

from orders import config
from orders.load import run_load
from orders.logging_setup import configure as configure_logging


def cmd_load(args: argparse.Namespace) -> int:
    """`python etl.py load <csv_path>`."""
    settings = config.get_settings()
    summary = run_load(
        csv_path=args.csv,
        db_path=settings.db_path,
        index_path=settings.index_path,
    )
    print(summary.as_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="etl", description="Customer orders ETL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    load_p = sub.add_parser("load", help="Load and clean a CSV into SQLite + rebuild FAISS index")
    load_p.add_argument("csv", help="Path to the input CSV")
    load_p.set_defaults(func=cmd_load)

    return parser


def main(argv: list[str] | None = None) -> int:
    settings = config.get_settings()
    configure_logging(settings.log_level)

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
