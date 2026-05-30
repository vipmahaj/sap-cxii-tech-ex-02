"""ETL orchestration: read CSV → transform → upsert into SQLite.

This module is the I/O layer for the load pipeline. orders/etl_core.py
stays pure; this is where we touch files, the DB, and (in Step 4) the
FAISS index.

The single entry point is `run_load(csv_path, db_path)`. It's importable
from tests without going through argparse or env vars.
"""

from __future__ import annotations

import csv
from pathlib import Path

from orders import db, search
from orders.etl_core import LoadSummary, transform_row
from orders.logging_setup import get_logger

log = get_logger("etl.load")


UPSERT_SQL = """
INSERT OR REPLACE INTO orders
    (order_id, customer_id, order_date, amount_usd, currency)
VALUES
    (:order_id, :customer_id, :order_date, :amount_usd, :currency)
"""


def run_load(csv_path: str, db_path: str, index_path: str | None = None) -> LoadSummary:
    """Load a CSV into the orders table. Idempotent (upserts on order_id).

    Args:
        csv_path:   Path to the input CSV. Must have the columns listed in
                    data-contract.md §1.
        db_path:    Target SQLite file. Created if missing.
        index_path: FAISS index path. When None, the rebuild step is skipped
                    (used by tests that don't care about embeddings).

    Returns:
        A LoadSummary with per-branch counters.

    Raises:
        FileNotFoundError if csv_path doesn't exist.
        ValueError       if the CSV is missing required columns.
    """
    csv_p = Path(csv_path)
    if not csv_p.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    db.ensure_schema(db_path)

    summary = LoadSummary()
    cleaned: list[dict[str, object]] = []

    with csv_p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _require_columns(reader.fieldnames or [])
        for raw in reader:
            summary.read += 1
            row = transform_row(raw, summary)
            if row is not None:
                cleaned.append(row)

    if cleaned:
        with db.read_write(db_path) as conn:
            conn.executemany(UPSERT_SQL, cleaned)
        summary.loaded = len(cleaned)

    # FAISS rebuild — a real implementation lands in Step 4.
    if index_path is not None:
        search.rebuild_from_db(db_path, index_path)

    log.info("etl_load_done", **_summary_dict(summary))
    return summary


def _require_columns(fieldnames: list[str]) -> None:
    """Validate the CSV header. Missing columns are a hard error."""
    required = {"order_id", "customer_id", "order_date", "amount", "currency"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Got: {fieldnames}"
        )


def _summary_dict(s: LoadSummary) -> dict[str, int]:
    """Flatten LoadSummary into kwargs for the structured logger."""
    return {
        "read": s.read,
        "loaded": s.loaded,
        "dropped_no_order_id": s.dropped_no_order_id,
        "dropped_no_customer_id": s.dropped_no_customer_id,
        "dropped_bad_date": s.dropped_bad_date,
        "dropped_unknown_currency": s.dropped_unknown_currency,
        "amount_fixed_to_zero": s.amount_fixed_to_zero,
        "currency_filled_to_usd": s.currency_filled_to_usd,
    }
