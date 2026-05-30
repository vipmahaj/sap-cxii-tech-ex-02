"""SQLite connection helpers.

Two flavours of connection:
- read_write(): used by ETL to create/upsert rows.
- read_only():  used by /orders/ask so even an LLM-emitted DROP cannot mutate.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager


SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS orders (
  order_id    TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL,
  order_date  TEXT NOT NULL,
  amount_usd  REAL NOT NULL,
  currency    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date     ON orders(order_date);
"""


def ensure_schema(db_path: str) -> None:
    """Idempotent. Create table + indexes if missing."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_DDL)


@contextmanager
def read_write(db_path: str) -> Iterator[sqlite3.Connection]:
    """Read-write connection used by the ETL loader."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def read_only(db_path: str, timeout_seconds: float = 5.0) -> Iterator[sqlite3.Connection]:
    """Read-only connection used by /orders/ask and deterministic endpoints.

    Uses the file: URI with mode=ro so any DML raises sqlite3.OperationalError.
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
