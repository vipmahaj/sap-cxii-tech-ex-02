"""ETL integration tests. Scenarios mirror specs/acceptance.md §ETL.

Each test writes a small CSV to tmp_path, calls run_load directly, and
inspects the resulting SQLite DB. No subprocess, no env-var mocking.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from orders.load import run_load


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


CSV_HEADER = ["order_id", "customer_id", "order_date", "amount", "currency"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write a CSV with the standard header and the given rows. Returns path."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_HEADER})
    return path


def _fetch_all(db_path: str) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM orders ORDER BY order_id")]


# --------------------------------------------------------------------------
# Scenarios from specs/acceptance.md
# --------------------------------------------------------------------------


def test_clean_csv_loads_end_to_end(tmp_path, tmp_db_path):
    """Given a clean 10-row CSV, all 10 rows land in orders."""
    rows = [
        {"order_id": f"100{i}", "customer_id": f"C{i:03}",
         "order_date": "2020-01-15", "amount": "100", "currency": "USD"}
        for i in range(10)
    ]
    csv_path = _write_csv(tmp_path / "orders.csv", rows)

    summary = run_load(str(csv_path), tmp_db_path)

    assert summary.read == 10
    assert summary.loaded == 10
    assert len(_fetch_all(tmp_db_path)) == 10


def test_date_normalization_across_formats(tmp_path, tmp_db_path):
    """Mixed date formats all become YYYY-MM-DD."""
    rows = [
        {"order_id": "1", "customer_id": "C1", "order_date": "2020-01-15", "amount": "10", "currency": "USD"},
        {"order_id": "2", "customer_id": "C1", "order_date": "01/15/2020", "amount": "10", "currency": "USD"},
        {"order_id": "3", "customer_id": "C1", "order_date": "15-01-2020", "amount": "10", "currency": "USD"},
        {"order_id": "4", "customer_id": "C1", "order_date": "2020/01/15", "amount": "10", "currency": "USD"},
    ]
    _write_csv(tmp_path / "orders.csv", rows)
    run_load(str(tmp_path / "orders.csv"), tmp_db_path)

    dates = {r["order_date"] for r in _fetch_all(tmp_db_path)}
    assert dates == {"2020-01-15"}


def test_missing_amount_becomes_zero(tmp_path, tmp_db_path):
    """Empty amount field is kept and stored as 0.0."""
    _write_csv(tmp_path / "orders.csv", [
        {"order_id": "1", "customer_id": "C1", "order_date": "2020-01-15", "amount": "", "currency": "USD"},
    ])
    summary = run_load(str(tmp_path / "orders.csv"), tmp_db_path)

    rows = _fetch_all(tmp_db_path)
    assert rows[0]["amount_usd"] == 0.0
    assert summary.amount_fixed_to_zero == 1


def test_missing_currency_defaults_to_usd(tmp_path, tmp_db_path):
    """Empty currency is treated as USD."""
    _write_csv(tmp_path / "orders.csv", [
        {"order_id": "1", "customer_id": "C1", "order_date": "2020-01-15", "amount": "100", "currency": ""},
    ])
    summary = run_load(str(tmp_path / "orders.csv"), tmp_db_path)

    rows = _fetch_all(tmp_db_path)
    assert rows[0]["amount_usd"] == 100.0
    assert rows[0]["currency"] == "USD"
    assert summary.currency_filled_to_usd == 1


def test_eur_converted_at_1_1(tmp_path, tmp_db_path):
    """100 EUR becomes 110.00 USD."""
    _write_csv(tmp_path / "orders.csv", [
        {"order_id": "1", "customer_id": "C1", "order_date": "2020-01-15", "amount": "100", "currency": "EUR"},
    ])
    run_load(str(tmp_path / "orders.csv"), tmp_db_path)

    rows = _fetch_all(tmp_db_path)
    assert rows[0]["amount_usd"] == pytest.approx(110.0)
    assert rows[0]["currency"] == "EUR"


def test_unknown_currency_drops_row(tmp_path, tmp_db_path):
    """100 GBP is dropped, summary increments dropped_unknown_currency."""
    _write_csv(tmp_path / "orders.csv", [
        {"order_id": "1", "customer_id": "C1", "order_date": "2020-01-15", "amount": "100", "currency": "GBP"},
    ])
    summary = run_load(str(tmp_path / "orders.csv"), tmp_db_path)

    assert summary.loaded == 0
    assert summary.dropped_unknown_currency == 1
    assert _fetch_all(tmp_db_path) == []


def test_missing_identifiers_drop_row(tmp_path, tmp_db_path):
    """Rows missing order_id or customer_id are dropped, both counters bump."""
    _write_csv(tmp_path / "orders.csv", [
        {"order_id": "",  "customer_id": "C1", "order_date": "2020-01-15", "amount": "10", "currency": "USD"},
        {"order_id": "2", "customer_id": "",   "order_date": "2020-01-15", "amount": "10", "currency": "USD"},
        {"order_id": "3", "customer_id": "C1", "order_date": "2020-01-15", "amount": "10", "currency": "USD"},
    ])
    summary = run_load(str(tmp_path / "orders.csv"), tmp_db_path)

    assert summary.read == 3
    assert summary.loaded == 1
    assert summary.dropped_no_order_id == 1
    assert summary.dropped_no_customer_id == 1


def test_etl_is_idempotent(tmp_path, tmp_db_path):
    """Loading the same CSV twice produces the same row count."""
    _write_csv(tmp_path / "orders.csv", [
        {"order_id": "1", "customer_id": "C1", "order_date": "2020-01-15", "amount": "10", "currency": "USD"},
        {"order_id": "2", "customer_id": "C1", "order_date": "2020-01-16", "amount": "20", "currency": "EUR"},
    ])

    run_load(str(tmp_path / "orders.csv"), tmp_db_path)
    first = _fetch_all(tmp_db_path)

    run_load(str(tmp_path / "orders.csv"), tmp_db_path)
    second = _fetch_all(tmp_db_path)

    assert first == second
    assert len(second) == 2


# --------------------------------------------------------------------------
# Error path tests (not in acceptance.md but cheap and useful)
# --------------------------------------------------------------------------


def test_missing_csv_raises(tmp_db_path, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_load(str(tmp_path / "does_not_exist.csv"), tmp_db_path)


def test_csv_missing_required_columns_raises(tmp_path, tmp_db_path):
    """A CSV without 'amount' is rejected upfront, not row-by-row."""
    bad = tmp_path / "bad.csv"
    bad.write_text("order_id,customer_id\n1,C1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        run_load(str(bad), tmp_db_path)
