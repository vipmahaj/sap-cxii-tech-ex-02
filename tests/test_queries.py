"""Unit tests for orders.queries — runs against a tmp SQLite DB, no FastAPI."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from orders import queries
from orders.load import run_load


CSV_HEADER = ["order_id", "customer_id", "order_date", "amount", "currency"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def populated_db(tmp_path, tmp_db_path) -> str:
    """Three customers across three days, mix of USD and EUR."""
    _write_csv(tmp_path / "seed.csv", [
        # C001 — three orders across two days
        {"order_id": "1", "customer_id": "C001", "order_date": "2024-01-10", "amount": "100", "currency": "USD"},
        {"order_id": "2", "customer_id": "C001", "order_date": "2024-01-10", "amount": "200", "currency": "USD"},
        {"order_id": "3", "customer_id": "C001", "order_date": "2024-01-15", "amount": "100", "currency": "EUR"},  # → 110 USD
        # C002 — single order
        {"order_id": "4", "customer_id": "C002", "order_date": "2024-01-12", "amount": "50",  "currency": "USD"},
        # C003 — single order
        {"order_id": "5", "customer_id": "C003", "order_date": "2024-01-15", "amount": "300", "currency": "USD"},
    ])
    run_load(str(tmp_path / "seed.csv"), tmp_db_path)
    return tmp_db_path


# --------------------------------------------------------------------------
# get_orders_by_customer
# --------------------------------------------------------------------------


def test_by_customer_returns_their_orders(populated_db):
    rows = queries.get_orders_by_customer(populated_db, "C001")
    assert len(rows) == 3
    assert {r["order_id"] for r in rows} == {"1", "2", "3"}


def test_by_customer_unknown_returns_empty(populated_db):
    assert queries.get_orders_by_customer(populated_db, "NOPE") == []


def test_by_customer_sorted_newest_first(populated_db):
    rows = queries.get_orders_by_customer(populated_db, "C001")
    assert [r["order_date"] for r in rows] == ["2024-01-15", "2024-01-10", "2024-01-10"]


# --------------------------------------------------------------------------
# get_stats
# --------------------------------------------------------------------------


def test_stats_total_and_avg(populated_db):
    stats = queries.get_stats(populated_db)
    # 100 + 200 + 110 (EUR → USD) + 50 + 300 = 760
    assert stats["total_revenue"] == pytest.approx(760.0)
    assert stats["avg_order_value"] == pytest.approx(152.0)  # 760 / 5


def test_stats_per_day_buckets(populated_db):
    stats = queries.get_stats(populated_db)
    assert stats["orders_per_day"] == {
        "2024-01-10": 2,
        "2024-01-12": 1,
        "2024-01-15": 2,
    }


def test_stats_empty_db_is_zero(tmp_db_path):
    """Empty DB → zeros, not divide-by-zero."""
    from orders import db
    db.ensure_schema(tmp_db_path)
    stats = queries.get_stats(tmp_db_path)
    assert stats == {"total_revenue": 0.0, "avg_order_value": 0.0, "orders_per_day": {}}


# --------------------------------------------------------------------------
# get_recent
# --------------------------------------------------------------------------


def test_recent_filters_by_cutoff(populated_db):
    """With now=2024-01-16 and days=5, cutoff is 2024-01-11 — excludes 2024-01-10 rows."""
    rows = queries.get_recent(populated_db, days=5, now_override=date(2024, 1, 16))
    assert {r["order_date"] for r in rows} == {"2024-01-15", "2024-01-12"}
    assert len(rows) == 3


def test_recent_includes_today(populated_db):
    rows = queries.get_recent(populated_db, days=1, now_override=date(2024, 1, 15))
    assert all(r["order_date"] == "2024-01-15" for r in rows)


def test_recent_outside_window_is_empty(populated_db):
    rows = queries.get_recent(populated_db, days=1, now_override=date(2030, 1, 1))
    assert rows == []
