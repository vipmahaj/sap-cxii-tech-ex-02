"""Deterministic Query API integration tests.

Scenarios mirror specs/acceptance.md §Query API (Part 2). The DB path is
injected via FastAPI dependency override so the same TestClient can target
a per-test tmp database.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app, get_db_path, get_today
from orders.load import run_load


CSV_HEADER = ["order_id", "customer_id", "order_date", "amount", "currency"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def populated_client(tmp_path, tmp_db_path) -> TestClient:
    """A TestClient whose get_db_path dependency points at a seeded tmp DB."""
    _write_csv(tmp_path / "seed.csv", [
        {"order_id": "1", "customer_id": "C001", "order_date": "2024-01-10", "amount": "100", "currency": "USD"},
        {"order_id": "2", "customer_id": "C001", "order_date": "2024-01-10", "amount": "200", "currency": "USD"},
        {"order_id": "3", "customer_id": "C001", "order_date": "2024-01-15", "amount": "100", "currency": "EUR"},
        {"order_id": "4", "customer_id": "C002", "order_date": "2024-01-12", "amount": "50",  "currency": "USD"},
        {"order_id": "5", "customer_id": "C003", "order_date": "2024-01-15", "amount": "300", "currency": "USD"},
    ])
    run_load(str(tmp_path / "seed.csv"), tmp_db_path)

    app.dependency_overrides[get_db_path] = lambda: tmp_db_path
    yield TestClient(app)
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# /healthz — already-implemented sanity check
# --------------------------------------------------------------------------


def test_healthz_returns_ok():
    r = TestClient(app).get("/healthz")
    assert r.status_code == 200
    assert r.text == "ok"


# --------------------------------------------------------------------------
# /orders/customer/{customer_id}
# --------------------------------------------------------------------------


def test_customer_lookup_returns_orders(populated_client):
    r = populated_client.get("/orders/customer/C001")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    assert {o["order_id"] for o in body} == {"1", "2", "3"}


def test_unknown_customer_returns_empty_200(populated_client):
    """Per the API contract: unknown customer is NOT a 404 — it's an empty list."""
    r = populated_client.get("/orders/customer/NOPE")
    assert r.status_code == 200
    assert r.json() == []


# --------------------------------------------------------------------------
# /orders/stats
# --------------------------------------------------------------------------


def test_stats_aggregates_correctly(populated_client):
    r = populated_client.get("/orders/stats")
    assert r.status_code == 200
    body = r.json()
    # 100 + 200 + 110 (EUR→USD) + 50 + 300 = 760
    assert body["total_revenue"] == pytest.approx(760.0)
    assert body["avg_order_value"] == pytest.approx(152.0)
    assert body["orders_per_day"] == {
        "2024-01-10": 2,
        "2024-01-12": 1,
        "2024-01-15": 2,
    }


# --------------------------------------------------------------------------
# /orders/recent
# --------------------------------------------------------------------------


def test_recent_filters_by_days(populated_client):
    """Pin `today` via the dependency override so the test isn't time-dependent."""
    app.dependency_overrides[get_today] = lambda: date(2024, 1, 16)
    try:
        r = populated_client.get("/orders/recent?days=5")
    finally:
        # Clean only our override; populated_client teardown clears db_path.
        app.dependency_overrides.pop(get_today, None)

    assert r.status_code == 200
    dates = {o["order_date"] for o in r.json()}
    assert dates == {"2024-01-15", "2024-01-12"}


def test_recent_rejects_negative_days():
    r = TestClient(app).get("/orders/recent?days=-1")
    assert r.status_code == 422


def test_recent_rejects_zero_days():
    r = TestClient(app).get("/orders/recent?days=0")
    assert r.status_code == 422
