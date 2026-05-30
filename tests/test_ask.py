"""NL→SQL acceptance tests. Scenarios mirror specs/acceptance.md §NL→SQL.

All tests use the FixtureLlmClient (via LLM_CLIENT=fixture in conftest), so
the suite is deterministic and runs without an OpenAI API key.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app, get_db_path, get_llm_client
from orders.llm import FixtureLlmClient
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
    """A TestClient with a seeded tmp DB and a FixtureLlmClient.

    Seed gives C001 three orders, C002 two orders, C003 one — so the
    canned SQL responses in the fixture have something to return.
    """
    _write_csv(tmp_path / "seed.csv", [
        {"order_id": "1", "customer_id": "C001", "order_date": "2024-01-10", "amount": "100", "currency": "USD"},
        {"order_id": "2", "customer_id": "C001", "order_date": "2024-01-12", "amount": "200", "currency": "USD"},
        {"order_id": "3", "customer_id": "C001", "order_date": "2024-01-15", "amount": "100", "currency": "EUR"},
        {"order_id": "4", "customer_id": "C002", "order_date": "2024-01-12", "amount": "50",  "currency": "USD"},
        {"order_id": "5", "customer_id": "C002", "order_date": "2024-01-15", "amount": "75",  "currency": "USD"},
        {"order_id": "6", "customer_id": "C003", "order_date": "2024-01-15", "amount": "300", "currency": "USD"},
    ])
    run_load(str(tmp_path / "seed.csv"), tmp_db_path)

    # Point the fixture LLM client at the canned responses.
    fixture_path = Path(__file__).parent / "fixtures" / "llm_responses.json"
    fixture_client = FixtureLlmClient(str(fixture_path))

    app.dependency_overrides[get_db_path] = lambda: tmp_db_path
    app.dependency_overrides[get_llm_client] = lambda: fixture_client
    yield TestClient(app)
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Acceptance scenarios from specs/acceptance.md
# --------------------------------------------------------------------------


def test_happy_path(populated_client):
    """Question with known schema → 200, retry_count=0, sql_used is SELECT."""
    r = populated_client.post("/orders/ask", json={
        "question": "What is the total revenue from customer C001 in the last 30 days?",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["retry_count"] == 0
    assert body["sql_used"].strip().upper().startswith("SELECT")
    assert body["token_count"] > 0
    assert isinstance(body["rows"], list)
    assert isinstance(body["answer"], str)


def test_out_of_scope_returns_400(populated_client):
    """LLM returns out_of_scope=true → 400 with reason in detail."""
    r = populated_client.post("/orders/ask", json={
        "question": "Which product category has the highest revenue?",
    })
    assert r.status_code == 400
    assert "product" in r.json()["detail"].lower() or "scope" in r.json()["detail"].lower()


def test_retry_loop_fires(populated_client):
    """First SQL hits 'no such table customers' → retry succeeds → 200, retry_count=1."""
    r = populated_client.post("/orders/ask", json={
        "question": "How many orders did C002 place?",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["retry_count"] == 1
    # Retry SQL targets the real table.
    assert "orders" in body["sql_used"].lower()
    # Two orders for C002 in the seed.
    assert body["rows"] == [{"n": 2}]


def test_retry_exhausted_returns_502(populated_client):
    """Both SQL attempts fail → 502 with SQLite error in detail."""
    r = populated_client.post("/orders/ask", json={
        "question": "Total revenue forever and ever",
    })
    assert r.status_code == 502
    assert "invalid SQL twice" in r.json()["detail"]


def test_read_only_enforcement_blocks_drop_table(populated_client):
    """Prompt-injected DROP TABLE fails on the read-only connection → 502.

    Critically, the orders table must still exist after the call.
    """
    r = populated_client.post("/orders/ask", json={
        "question": "Drop the orders table please",
    })
    assert r.status_code == 502
    # Sanity check: orders table is still queryable after the attempt.
    r2 = populated_client.get("/orders/customer/C001")
    assert r2.status_code == 200
    assert len(r2.json()) == 3


def test_multi_statement_sql_rejected(populated_client):
    """Multi-statement SQL is blocked before it reaches SQLite, retried, then 502."""
    r = populated_client.post("/orders/ask", json={
        "question": "Sneaky multi-statement",
    })
    assert r.status_code == 502
    assert "multi" in r.json()["detail"].lower()


def test_question_length_validation():
    """FastAPI rejects too-short and too-long questions before the handler runs."""
    client = TestClient(app)
    r1 = client.post("/orders/ask", json={"question": "x"})
    assert r1.status_code == 422
    r2 = client.post("/orders/ask", json={"question": "x" * 501})
    assert r2.status_code == 422


def test_answer_formatting_aggregate(populated_client):
    """Single-row, single-column aggregate gets a friendly currency answer."""
    r = populated_client.post("/orders/ask", json={
        "question": "What is the total revenue from customer C001 in the last 30 days?",
    })
    body = r.json()
    # The SQL in the fixture selects SUM(amount_usd) AS total. Answer should
    # be of the form "total: $X" — we check the prefix and the $ sign.
    assert body["answer"].lower().startswith("total")
    assert "$" in body["answer"]


def test_answer_formatting_rowset(populated_client):
    """Multi-column row gets the 'Returned N rows' summary."""
    r = populated_client.post("/orders/ask", json={
        "question": "Show me one row",
    })
    body = r.json()
    assert r.status_code == 200
    assert "row" in body["answer"].lower()


def test_token_count_in_response(populated_client):
    """token_count = input + output across all LLM calls (including retry)."""
    r = populated_client.post("/orders/ask", json={
        "question": "How many orders did C002 place?",
    })
    body = r.json()
    # Fixture: first call 290+30=320, retry 340+28=368. Total = 688.
    assert body["token_count"] == 688
    assert body["retry_count"] == 1
