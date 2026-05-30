"""Unit tests for orders.ask — exercise the orchestrator directly.

These bypass FastAPI so we can lock down the retry logic and the safety
rails without TestClient overhead.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from orders import ask
from orders.llm import FixtureLlmClient
from orders.load import run_load


CSV_HEADER = ["order_id", "customer_id", "order_date", "amount", "currency"]


@pytest.fixture
def populated_db(tmp_path, tmp_db_path) -> str:
    with (tmp_path / "seed.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows([
            {"order_id": "1", "customer_id": "C001", "order_date": "2024-01-10", "amount": "100", "currency": "USD"},
            {"order_id": "2", "customer_id": "C002", "order_date": "2024-01-12", "amount": "50",  "currency": "USD"},
            {"order_id": "3", "customer_id": "C002", "order_date": "2024-01-15", "amount": "75",  "currency": "USD"},
        ])
    run_load(str(tmp_path / "seed.csv"), tmp_db_path)
    return tmp_db_path


@pytest.fixture
def fixture_llm() -> FixtureLlmClient:
    return FixtureLlmClient(str(Path(__file__).parent / "fixtures" / "llm_responses.json"))


# --------------------------------------------------------------------------
# answer_question — happy paths
# --------------------------------------------------------------------------


def test_happy_path_no_retry(populated_db, fixture_llm):
    result = ask.answer_question(
        "What is the total revenue from customer C001 in the last 30 days?",
        fixture_llm,
        populated_db,
    )
    assert result.retry_count == 0
    assert result.token_count == 312 + 48
    assert result.sql.strip().upper().startswith("SELECT")


def test_retry_succeeds(populated_db, fixture_llm):
    result = ask.answer_question("How many orders did C002 place?", fixture_llm, populated_db)
    assert result.retry_count == 1
    assert result.rows == [{"n": 2}]
    assert result.token_count == (290 + 30) + (340 + 28)


# --------------------------------------------------------------------------
# answer_question — exceptions
# --------------------------------------------------------------------------


def test_out_of_scope_raises(populated_db, fixture_llm):
    with pytest.raises(ask.OutOfScope, match="product"):
        ask.answer_question(
            "Which product category has the highest revenue?",
            fixture_llm,
            populated_db,
        )


def test_retry_exhausted_raises(populated_db, fixture_llm):
    with pytest.raises(ask.RetryExhausted):
        ask.answer_question("Total revenue forever and ever", fixture_llm, populated_db)


def test_drop_table_raises_retry_exhausted(populated_db, fixture_llm):
    """Both DROP and DELETE fail on the read-only conn → RetryExhausted."""
    with pytest.raises(ask.RetryExhausted):
        ask.answer_question("Drop the orders table please", fixture_llm, populated_db)


def test_multi_statement_raises_retry_exhausted(populated_db, fixture_llm):
    with pytest.raises(ask.RetryExhausted, match="(?i)multi"):
        ask.answer_question("Sneaky multi-statement", fixture_llm, populated_db)


# --------------------------------------------------------------------------
# Internals — safety + formatting
# --------------------------------------------------------------------------


def test_multi_stmt_detector_trailing_semicolon_is_ok(populated_db):
    rows, err = ask._try_execute(populated_db, "SELECT 1 ;")
    assert err is None
    assert rows == [{"1": 1}]


def test_multi_stmt_detector_blocks_real_multi(populated_db):
    rows, err = ask._try_execute(populated_db, "SELECT 1; SELECT 2;")
    assert err is not None and "multi" in err.lower()


def test_format_empty_rows():
    assert ask._format_answer([], "SELECT * FROM orders WHERE 1=0") == "No matching orders."


def test_format_single_aggregate_uses_currency():
    out = ask._format_answer([{"total_revenue": 4230.0}], "SELECT SUM(amount_usd) AS total_revenue ...")
    assert "$" in out
    assert "4,230" in out


def test_format_multi_row():
    rows = [{"order_id": "1", "amount_usd": 100.0}, {"order_id": "2", "amount_usd": 200.0}]
    out = ask._format_answer(rows, "SELECT order_id, amount_usd FROM orders")
    assert "2 rows" in out


# --------------------------------------------------------------------------
# Prompt-hash stability
# --------------------------------------------------------------------------


def test_prompt_hash_is_deterministic():
    """The PROMPT_HASH constant must not move unless SYSTEM_PROMPT changes."""
    assert ask.PROMPT_HASH.startswith("sha256:")
    assert len(ask.PROMPT_HASH) == len("sha256:") + 16
