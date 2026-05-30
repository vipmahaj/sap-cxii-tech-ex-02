"""Unit tests for the pure transformation helpers in orders.etl_core.

These run in milliseconds and have no I/O. They lock down the per-field
behaviour described in docs/data-contract.md §2.
"""

from __future__ import annotations

import pytest

from orders.etl_core import (
    LoadSummary,
    normalize_currency,
    parse_amount,
    parse_date,
    to_usd,
    transform_row,
)


# --------------------------------------------------------------------------
# parse_date
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2020-01-15", "2020-01-15"),
        ("2020/01/15", "2020-01-15"),
        ("01/15/2020", "2020-01-15"),     # MM/DD/YYYY per README example
        ("15-01-2020", "2020-01-15"),     # DD-MM-YYYY
        ("  2020-01-15  ", "2020-01-15"),  # leading/trailing whitespace tolerated
    ],
)
def test_parse_date_known_formats(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "nan", "yesterday", "2020-13-40", "garbage"])
def test_parse_date_returns_none_for_garbage(raw):
    assert parse_date(raw) is None


# --------------------------------------------------------------------------
# parse_amount
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("100", 100.0),
        ("100.50", 100.5),
        ("$1,234.56", 1234.56),
        ("  $42 ", 42.0),
        ("-50", -50.0),
    ],
)
def test_parse_amount_clean_values(raw, expected):
    assert parse_amount(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "  ", "abc", "$$$", "nan"])
def test_parse_amount_garbage_becomes_zero(raw):
    assert parse_amount(raw) == 0.0


# --------------------------------------------------------------------------
# normalize_currency
# --------------------------------------------------------------------------


def test_currency_uppercased():
    assert normalize_currency("usd") == ("USD", False)
    assert normalize_currency("eur") == ("EUR", False)


def test_currency_empty_defaults_to_usd():
    cur, was_default = normalize_currency("")
    assert cur == "USD"
    assert was_default is True


def test_currency_unknown_returns_none():
    cur, was_default = normalize_currency("GBP")
    assert cur is None
    assert was_default is False


# --------------------------------------------------------------------------
# to_usd
# --------------------------------------------------------------------------


def test_to_usd_eur_converts_at_1_1():
    assert to_usd(100.0, "EUR") == pytest.approx(110.0)


def test_to_usd_usd_is_identity():
    assert to_usd(42.0, "USD") == 42.0


# --------------------------------------------------------------------------
# transform_row — the orchestrator
# --------------------------------------------------------------------------


def _row(**overrides) -> dict[str, object]:
    base = {
        "order_id": "1001",
        "customer_id": "C123",
        "order_date": "2020-01-15",
        "amount": "100",
        "currency": "USD",
    }
    base.update(overrides)
    return base


def test_transform_row_happy_path():
    s = LoadSummary()
    out = transform_row(_row(), s)
    assert out == {
        "order_id": "1001",
        "customer_id": "C123",
        "order_date": "2020-01-15",
        "amount_usd": 100.0,
        "currency": "USD",
    }


def test_transform_row_eur_converts():
    s = LoadSummary()
    out = transform_row(_row(amount="100", currency="EUR"), s)
    assert out["amount_usd"] == pytest.approx(110.0)
    assert out["currency"] == "EUR"


def test_transform_row_missing_order_id_drops():
    s = LoadSummary()
    assert transform_row(_row(order_id=""), s) is None
    assert s.dropped_no_order_id == 1


def test_transform_row_missing_customer_drops():
    s = LoadSummary()
    assert transform_row(_row(customer_id=""), s) is None
    assert s.dropped_no_customer_id == 1


def test_transform_row_bad_date_drops():
    s = LoadSummary()
    assert transform_row(_row(order_date="yesterday"), s) is None
    assert s.dropped_bad_date == 1


def test_transform_row_unknown_currency_drops():
    s = LoadSummary()
    assert transform_row(_row(currency="GBP"), s) is None
    assert s.dropped_unknown_currency == 1


def test_transform_row_missing_currency_defaults_usd():
    s = LoadSummary()
    out = transform_row(_row(currency=""), s)
    assert out["currency"] == "USD"
    assert s.currency_filled_to_usd == 1


def test_transform_row_missing_amount_becomes_zero():
    s = LoadSummary()
    out = transform_row(_row(amount=""), s)
    assert out["amount_usd"] == 0.0
    assert s.amount_fixed_to_zero == 1


def test_transform_row_literal_zero_amount_not_counted_as_fixed():
    """User-entered $0 is legitimate data, not a 'fix'."""
    s = LoadSummary()
    out = transform_row(_row(amount="0"), s)
    assert out["amount_usd"] == 0.0
    assert s.amount_fixed_to_zero == 0
