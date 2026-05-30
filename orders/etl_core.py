"""Pure ETL functions called by etl.py.

Each helper handles exactly one field per data-contract.md §2. The
orchestrator transform_row() calls them in order and updates the
LoadSummary counters.

These functions are deliberately I/O-free so they're directly unit-testable
without temp files, DBs, or pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


FX_RATES = {"USD": 1.0, "EUR": 1.1}
KNOWN_CURRENCIES = set(FX_RATES.keys())

# Multi-format date parser. Order matters: more specific patterns first so
# ambiguous strings like "01/02/2020" resolve consistently (MM/DD/YYYY per the
# README's example). YYYY-MM-DD is tried first because it's the canonical form.
DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d-%m-%Y",
)


@dataclass
class LoadSummary:
    """Counts surfaced to stdout after each `etl.py load` run.

    Every field name maps to one branch in transform_row so the summary
    output is a faithful audit log of what the ETL did.
    """

    read: int = 0
    loaded: int = 0
    dropped_no_order_id: int = 0
    dropped_no_customer_id: int = 0
    dropped_bad_date: int = 0
    dropped_unknown_currency: int = 0
    amount_fixed_to_zero: int = 0
    currency_filled_to_usd: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def as_text(self) -> str:
        return (
            f"ETL summary:\n"
            f"  read:                  {self.read}\n"
            f"  loaded:                {self.loaded}\n"
            f"  dropped/no_order_id:   {self.dropped_no_order_id}\n"
            f"  dropped/no_customer:   {self.dropped_no_customer_id}\n"
            f"  dropped/bad_date:      {self.dropped_bad_date}\n"
            f"  dropped/bad_currency:  {self.dropped_unknown_currency}\n"
            f"  amount_fixed_to_zero:  {self.amount_fixed_to_zero}\n"
            f"  currency_filled_USD:   {self.currency_filled_to_usd}\n"
        )


def _clean(raw: object) -> str:
    """Coerce any cell value to a stripped string. Handles None / floats / NaN.

    pandas hands us numpy NaN for empty CSV cells; str(NaN) == 'nan', so we
    normalize that and similar oddities to empty string.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.lower() in {"nan", "none", "null", ""}:
        return ""
    return s


def parse_date(raw: str) -> str | None:
    """Normalize raw date string to ISO 8601 YYYY-MM-DD or return None to drop.

    Tries DATE_FORMATS in order. The first format that parses wins.
    """
    s = _clean(raw)
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_amount(raw: str) -> float:
    """Coerce amount string to float. Empty / unparseable → 0.0.

    Strips currency symbols ('$'), thousands separators (','), and whitespace.
    Negative amounts pass through — reviewer can decide if that's a refund.
    """
    s = _clean(raw)
    if not s:
        return 0.0
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def normalize_currency(raw: str) -> tuple[str | None, bool]:
    """Return (currency, was_filled_default).

    Returns:
        (currency_code, False) — value was present and recognised.
        ('USD',  True)         — value was empty, defaulted per spec.
        (None,   False)        — value was present but unrecognised; drop the row.
    """
    s = _clean(raw).upper()
    if not s:
        return ("USD", True)
    if s in KNOWN_CURRENCIES:
        return (s, False)
    return (None, False)


def to_usd(amount: float, currency: str) -> float:
    """Convert using fixed FX_RATES. Caller has already validated currency."""
    return amount * FX_RATES[currency]


def _is_literal_zero(s: str) -> bool:
    """True iff the cleaned input was a user-entered zero, not empty/garbage."""
    if not s:
        return False
    try:
        return float(s.replace("$", "").replace(",", "").replace(" ", "")) == 0.0
    except ValueError:
        return False


def transform_row(raw: dict[str, object], summary: LoadSummary) -> dict[str, object] | None:
    """Apply all transformations in order. Return cleaned dict or None to drop.

    Order matches data-contract.md §2. The summary counters increment in
    one place so the audit log is precise about *why* each row was dropped.
    """
    # 1. Identifiers — required.
    order_id = _clean(raw.get("order_id"))
    if not order_id:
        summary.dropped_no_order_id += 1
        return None

    customer_id = _clean(raw.get("customer_id"))
    if not customer_id:
        summary.dropped_no_customer_id += 1
        return None

    # 2. Date — drop if unparseable.
    iso_date = parse_date(str(raw.get("order_date", "")))
    if iso_date is None:
        summary.dropped_bad_date += 1
        return None

    # 3. Currency — empty defaults to USD, unknown drops the row.
    currency, was_default = normalize_currency(str(raw.get("currency", "")))
    if currency is None:
        summary.dropped_unknown_currency += 1
        return None
    if was_default:
        summary.currency_filled_to_usd += 1

    # 4. Amount — empty/unparseable becomes 0. Count as "fixed" only when
    # the original cell was *not* a legitimate zero entry.
    amount_raw = _clean(raw.get("amount"))
    amount = parse_amount(amount_raw)
    if amount == 0.0 and not _is_literal_zero(amount_raw):
        summary.amount_fixed_to_zero += 1

    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "order_date": iso_date,
        "amount_usd": to_usd(amount, currency),
        "currency": currency,
    }
