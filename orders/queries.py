"""Read-only SQL queries for the deterministic API endpoints.

This module owns every line of SQL that runs against the orders table from
the API side. The /orders/ask endpoint in Step 3 will execute LLM-generated
SQL through a different code path; keeping the two patterns visually
separate makes the security boundary explicit.

All queries open a read-only SQLite URI connection so even programmer
error cannot mutate the orders table.
"""

from __future__ import annotations

from datetime import date, timedelta

from orders import db


# --------------------------------------------------------------------------
# Query: orders for a customer
# --------------------------------------------------------------------------


_SQL_BY_CUSTOMER = """
SELECT order_id, customer_id, order_date, amount_usd, currency
FROM   orders
WHERE  customer_id = ?
ORDER  BY order_date DESC, order_id ASC
"""


def get_orders_by_customer(db_path: str, customer_id: str) -> list[dict]:
    """Every order for one customer. Empty list if the customer is unknown."""
    with db.read_only(db_path) as conn:
        return [dict(r) for r in conn.execute(_SQL_BY_CUSTOMER, (customer_id,))]


# --------------------------------------------------------------------------
# Query: aggregate stats
# --------------------------------------------------------------------------


_SQL_STATS_AGG = """
SELECT
  COALESCE(SUM(amount_usd), 0.0) AS total,
  COUNT(*)                       AS n_orders
FROM orders
"""

_SQL_STATS_PER_DAY = """
SELECT order_date, COUNT(*) AS n
FROM   orders
GROUP  BY order_date
ORDER  BY order_date
"""


def get_stats(db_path: str) -> dict:
    """Total revenue, average order value, and orders-per-day histogram.

    Empty DB → total=0, avg=0, per_day={} — no divide-by-zero exception.
    Numeric outputs are rounded to 2dp to match cents precision.
    """
    with db.read_only(db_path) as conn:
        agg = conn.execute(_SQL_STATS_AGG).fetchone()
        total = float(agg["total"] or 0.0)
        n = int(agg["n_orders"])
        avg = (total / n) if n > 0 else 0.0
        per_day = {r["order_date"]: int(r["n"]) for r in conn.execute(_SQL_STATS_PER_DAY)}

    return {
        "total_revenue": round(total, 2),
        "avg_order_value": round(avg, 2),
        "orders_per_day": per_day,
    }


# --------------------------------------------------------------------------
# Query: recent orders within last N days
# --------------------------------------------------------------------------


_SQL_RECENT = """
SELECT order_id, customer_id, order_date, amount_usd, currency
FROM   orders
WHERE  order_date >= ?
ORDER  BY order_date DESC, order_id ASC
"""


def get_recent(db_path: str, days: int, now_override: date | None = None) -> list[dict]:
    """Orders whose order_date is within the last `days` days from `now`.

    The `now_override` argument is the seam tests use to avoid wall-clock
    flakiness. In production, callers should leave it None.
    """
    now = now_override or date.today()
    cutoff = (now - timedelta(days=days)).isoformat()
    with db.read_only(db_path) as conn:
        return [dict(r) for r in conn.execute(_SQL_RECENT, (cutoff,))]
