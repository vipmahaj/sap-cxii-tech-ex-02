"""Generate a synthetic customer-orders CSV with controllable dirt.

Usage:
    python scripts/generate_sample_csv.py
    python scripts/generate_sample_csv.py --rows 1000 --output data/big.csv
    python scripts/generate_sample_csv.py --seed 7 --rows 50 --output -

Defaults produce a 500-row CSV at data/orders.csv with the dirt distribution
listed below. The output is deterministic for a given seed.

Dirt distribution (per row, independent draws):
    3% missing order_id      → should be dropped by ETL
    3% missing customer_id   → should be dropped
    5% unparseable date      → should be dropped
    3% unknown currency (GBP/JPY) → should be dropped
    5% missing amount        → kept, amount_fixed_to_zero += 1
    5% missing currency      → kept as USD, currency_filled_to_usd += 1
    25% non-ISO date format (mix of MM/DD/YYYY, DD-MM-YYYY, YYYY/MM/DD)
    20% amount contains $ and/or commas
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import date, timedelta


# --------------------------------------------------------------------------
# Generators for each field
# --------------------------------------------------------------------------


CUSTOMER_POOL = [f"C{i:03d}" for i in range(1, 31)]  # C001..C030
TODAY = date(2026, 5, 28)  # fixed so the CSV is deterministic regardless of clock


def _gen_order_id(rng: random.Random, idx: int, missing_pct: float) -> str:
    if rng.random() < missing_pct:
        return ""
    return str(10000 + idx)


def _gen_customer_id(rng: random.Random, missing_pct: float) -> str:
    if rng.random() < missing_pct:
        return ""
    return rng.choice(CUSTOMER_POOL)


def _gen_date(rng: random.Random, bad_pct: float, nonstandard_pct: float) -> str:
    """Return a date string. Bad dates are unparseable; nonstandard are
    parseable but in a non-ISO format."""
    if rng.random() < bad_pct:
        return rng.choice(["not-a-date", "yesterday", "2020-13-40", "abc"])

    # Random day in the last 365 days.
    d = TODAY - timedelta(days=rng.randint(0, 365))

    if rng.random() < nonstandard_pct:
        fmt = rng.choice(["%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"])
        return d.strftime(fmt)
    return d.isoformat()


def _gen_amount(rng: random.Random, missing_pct: float, formatted_pct: float) -> str:
    if rng.random() < missing_pct:
        return ""
    # Realistic distribution: mostly 10–1000, occasional bigger.
    if rng.random() < 0.05:
        value = round(rng.uniform(1000, 10000), 2)
    else:
        value = round(rng.uniform(10, 1000), 2)
    if rng.random() < formatted_pct:
        return f"${value:,.2f}"
    return f"{value:.2f}"


def _gen_currency(rng: random.Random, missing_pct: float, unknown_pct: float) -> str:
    r = rng.random()
    if r < missing_pct:
        return ""
    if r < missing_pct + unknown_pct:
        return rng.choice(["GBP", "JPY", "XYZ"])
    return rng.choices(["USD", "EUR"], weights=[0.6, 0.4])[0]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dirty customer-orders CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows", type=int, default=500, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument(
        "--output",
        type=str,
        default="data/orders.csv",
        help="Output path. Use '-' for stdout.",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)

    # Open the output sink.
    if args.output == "-":
        sink = sys.stdout
        close_after = False
    else:
        sink = open(args.output, "w", newline="", encoding="utf-8")
        close_after = True

    try:
        writer = csv.DictWriter(
            sink,
            fieldnames=["order_id", "customer_id", "order_date", "amount", "currency"],
        )
        writer.writeheader()

        for i in range(args.rows):
            writer.writerow({
                "order_id":    _gen_order_id(rng, i, missing_pct=0.03),
                "customer_id": _gen_customer_id(rng, missing_pct=0.03),
                "order_date":  _gen_date(rng, bad_pct=0.05, nonstandard_pct=0.25),
                "amount":      _gen_amount(rng, missing_pct=0.05, formatted_pct=0.20),
                "currency":    _gen_currency(rng, missing_pct=0.05, unknown_pct=0.03),
            })
    finally:
        if close_after:
            sink.close()

    print(
        f"wrote {args.rows} rows to {args.output} (seed={args.seed})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
