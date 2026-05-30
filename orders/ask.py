"""NL→SQL orchestration: call LLM, execute SQL, retry once, format answer.

This module owns the retry loop and the safety rails. The handler in app.py
just translates exceptions to HTTP status codes.

Error model:
    OutOfScope        — LLM said the question is unanswerable from the schema.
    RetryExhausted    — Both SQL attempts failed.
    LlmError          — The LLM API itself failed (network, auth, timeout).

Per ai-layer.md §1.4, only sqlite3 execution errors trigger the retry.
LLM-API failures do NOT retry.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from orders import db
from orders.llm import LlmClient
from orders.prompts import SYSTEM_PROMPT


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------


PROMPT_HASH = "sha256:" + hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]


@dataclass
class AskResult:
    """Successful response from answer_question. Maps 1:1 to AskResponse."""

    answer: str
    sql: str
    rows: list[dict[str, Any]]
    retry_count: int
    input_tokens: int
    output_tokens: int

    @property
    def token_count(self) -> int:
        return self.input_tokens + self.output_tokens


class OutOfScope(Exception):
    """LLM determined the question cannot be answered from the schema."""


class RetryExhausted(Exception):
    """Both SQL attempts failed with sqlite3 errors."""


class LlmError(Exception):
    """The LLM API call itself failed (does not trigger a retry)."""


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def answer_question(question: str, llm: LlmClient, db_path: str) -> AskResult:
    """Run the NL→SQL pipeline.

    1. Ask LLM for SQL.
    2. If out_of_scope → raise OutOfScope.
    3. Execute SQL on read-only connection.
    4. If SQL errors → ask LLM for a corrected SQL with the error appended.
    5. Execute corrected SQL.
    6. If still errors → raise RetryExhausted.

    Returns AskResult on success.
    """
    # First attempt --------------------------------------------------------
    try:
        result = llm.generate_sql(question)
    except Exception as e:  # noqa: BLE001 — anything from the LLM client
        raise LlmError(f"LLM call failed: {e}") from e

    input_tokens = result.input_tokens
    output_tokens = result.output_tokens

    if result.out_of_scope:
        raise OutOfScope(result.reason or "Question is out of scope for the orders schema.")

    sql = result.sql or ""
    rows, error = _try_execute(db_path, sql)

    if error is None:
        return AskResult(
            answer=_format_answer(rows, sql),
            sql=sql,
            rows=rows,
            retry_count=0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # Retry — exactly once -------------------------------------------------
    try:
        result = llm.generate_sql(question, retry_context=(sql, error))
    except Exception as e:  # noqa: BLE001
        raise LlmError(f"LLM call failed on retry: {e}") from e

    input_tokens += result.input_tokens
    output_tokens += result.output_tokens

    if result.out_of_scope:
        # Unusual but plausible: the LLM concludes on retry that it can't answer.
        raise OutOfScope(result.reason or "Question is out of scope (declared on retry).")

    retry_sql = result.sql or ""
    rows, error = _try_execute(db_path, retry_sql)

    if error is not None:
        raise RetryExhausted(
            f"LLM produced invalid SQL twice. Last error: {error}. Last SQL: {retry_sql}"
        )

    return AskResult(
        answer=_format_answer(rows, retry_sql),
        sql=retry_sql,
        rows=rows,
        retry_count=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# --------------------------------------------------------------------------
# Safety + execution
# --------------------------------------------------------------------------


_MULTI_STMT_RE = re.compile(r";\s*\S")


def _try_execute(db_path: str, sql: str) -> tuple[list[dict[str, Any]], str | None]:
    """Execute SQL on a read-only SQLite connection.

    Returns (rows, error). rows is [] when error is not None.
    """
    sql = (sql or "").strip()
    if not sql:
        return [], "Empty SQL"

    if _MULTI_STMT_RE.search(sql):
        # Reject before sending to SQLite. Multi-statement strings are a known
        # SQL-injection surface even on read-only connections.
        return [], "Multi-statement SQL is not allowed"

    try:
        with db.read_only(db_path) as conn:
            cursor = conn.execute(sql)
            return [dict(r) for r in cursor.fetchall()], None
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return [], str(e)


# --------------------------------------------------------------------------
# Answer formatting (rule-based, no second LLM call)
# --------------------------------------------------------------------------


_AGG_KEYWORDS = ("sum", "avg", "min", "max", "count", "total")


def _format_answer(rows: list[dict[str, Any]], sql: str) -> str:
    """Turn the row set into a one-line natural-language summary.

    Three cases:
      empty rows                       → "No matching orders."
      single row with a single column  → "{column}: {value}" with currency hint
      anything else                    → "Returned N rows."
    """
    if not rows:
        return "No matching orders."

    if len(rows) == 1 and len(rows[0]) == 1:
        (col, value), = rows[0].items()
        return f"{col}: {_format_value(col, value)}"

    return f"Returned {len(rows)} row{'s' if len(rows) != 1 else ''}."


def _format_value(column_name: str, value: Any) -> str:
    """Apply currency-style formatting when the column looks money-shaped.

    NULL aggregates (e.g. SUM over zero matching rows) render as $0.00 for
    money-shaped columns rather than the unhelpful 'None'.
    """
    col_lower = column_name.lower()
    money_shaped = any(m in col_lower for m in ("amount", "revenue", "sum", "total", "avg"))

    if value is None:
        return "$0.00" if money_shaped else "0"

    if isinstance(value, (int, float)):
        if money_shaped:
            return f"${value:,.2f}"
        if any(k in col_lower for k in _AGG_KEYWORDS):
            return f"{value:,}"
    return str(value)
