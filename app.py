"""FastAPI service.

Six endpoints, matching docs/api-contract.yaml:
    GET  /healthz
    GET  /orders/customer/{customer_id}
    GET  /orders/stats
    GET  /orders/recent
    POST /orders/ask
    GET  /orders/semantic_search

The handlers are stubs — they validate input and return the right shape,
but the bodies raise NotImplementedError. Implementation comes next.
"""

from __future__ import annotations

from datetime import date
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from orders import ask, config, db as db_module, embeddings, llm as llm_module, queries, search
from orders.ask import LlmError, OutOfScope, RetryExhausted
from orders.llm import LlmClient
from orders.logging_setup import configure as configure_logging, get_logger
from orders.prompts import PROMPT_VERSION


# --------------------------------------------------------------------------
# Response models — keep aligned with docs/api-contract.yaml
# --------------------------------------------------------------------------


class Order(BaseModel):
    order_id: str
    customer_id: str
    order_date: str
    amount_usd: float
    currency: str


class ScoredOrder(Order):
    score: float


class Stats(BaseModel):
    total_revenue: float
    avg_order_value: float
    orders_per_day: dict[str, int]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    sql_used: str
    rows: list[dict[str, Any]]
    retry_count: int = 0
    token_count: int = 0


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


settings = config.get_settings()
configure_logging(settings.log_level)
log = get_logger("api")

app = FastAPI(
    title="Customer Orders Query API",
    version="0.1.0",
    description="See docs/api-contract.yaml for the full contract.",
)


# --------------------------------------------------------------------------
# Dependencies — keep handlers free of env reads so tests can inject paths
# --------------------------------------------------------------------------


def get_db_path() -> str:
    return config.get_settings().db_path


def get_today() -> date:
    """Wall-clock today. Override in tests to pin date arithmetic."""
    return date.today()


def get_llm_client() -> LlmClient:
    """Build the LLM client per settings.llm_client. Override in tests."""
    return llm_module.build_client(config.get_settings())


def get_index_path() -> str:
    return config.get_settings().index_path


def get_encode_fn():
    """Return a function that encodes texts with the configured model.

    Wrapped in a dependency so tests can inject a synthetic encoder and skip
    the sentence-transformers download.
    """
    model_name = config.get_settings().embed_model_name

    def encode(texts: list[str]):
        return embeddings.encode(model_name, texts)

    return encode


# --------------------------------------------------------------------------
# Liveness
# --------------------------------------------------------------------------


@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    """Returns 'ok' if the service is up. K8s liveness target."""
    return "ok"


# --------------------------------------------------------------------------
# Deterministic endpoints
# --------------------------------------------------------------------------


@app.get("/orders/customer/{customer_id}", response_model=list[Order])
def get_orders_by_customer(
    customer_id: str,
    db_path: str = Depends(get_db_path),
) -> list[Order]:
    """Return all orders for a customer. Empty array if unknown (200, not 404)."""
    rows = queries.get_orders_by_customer(db_path, customer_id)
    return [Order(**r) for r in rows]


@app.get("/orders/stats", response_model=Stats)
def get_orders_stats(db_path: str = Depends(get_db_path)) -> Stats:
    """Aggregate metrics across all orders. Empty DB → zeros, not an error."""
    return Stats(**queries.get_stats(db_path))


@app.get("/orders/recent", response_model=list[Order])
def get_orders_recent(
    days: int = Query(..., ge=1, le=3650),
    db_path: str = Depends(get_db_path),
    today: date = Depends(get_today),
) -> list[Order]:
    """Orders with order_date within the last N days from `today`."""
    rows = queries.get_recent(db_path, days, now_override=today)
    return [Order(**r) for r in rows]


# --------------------------------------------------------------------------
# AI-augmented endpoints
# --------------------------------------------------------------------------


@app.post("/orders/ask", response_model=AskResponse)
def post_orders_ask(
    req: AskRequest,
    db_path: str = Depends(get_db_path),
    llm: LlmClient = Depends(get_llm_client),
) -> AskResponse:
    """NL→SQL with one retry on SQL execution error.

    Returns:
        200 — happy path or success on retry.
        400 — LLM returned out_of_scope=true.
        502 — retry exhausted, multi-statement SQL twice, or LLM API failure.

    See docs/ai-layer.md §1 for the full lifecycle.
    """
    request_id = uuid4().hex
    ask_log = get_logger("api.ask").bind(
        request_id=request_id,
        endpoint="/orders/ask",
        prompt_version=PROMPT_VERSION,
        prompt_hash=ask.PROMPT_HASH,
    )
    started = perf_counter()

    try:
        result = ask.answer_question(req.question, llm, db_path)
    except OutOfScope as e:
        ask_log.info("ask_out_of_scope", outcome="out_of_scope", detail=str(e),
                     latency_ms=int((perf_counter() - started) * 1000))
        raise HTTPException(status_code=400, detail=f"Out of scope: {e}")
    except RetryExhausted as e:
        ask_log.warning("ask_retry_exhausted", outcome="retry_exhausted", detail=str(e),
                        latency_ms=int((perf_counter() - started) * 1000))
        raise HTTPException(status_code=502, detail=str(e))
    except LlmError as e:
        ask_log.warning("ask_llm_error", outcome="llm_error", detail=str(e),
                        latency_ms=int((perf_counter() - started) * 1000))
        raise HTTPException(status_code=502, detail=str(e))

    ask_log.info(
        "ask_ok",
        outcome="ok",
        sql=result.sql,
        retry_count=result.retry_count,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        rows_returned=len(result.rows),
        latency_ms=int((perf_counter() - started) * 1000),
    )

    return AskResponse(
        answer=result.answer,
        sql_used=result.sql,
        rows=result.rows,
        retry_count=result.retry_count,
        token_count=result.token_count,
    )


@app.get("/orders/semantic_search", response_model=list[ScoredOrder])
def get_orders_semantic_search(
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=50),
    db_path: str = Depends(get_db_path),
    index_path: str = Depends(get_index_path),
    encode=Depends(get_encode_fn),
) -> list[ScoredOrder]:
    """Top-k orders by cosine similarity over the FAISS index.

    Returns 503 if the index has not been built yet (run `etl.py load` first).
    """
    bundle = search.get_or_load_index(index_path)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic index not ready. Run `python etl.py load <csv>` to build it.",
        )

    query_vec = encode([q])
    matches = search.search(bundle, query_vec, top_k)
    if not matches:
        return []

    matched_ids = [oid for oid, _ in matches]
    score_by_id = dict(matches)

    placeholders = ",".join("?" * len(matched_ids))
    with db_module.read_only(db_path) as conn:
        rows = [
            dict(r) for r in conn.execute(
                f"SELECT order_id, customer_id, order_date, amount_usd, currency "
                f"FROM orders WHERE order_id IN ({placeholders})",
                matched_ids,
            )
        ]
    rows_by_id = {r["order_id"]: r for r in rows}

    # Preserve FAISS ranking order and attach scores. Any matched_ids missing
    # from the DB (shouldn't happen, but defensive) are silently skipped.
    return [
        ScoredOrder(**rows_by_id[oid], score=score_by_id[oid])
        for oid in matched_ids
        if oid in rows_by_id
    ]
