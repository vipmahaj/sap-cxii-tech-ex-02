# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ETL pipeline and FastAPI REST API for customer order data, with an AI-augmented query layer (NL→SQL via LLM) and semantic search using FAISS embeddings. Built as a SAP CX II Technical Exercise submission.

## Common Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate sample data (500 rows, seed=42, deterministic)
python scripts/generate_sample_csv.py

# Run ETL (populates SQLite DB + FAISS index)
# First run is slow (~30-60s) — downloads sentence-transformers model (~80 MB)
python etl.py load data/orders.csv

# Start API server
uvicorn app:app --reload

# Run all tests (~104 tests, <2s, no API key needed — uses fixture LLM)
pytest -v

# Run single test file
pytest tests/test_api.py -v

# Run tests by keyword
pytest -k "test_ask" -v

# Lint
ruff check .

# Docker
docker build -t sap-orders:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY -v "$PWD/data":/app/data sap-orders:latest

# Kubernetes (validate / apply)
kubectl apply --dry-run=client -k k8s/
kubectl apply -k k8s/
```

## Architecture

### Entry Points

- **etl.py** — CLI: `python etl.py load <csv_path>` — thin argparse wrapper around `orders.load.run_load()`
- **app.py** — FastAPI with 6 endpoints (see `docs/api-contract.yaml` for full contract):
  - `GET /healthz`, `GET /orders/customer/{customer_id}`, `GET /orders/stats`, `GET /orders/recent?days=N`
  - `POST /orders/ask` — NL→SQL with single-retry loop (Part 4a)
  - `GET /orders/semantic_search?q=...&top_k=5` — FAISS vector search (Part 4b)

### Data Flow

**Ingestion:** CSV → `etl_core.transform_row()` per row → SQLite upsert (`INSERT OR REPLACE`) → `search.rebuild_from_db()` (embed all orders → build FAISS → atomic persist)

**Query:** HTTP request → FastAPI handler → `read_only()` SQLite connection (or LLM/FAISS for AI endpoints) → JSON response

### Key Cross-Cutting Patterns

- **Two SQLite connection modes** (`db.py`): `read_write()` for ETL, `read_only()` (uses `file:...?mode=ro` URI) for all query endpoints — prevents prompt injection from mutating data.
- **LlmClient Protocol** (`llm.py`): `OpenAiLlmClient` (production, gpt-4o-mini, JSON mode) and `FixtureLlmClient` (tests, reads from `tests/fixtures/llm_responses.json`). Selected by `LLM_CLIENT` env var via `build_client()` factory.
- **NL→SQL safety** (`ask.py`): Multi-statement SQL is rejected before execution. Answer formatting is rule-based (not LLM-generated) to avoid sending customer data back through the LLM. Only sqlite3 execution errors trigger retry; LLM-API failures do NOT retry.
- **Atomic FAISS index writes** (`search.py`): `.tmp` file → fsync → rename. `get_or_load_index()` caches by `.idx` mtime and reloads transparently — no restart needed. Thread-safe via `threading.Lock`.
- **FastAPI dependency injection** (`app.py`): `get_db_path()`, `get_llm_client()`, `get_today()`, `get_encode_fn()` — all overridable in tests via `app.dependency_overrides`.

### SQLite Schema

```sql
orders(order_id TEXT PK, customer_id TEXT, order_date TEXT, amount_usd REAL, currency TEXT)
-- Indexes on customer_id and order_date
```

## Testing

Tests use `FixtureLlmClient` automatically (conftest.py autouse fixture sets `LLM_CLIENT=fixture`). No OpenAI API key needed.

**Key conftest fixtures:** `tmp_db_path`, `tmp_index_path`, `fixture_llm_env` (autouse — forces fixture LLM for every test).

**API integration tests** use `app.dependency_overrides` to inject temp DB paths, pinned dates, and synthetic encoders. Pattern:
```python
app.dependency_overrides[get_db_path] = lambda: tmp_db_path
yield TestClient(app)
app.dependency_overrides.clear()
```

**Adding `/ask` test cases:** Add a new entry to `tests/fixtures/llm_responses.json` keyed by exact question text, with `"first"` and optionally `"retry"` sub-objects containing `{out_of_scope, reason, sql, input_tokens, output_tokens}`.

## Key Design Decisions

- **No pandas** — stdlib `csv.DictReader` for ETL; keeps image small and dependencies minimal
- **Protocol-based LlmClient** — enables fixture-based testing with zero API cost; swapping to another LLM provider is a single-file change
- **Rule-based answer formatting** (not LLM-generated) — avoids sending customer data back through LLM
- **Single retry** for NL→SQL — error message and bad SQL appended as conversation turns; LLM-API failures are NOT retried
- **ETL currency normalization** — EUR→USD at hardcoded 1.1 rate; all amounts stored as `amount_usd`

## Environment Variables

See `.env.example` for all variables. Key ones: `DB_PATH`, `INDEX_PATH`, `LLM_CLIENT` (openai|fixture), `OPENAI_API_KEY`, `EMBED_MODEL_NAME`, `LLM_MODEL_NAME`, `LOG_LEVEL`, `SQL_TIMEOUT_SECONDS`, `MAX_QUESTION_LENGTH`.

## Design Docs

- `docs/ai-layer.md` — LLM choice rationale, system prompt, retry policy, FAISS lifecycle
- `docs/decisions.md` — seven ADRs covering non-obvious choices
- `docs/architecture-multi-tenant.md` — Part 4d multi-tenant extension
