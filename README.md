# sap-cxii-tech-ex-02 — Submission

> **Submitted by:** Vipul Mahajan · **Role:** Development Architect – Application AI Team
> **Original exercise prompt:** [`EXERCISE.md`](./EXERCISE.md)  ·  **Design docs:** [`docs/`](./docs)  ·  **Part 4 write-up:** [`docs/architecture-multi-tenant.md`](./docs/architecture-multi-tenant.md)

A small ETL pipeline and REST API for customer order data, with an AI-augmented query layer (NL→SQL + semantic search) and an architectural extension for multi-tenant scale.

---

## Quickstart

```bash
# 1. Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate a sample dirty CSV (writes to data/orders.csv)
python scripts/generate_sample_csv.py
# Same data every run — seed=42 by default. Use --seed N for variation.

# 3. Run the ETL — populates data/orders.db AND builds the FAISS index.
#    First run is slow (~30–60s) because sentence-transformers downloads
#    the MiniLM model (~80 MB) to ~/.cache/huggingface/.
python etl.py load data/orders.csv

# 4. Start the API
export OPENAI_API_KEY=sk-...
uvicorn app:app --reload
# → http://localhost:8000/docs   (Swagger UI)
```

Run the tests (the full suite runs in under 1 second, no API key needed):

```bash
pytest -v
# Expected: ~104 passed
```

Build and run the container:

```bash
docker build -t sap-orders:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v "$PWD/data":/app/data sap-orders:latest
```

Apply the Kubernetes manifests (dry-run):

```bash
kubectl apply --dry-run=client -k k8s/
```

---

## Worked examples

Sample requests against a running service with the default sample CSV loaded.

**1. Deterministic stats**

```bash
curl -s http://localhost:8000/orders/stats | python3 -m json.tool
```

```json
{
  "total_revenue": 309788.11,
  "avg_order_value": 730.63,
  "orders_per_day": { "2025-05-28": 1, "2025-05-29": 2, "...": "..." }
}
```

**2. Natural-language question (real OpenAI call, happy path)**

```bash
curl -s -X POST http://localhost:8000/orders/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total revenue across all orders?"}' \
  | python3 -m json.tool
```

```json
{
  "answer": "total_revenue: $309,788.11",
  "sql_used": "SELECT SUM(amount_usd) AS total_revenue FROM orders;",
  "rows": [{"total_revenue": 309788.114}],
  "retry_count": 0,
  "token_count": 303
}
```

**3. Out-of-scope question → 400 (real OpenAI call)**

```bash
curl -s -i -X POST http://localhost:8000/orders/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What product categories sold best last quarter?"}'
```

```
HTTP/1.1 400 Bad Request
{"detail":"Out of scope: The database does not contain product or category information."}
```

**4. Retry loop fires (fixture mode — deterministic)**

Set `LLM_CLIENT=fixture` then ask the canned retry question. The first SQL the fixture returns references a non-existent `customers` table; the orchestrator retries with the error appended, the second SQL references the real `orders` table, and the result returns with `retry_count: 1`. Full trace in [`docs/ai-layer.md` §1.5b](./docs/ai-layer.md#15b-canonical-retry-trace-part-4a-requirement).

```bash
export LLM_CLIENT=fixture
uvicorn app:app --reload

# In another terminal
curl -s -X POST http://localhost:8000/orders/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many orders did C002 place?"}' \
  | python3 -m json.tool
```

```json
{
  "answer": "n: 10",
  "sql_used": "SELECT COUNT(*) AS n FROM orders WHERE customer_id='C002';",
  "rows": [{"n": 10}],
  "retry_count": 1,
  "token_count": 688
}
```

**5. Semantic search**

```bash
curl -s "http://localhost:8000/orders/semantic_search?q=large+EUR+order&top_k=3" \
  | python3 -m json.tool
```

```json
[
  {"order_id": "10239", "customer_id": "C014", "order_date": "2025-08-12",
   "amount_usd": 1083.41, "currency": "EUR", "score": 0.71},
  {"order_id": "10044", "customer_id": "C022", "order_date": "2025-09-30",
   "amount_usd": 921.04, "currency": "EUR", "score": 0.68},
  "..."
]
```

---

## Repository layout

```
.
├── README.md              this file — submission overview
├── EXERCISE.md            original prompt as received
├── etl.py                 CLI: python etl.py load <csv>
├── app.py                 FastAPI service
├── orders/                internal package — see orders/__init__.py
├── tests/                 pytest suite (~104 tests, runs without an API key)
├── docs/                  AI-layer spec, ADRs, Part 4d write-up
├── k8s/                   Deployment, Service, ConfigMap
├── data/                  runtime data (DB + FAISS index); gitignored
├── Dockerfile             multi-stage, non-root, port 8000
└── requirements.txt
```

---

## Design docs

Deep design lives in two short files; everything else is in this README or in code docstrings.

1. **[docs/ai-layer.md](./docs/ai-layer.md)** — LLM choice rationale, system prompt verbatim, retry policy, FAISS lifecycle, client testability.
2. **[docs/decisions.md](./docs/decisions.md)** — seven ADRs covering every non-obvious choice and the trade-off accepted.

The Part 4d architectural extension is in [docs/architecture-multi-tenant.md](./docs/architecture-multi-tenant.md) (linked from the top callout).

A live OpenAPI 3.1 spec is auto-generated by FastAPI and visible at `http://localhost:8000/docs` once the service is running.

---

## Part 4 deliverables

| Sub-part | Where to find it |
|----------|------------------|
| 4a — NL→SQL endpoint | `POST /orders/ask` in `app.py`; spec in `docs/ai-layer.md` §1. |
| 4a — system prompt | `orders/prompts.py`; reproduced in `docs/ai-layer.md` §1.3. |
| 4a — retry trace example | Documented below in [Part 4 design notes](#part-4-design-notes); see `tests/fixtures/llm_responses.json` for the canonical example. |
| 4b — semantic search | `GET /orders/semantic_search`; embedding model + FAISS lifecycle in `docs/ai-layer.md` §2. |
| 4c — LangGraph bonus | Skipped (optional). |
| 4d — multi-tenant write-up | [`docs/architecture-multi-tenant.md`](./docs/architecture-multi-tenant.md). |

---

## Part 4 design notes

This section satisfies the "in your README, document…" requirements in `EXERCISE.md` for Parts 4a and 4b. Deeper rationale and ADRs live in [`docs/ai-layer.md`](./docs/ai-layer.md) and [`docs/decisions.md`](./docs/decisions.md).

### Part 4a — Model choice

OpenAI **`gpt-4o-mini`** via the chat completions API with `response_format={"type": "json_object"}`. Three reasons:

1. **Cost.** ~$0.15 per million input tokens makes each `/orders/ask` call a fraction of a cent. The full suite of ~10 calls in the worked examples below costs under one US cent.
2. **Native JSON mode.** Removes a class of fragile string parsing. The model returns a strict envelope `{out_of_scope, reason, sql}` we can `json.loads` directly.
3. **Quality at narrow tasks.** For a five-column SQLite schema, `gpt-4o-mini` matches `gpt-4o` on NL→SQL accuracy at a fraction of the cost.

The model choice is hidden behind an `LlmClient` Protocol in [`orders/llm.py`](./orders/llm.py). Swapping to Anthropic Claude or a local Llama endpoint is a single-file change; see ADR-002 in [`docs/decisions.md`](./docs/decisions.md).

### Part 4a — System prompt (verbatim)

The prompt below is the literal `SYSTEM_PROMPT` constant in [`orders/prompts.py`](./orders/prompts.py). The version is tracked in `PROMPT_VERSION` and a SHA256 prefix is logged with every request so behaviour changes are traceable even without a version bump.

```
You convert natural-language questions about a customer-order database
into a single SQLite query.

The database has exactly one table:

Table: orders
Columns:
  order_id     TEXT    primary key, opaque string
  customer_id  TEXT    opaque string, e.g. "C001"
  order_date   TEXT    ISO 8601 date string, format YYYY-MM-DD
  amount_usd   REAL    already in USD, no other currencies present
  currency     TEXT    original currency for provenance only

Rules:
- Use only SQLite syntax.
- Use only the columns and table listed above. Do not invent columns.
- All amounts are already in USD; do not multiply by exchange rates.
- For relative dates, use SQLite functions like date('now', '-30 days').
- If the question cannot be answered from these columns, respond with
  out_of_scope=true and a short reason.

Respond ONLY with JSON matching this schema:
{
  "out_of_scope": boolean,
  "reason": string | null,
  "sql": string | null
}

When out_of_scope is true, sql must be null.
When out_of_scope is false, sql must be a single SELECT statement and
reason must be null.
```

On retry, the orchestrator appends one corrective user-role message: `Your previous SQL failed with: <error>. The bad SQL was: <sql>. Generate a corrected SQL query.` See `orders/prompts.py::retry_user_message`.

### Part 4a — Retry-loop example

See [Worked examples §4](#worked-examples) above for the runnable retry case. The full step-by-step trace (question → bad SQL → SQLite error → corrective turn → corrected SQL → success → HTTP response) is in [`docs/ai-layer.md` §1.5b](./docs/ai-layer.md).

### Part 4b — Embedding model

**`sentence-transformers/all-MiniLM-L6-v2`**. Justification for short structured-text records:

- **Size.** 22 MB on disk, ~80 MB resident. Ships in the container without a GPU; sub-20 ms query-time encoding on CPU.
- **Trained for sentence-level similarity.** Our embedding template is ~15 tokens per order (`order {id}: customer {cid} spent ${amt} USD on {date}`). MiniLM is calibrated for this length; larger models like `all-mpnet-base-v2` (768d vs 384d) gain little when the input is this constrained.
- **Stable, widely deployed.** Apache 2.0 licensed, no PII implications, and reproducible across machines.

Full ADR in [`docs/decisions.md ADR-004`](./docs/decisions.md).

### Part 4b — Index rebuild without blocking in-flight requests

`etl.py load` is the sole writer of the FAISS index. After upserting cleaned rows into SQLite, `orders.search.rebuild_from_db` encodes every order, builds an `IndexFlatIP`, and persists three files (`.idx`, `.map.json`, `.version`) using `write_tmp → fsync → atomic rename`. On POSIX the rename is atomic, so a concurrent reader either sees the old set of files or the new set, never half of each.

The API never writes the index; it only reads. `orders.search.get_or_load_index` caches the loaded `IndexBundle` in memory keyed by the `.idx` file's mtime. When `os.path.getmtime` returns a value different from the cached one (i.e. ETL has written a new index), the next request reloads the bundle transparently — no uvicorn restart, no blocked endpoint. In-flight requests continue against the previously-loaded bundle until they complete.

The race window is bounded by a single `threading.Lock` so concurrent reads don't both decide to reload at once. The "Plan B" double-buffer pattern for online ingest is documented but not implemented; see [`docs/ai-layer.md` §2.4](./docs/ai-layer.md).

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DB_PATH` | `./data/orders.db` | SQLite file |
| `INDEX_PATH` | `./data/orders.idx` | FAISS index file |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | OpenAI model |
| `LLM_CLIENT` | `openai` | `openai` or `fixture` (tests) |
| `OPENAI_API_KEY` | — | Required when `LLM_CLIENT=openai` |
| `LOG_LEVEL` | `INFO` | |
| `PROMPT_VERSION` | `v1` | Bumped when system prompt changes |
