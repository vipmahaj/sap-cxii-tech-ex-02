# AI Layer Spec

This document defines the contract for the two AI-augmented endpoints: `POST /orders/ask` (NL→SQL) and `GET /orders/semantic_search`.

---

## 1. NL→SQL endpoint (`POST /orders/ask`)

### 1.1 Model choice

**Selected: OpenAI `gpt-4o-mini`.**

Justification:

- **Cost.** ~$0.15 / 1M input tokens makes the per-question cost negligible (~300 tokens × $0.15/1M ≈ $0.00005). Important if this endpoint ever gets pointed at a batch.
- **Latency.** Sub-second time-to-first-token in practice, which keeps `/orders/ask` feeling synchronous.
- **NL→SQL quality.** Strong at structured output, especially with a constrained schema and a few-shot system prompt. Comparable to larger models on this narrow task; the upside of `gpt-4o` over `gpt-4o-mini` is not justified for a five-column SQLite schema.
- **JSON mode.** Native support for `response_format={"type": "json_object"}`, which we use to force a strict envelope and remove fragile string parsing.
- **Operational simplicity.** Single API key, no quota dance, no model server to host.

Alternatives considered:
- Anthropic Claude Haiku — equally strong; chose OpenAI for the native JSON mode in this exercise.
- Local Ollama (`qwen2.5-coder:7b` or `sqlcoder`) — eliminates third-party PII concern but adds ~3 GB image weight and inference latency, neither acceptable in a 5–6 hour exercise. Discussed in Part 4d.

### 1.2 Request lifecycle

```
client → app.py
            │
            ├── 1. validate question (length 3–500, not empty)
            ├── 2. build messages = [system_prompt, user_question]
            ├── 3. call gpt-4o-mini with response_format=json
            ├── 4. parse {sql, out_of_scope, reason}
            │     ├── if out_of_scope → return 400
            │     └── else continue
            ├── 5. execute sql against SQLite (read-only connection)
            │     ├── on success → format answer, return 200
            │     └── on sqlite OperationalError →
            │            6. retry once: append error to messages, repeat 3–5
            │                  ├── on success → return 200 with retry_count=1
            │                  └── on second failure → return 502
            └── log {request_id, prompt_hash, sql, retry_count, token_count, latency_ms, outcome}
```

### 1.3 System prompt template

The system prompt is a single Python f-string. Versioned via a `PROMPT_VERSION` constant — logged with each request so we can correlate behavior to prompt edits.

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

### 1.4 Retry policy

- **One retry**, exactly. Implemented in code, not by the model.
- On `sqlite3.OperationalError`, `sqlite3.DatabaseError`, or empty SQL: append a user-role message:
  `Your previous SQL failed with: <error>. The bad SQL was: <sql>. Generate a corrected SQL query.`
- After the retry, any failure → `502 Bad Gateway` with the last error in `detail`.

### 1.5 Safety constraints

- The SQLite connection used for `/orders/ask` is opened with `mode=ro` (read-only URI) so that even if the model generates DML it cannot mutate state.
- SQL containing `;` followed by a non-whitespace character is rejected before execution — defense against multi-statement injection in a single string.
- A 5-second timeout is applied to the SQL execution.

### 1.5b Canonical retry trace (Part 4a requirement)

The README requires "one example where the retry loop fired: what the bad SQL was, what error it produced, and what the corrected SQL looked like." This is that example, reproduced verbatim from `tests/fixtures/llm_responses.json` and `tests/test_ask.py::test_retry_loop_fires`.

**Question (input)**

```
How many orders did C002 place?
```

**Attempt 1 — LLM output**

```json
{
  "out_of_scope": false,
  "reason": null,
  "sql": "SELECT COUNT(*) AS n FROM customers WHERE id='C002';"
}
```

The LLM hallucinated a `customers` table — that table doesn't exist in our schema; only `orders` does.

**Attempt 1 — SQLite response**

```
sqlite3.OperationalError: no such table: customers
```

**Retry — corrective user-role message appended**

The orchestrator (`orders/ask.py::answer_question`) calls `llm.generate_sql(question, retry_context=("SELECT COUNT(*) AS n FROM customers WHERE id='C002';", "no such table: customers"))`. The client reconstructs the previous assistant turn and appends:

```
Your previous SQL failed with: no such table: customers.
The bad SQL was: SELECT COUNT(*) AS n FROM customers WHERE id='C002';.
Generate a corrected SQL query.
```

**Attempt 2 — LLM output**

```json
{
  "out_of_scope": false,
  "reason": null,
  "sql": "SELECT COUNT(*) AS n FROM orders WHERE customer_id='C002';"
}
```

The model now references the correct table and column.

**Attempt 2 — SQLite response (success)**

```
[{"n": 10}]
```

**API response (200 OK)**

```json
{
  "answer": "n: 10",
  "sql_used": "SELECT COUNT(*) AS n FROM orders WHERE customer_id='C002';",
  "rows": [{"n": 10}],
  "retry_count": 1,
  "token_count": 688
}
```

`retry_count = 1` is the explicit signal that the loop fired exactly once. `token_count = 688` is the sum across both LLM calls (320 first + 368 retry).

**Why this example is in the fixture file, not a live OpenAI trace**

Reproducing this scenario against a real OpenAI call would require either prompt-injection to force the model to hallucinate a table (fragile) or a recorded cassette (more infra). The fixture client makes the scenario deterministic and CI-runnable. The same orchestration code paths execute regardless of which client is wired in.

### 1.6 Logging schema

One JSON line per request, written to stdout:

```json
{
  "ts": "2026-05-28T10:34:21.451Z",
  "request_id": "01J...",
  "endpoint": "/orders/ask",
  "prompt_version": "v1",
  "prompt_hash": "sha256:abcd...",
  "model": "gpt-4o-mini",
  "input_tokens": 312,
  "output_tokens": 64,
  "sql": "SELECT SUM(amount_usd) FROM orders WHERE ...",
  "retry_count": 0,
  "outcome": "ok",
  "latency_ms": 842
}
```

---

## 2. Semantic search endpoint (`GET /orders/semantic_search`)

### 2.1 Embedding model

**Selected: `sentence-transformers/all-MiniLM-L6-v2`.**

Justification:

- **Size.** 22 MB on disk, ~80 MB resident. Fits comfortably in a Kubernetes pod sidecar memory budget.
- **Speed.** ~14 ms / sentence on CPU for short inputs — encoding 10k orders at ETL time takes ~2 minutes; query-time encoding is sub-20 ms.
- **Quality on short structured text.** The embedding-text template is ~15 tokens. MiniLM is trained on sentence-level tasks and is well-calibrated for short inputs. Larger models (`mpnet-base-v2`) gain little when the input is this constrained.
- **No GPU required.** Aligns with the Docker / Kubernetes story (no CUDA in the runtime image).

### 2.2 Vector store

**Selected: `faiss-cpu` with `IndexFlatIP`.**

- **`IndexFlatIP` over normalized vectors = exact cosine search.** Toy dataset doesn't justify approximate indexes (IVF, HNSW). On 10k–100k orders the latency is sub-millisecond.
- **Persisted to disk** so service restarts don't trigger re-encoding.
- Discussion of when to switch to IVF/HNSW or a managed vector DB (pgvector, Pinecone) is in `decisions.md` and Part 4d.

### 2.3 Index lifecycle

| Trigger | Action |
|---------|--------|
| `etl.py load` succeeds | Encode all rows, build new index in memory, write to `orders.idx.tmp`, fsync, rename atomically to `orders.idx`. |
| App starts | mmap-load `orders.idx` and the `orders.idx.map.json` sidecar. If absent → search endpoint returns 503. |
| App in steady state | Index is read-only at runtime; ETL is the only writer. |

### 2.4 Index rebuilds without blocking in-flight requests

**Plan A (implemented in this exercise):**

The index is built by the offline `etl.py` process, not by the API service. The API never rebuilds in-process. When the file changes, the API picks it up on the next request via a stat check; in-flight requests continue using the previously mmaped index until they finish. Worst case during the swap: a request reads a slightly stale index. There is no read-vs-write race because the rename is atomic on POSIX filesystems.

**Plan B (documented but not implemented, called out in Part 4d):**

For online ingest, a double-buffer pattern: hold both `current_index` and `next_index` references in app state, swap atomically under a read-write lock after the rebuild completes. This is the design I would adopt in production; for the exercise it is acknowledged but skipped to stay inside the time budget.

### 2.5 Query lifecycle

```
client → app.py
            │
            ├── 1. validate q (min length 2), top_k (1–50, default 5)
            ├── 2. embed q with MiniLM, L2-normalize
            ├── 3. faiss.search(top_k) → (distances, vector_rows)
            ├── 4. resolve vector_rows → order_ids via sidecar map
            ├── 5. SELECT * FROM orders WHERE order_id IN (...)
            ├── 6. merge scores onto orders, return sorted by score desc
            └── log {request_id, q, top_k, latency_ms}
```

---

## 3. Client shape and testability

The LLM call is hidden behind a small interface so the API layer is provider-agnostic and the test suite is deterministic.

```python
from typing import Protocol

class LlmClient(Protocol):
    def generate_sql(
        self,
        question: str,
        retry_context: tuple[str, str] | None = None,  # (bad_sql, error_msg)
    ) -> "LlmResult": ...

@dataclass
class LlmResult:
    out_of_scope: bool
    reason: str | None
    sql: str | None
    input_tokens: int
    output_tokens: int
```

Two implementations:

- `OpenAiLlmClient` — production. Calls `gpt-4o-mini` with JSON mode.
- `FixtureLlmClient` — tests. Reads responses from `tests/fixtures/llm_responses.json` keyed by question. Makes the "retry loop fires" and "retry exhausted" acceptance scenarios deterministic and CI-runnable without an API key.

Switched via `LLM_CLIENT` env var (`openai` | `fixture`). The choice happens at app startup; the API layer never sees the concrete class.

This also previews the Part 4d "model-agnostic prompt layer" argument: swapping in `OnPremLlamaLlmClient` or `AnthropicLlmClient` is a new file, not a change to `app.py`.

---

## 4. Scope note: multi-tenancy

This service is single-tenant by design for the reference implementation. Tenant routing, per-tenant index isolation, per-tenant LLM backends, and PII guardrails are addressed in writing only — see `architecture-multi-tenant.md` (to be drafted as the Part 4d deliverable). No code in `app.py`, `etl.py`, or the Kubernetes manifests references a tenant ID.

---

## 5. Configuration

Settings exposed via env (Kubernetes ConfigMap / Secret):

| Variable | Default | Notes |
|----------|---------|-------|
| `DB_PATH` | `./data/orders.db` | SQLite file. |
| `INDEX_PATH` | `./data/orders.idx` | FAISS index file. |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Override for experiments. |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Override for experiments. |
| `LLM_CLIENT` | `openai` | `openai` or `fixture`. `fixture` is used by tests. |
| `OPENAI_API_KEY` | — | Secret; required when `LLM_CLIENT=openai`. |
| `LOG_LEVEL` | `INFO` | |
| `PROMPT_VERSION` | `v1` | Bumped when system prompt is edited. |
