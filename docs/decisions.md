# Architecture Decision Records

Short, dated, append-only. One ADR per non-obvious choice.

---

## ADR-001 — Storage: SQLite over Parquet

**Date:** 2026-05-28
**Status:** Accepted

**Context.** The exercise allows SQLite or Parquet. We have deterministic query endpoints (`/orders/customer`, `/orders/stats`, `/orders/recent`) and an LLM-generated SQL endpoint (`/orders/ask`).

**Decision.** Use SQLite as the authoritative store.

**Consequences.**
- `+` `/orders/ask` becomes trivial: the LLM emits SQL, we execute it directly.
- `+` Deterministic endpoints are also SQL queries, which means there's exactly one query language across the codebase.
- `+` Indexes on `customer_id` and `order_date` give us sub-millisecond lookups for the toy dataset and scale to millions of rows.
- `−` Single-writer, no horizontal scale. Acceptable: the service is read-heavy and the writer is the offline ETL.
- `−` Parquet would be denser on disk. Not material at this scale.

---

## ADR-002 — LLM provider: OpenAI gpt-4o-mini

**Date:** 2026-05-28
**Status:** Accepted

**Context.** Part 4a requires an LLM to convert questions to SQL. The README invites us to justify the pick.

**Decision.** OpenAI `gpt-4o-mini` via the chat completions API with JSON mode.

**Consequences.**
- `+` Cheapest credible option for NL→SQL.
- `+` Native JSON mode removes a class of parsing bugs.
- `+` Low operational overhead for a 5–6 hour exercise.
- `−` PII flows to a third-party API. Acceptable for the reference build; addressed explicitly in Part 4d for enterprise tenants.
- `−` Vendor lock-in on the client SDK. Mitigated by hiding the call behind a single `llm_client.generate_sql(question)` function.

**Alternatives considered.**
- Anthropic Claude Haiku — equally strong; chose OpenAI for native JSON mode.
- Local Ollama — eliminates PII concern but adds image weight and latency.

---

## ADR-003 — Vector store: FAISS in-process

**Date:** 2026-05-28
**Status:** Accepted

**Context.** Part 4b requires storing embeddings and doing nearest-neighbor search.

**Decision.** `faiss-cpu` with `IndexFlatIP` over L2-normalized vectors, persisted to disk.

**Consequences.**
- `+` Zero infrastructure beyond `pip install faiss-cpu`.
- `+` Exact cosine search is fast enough at this scale (sub-ms for 100k rows).
- `+` Easy to reason about: vectors live in one file, mapping lives next to it.
- `−` In-process. Doesn't scale beyond a single pod. Acceptable for the reference build; Part 4d discusses sharded indexes.
- `−` Approximate index types (IVF, HNSW) would be faster at million-row scale but are unnecessary here.

**Alternatives considered.**
- numpy + cosine — even simpler, but no incremental upgrade path. FAISS gives us optionality.
- pgvector / Weaviate / Pinecone — operationally heavy for an exercise; right call for a 50-tenant production system.

---

## ADR-004 — Embedding model: all-MiniLM-L6-v2

**Date:** 2026-05-28
**Status:** Accepted

**Context.** We need to embed short structured-text strings (~15 tokens each) at ETL time and at query time.

**Decision.** `sentence-transformers/all-MiniLM-L6-v2`.

**Consequences.**
- `+` 22 MB on disk, ~80 MB resident. Cheap to ship in a container.
- `+` ~14 ms / sentence on CPU. Encoding 10k orders ≈ 2 minutes.
- `+` Strong on short-text similarity, which is exactly what we have.
- `−` 384 dimensions vs. 768 for `mpnet-base-v2`. Marginally less semantic resolution; not noticeable on this task.
- `−` English-only training data. Acceptable: order data is structured text, not free-form prose.

---

## ADR-005 — Embedding text template

**Date:** 2026-05-28
**Status:** Accepted

**Context.** What text representation of an order do we embed? The README suggests `"customer C001, $320 USD, 2024-03-15"`.

**Decision.** Fixed template: `order {order_id}: customer {customer_id} spent ${amount_usd:.2f} USD on {order_date}`.

**Consequences.**
- `+` Includes the order ID so the embedding can also resolve "find me order 1042" style queries.
- `+` Natural-language verb (`spent`) anchors semantics — queries like "biggest spenders" embed nearby.
- `+` Identical at index time and (notionally) at query time, modulo the user's free-text input.
- `−` Schema-coupled — if a new column is added the template and the index version both bump.

---

## ADR-006 — One retry, in code, not in the prompt

**Date:** 2026-05-28
**Status:** Accepted

**Context.** README requires that on a SQL failure we retry once with the error appended.

**Decision.** Implement the retry loop as Python code in `app.py`, not as a "if your SQL was wrong, try again" hint in the system prompt. Failures appear as a real user-role turn appended to the conversation.

**Consequences.**
- `+` Deterministic: exactly one retry, by construction.
- `+` Observable: we log `retry_count` separately.
- `+` Honest to the LLM: it sees its own failed SQL and the real database error.
- `−` Slightly more code than a single API call.

---

## ADR-007 — Index rebuilds happen offline in ETL, not in the API

**Date:** 2026-05-28
**Status:** Accepted

**Context.** The README asks how we handle index rebuilds without blocking in-flight requests.

**Decision.** ETL is the only writer. It builds the index to a temp file and atomically renames into place. The API only reads.

**Consequences.**
- `+` No locking needed in the API process.
- `+` Atomic POSIX rename means in-flight requests never see a partial index.
- `−` Doesn't address online ingest. Acceptable for the exercise; Part 4d describes the double-buffer pattern for production.
