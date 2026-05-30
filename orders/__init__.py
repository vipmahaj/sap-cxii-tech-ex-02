"""Internal package for the customer-orders ETL + API service.

Module map:
    config           — env-driven settings (Settings dataclass).
    db               — SQLite connection helpers (read-only and read-write).
    etl_core         — pure functions for clean/transform; called by etl.py.
    embeddings       — sentence-transformers wrapper.
    search           — FAISS index lifecycle (build, persist, load, query).
    llm              — LlmClient Protocol + OpenAI and Fixture impls.
    prompts          — system prompt template + version.
    logging_setup    — structured-log configuration.

Public surface is intentionally small. app.py and etl.py compose these.
"""
