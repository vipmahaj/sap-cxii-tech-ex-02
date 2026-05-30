"""Shared pytest fixtures.

Each test gets a fresh temp DB and a fixture LLM client so the suite is
fully deterministic and runnable without an OpenAI key.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """Per-test SQLite path."""
    return str(tmp_path / "orders.db")


@pytest.fixture
def tmp_index_path(tmp_path: Path) -> str:
    """Per-test FAISS index path."""
    return str(tmp_path / "orders.idx")


@pytest.fixture(autouse=True)
def fixture_llm_env(monkeypatch):
    """Force the fixture LLM client for every test."""
    monkeypatch.setenv("LLM_CLIENT", "fixture")
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    yield
