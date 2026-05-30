"""Integration tests for GET /orders/semantic_search.

A fake encoder is injected via dependency override so the test suite doesn't
need sentence-transformers or the MiniLM model download. The fake encoder
maps known query strings to known vectors so we can assert the FAISS
ranking is wired correctly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app, get_db_path, get_encode_fn, get_index_path
from orders import search
from orders.load import run_load


CSV_HEADER = ["order_id", "customer_id", "order_date", "amount", "currency"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _unit_vec(angle: float, dim: int = 4) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = float(np.cos(angle))
    v[1] = float(np.sin(angle))
    return (v / np.linalg.norm(v)).reshape(1, -1)


# A tiny encoder: maps fixed query strings to fixed vectors so we can assert
# ranking deterministically. Anything else returns the origin (which ties
# all rows to score 0).
QUERY_VECTORS = {
    "high value":            _unit_vec(0),
    "recent EUR orders":     _unit_vec(np.pi / 4),
    "needle in a haystack":  _unit_vec(np.pi),
}


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Pretend embedder. One text in, one vector out."""
    vec = QUERY_VECTORS.get(texts[0], np.zeros((1, 4), dtype=np.float32))
    return vec.astype(np.float32)


@pytest.fixture
def index_paths(tmp_path):
    """Build a 5-order DB plus a FAISS index with hand-placed vectors."""
    db_path = str(tmp_path / "orders.db")
    idx_path = str(tmp_path / "orders.idx")

    _write_csv(tmp_path / "seed.csv", [
        {"order_id": "1", "customer_id": "C001", "order_date": "2024-01-10", "amount": "100", "currency": "USD"},
        {"order_id": "2", "customer_id": "C001", "order_date": "2024-01-12", "amount": "200", "currency": "USD"},
        {"order_id": "3", "customer_id": "C002", "order_date": "2024-01-15", "amount": "300", "currency": "EUR"},
        {"order_id": "4", "customer_id": "C003", "order_date": "2024-01-15", "amount": "400", "currency": "USD"},
        {"order_id": "5", "customer_id": "C003", "order_date": "2024-01-20", "amount": "500", "currency": "USD"},
    ])
    run_load(str(tmp_path / "seed.csv"), db_path)

    # Build the index with order vectors at different angles so we can predict
    # which one will rank first for each canned query.
    order_ids = ["1", "2", "3", "4", "5"]
    vectors = np.vstack([
        _unit_vec(0),               # order 1 → aligned with "high value"
        _unit_vec(np.pi / 8),
        _unit_vec(np.pi / 4),       # order 3 → aligned with "recent EUR orders"
        _unit_vec(3 * np.pi / 8),
        _unit_vec(np.pi / 2),
    ]).reshape(5, 4)
    bundle = search.build(vectors, order_ids)
    search.persist(bundle, idx_path)
    search.invalidate_cache()

    return db_path, idx_path


@pytest.fixture
def client(index_paths):
    db_path, idx_path = index_paths
    app.dependency_overrides[get_db_path] = lambda: db_path
    app.dependency_overrides[get_index_path] = lambda: idx_path
    app.dependency_overrides[get_encode_fn] = lambda: _fake_encode
    yield TestClient(app)
    app.dependency_overrides.clear()
    search.invalidate_cache()


@pytest.fixture
def client_no_index(tmp_path):
    """A client where the index path points at nothing — should give 503."""
    db_path = str(tmp_path / "orders.db")
    idx_path = str(tmp_path / "does_not_exist.idx")

    app.dependency_overrides[get_db_path] = lambda: db_path
    app.dependency_overrides[get_index_path] = lambda: idx_path
    app.dependency_overrides[get_encode_fn] = lambda: _fake_encode
    yield TestClient(app)
    app.dependency_overrides.clear()
    search.invalidate_cache()


# --------------------------------------------------------------------------
# Scenarios from specs/acceptance.md
# --------------------------------------------------------------------------


def test_top_k_returns_scored_orders(client):
    r = client.get("/orders/semantic_search?q=high+value&top_k=3")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    # Order 1 was placed at the exact angle of "high value" — it must rank first.
    assert body[0]["order_id"] == "1"
    assert body[0]["score"] == pytest.approx(1.0, abs=1e-5)
    # Scores must be descending.
    scores = [o["score"] for o in body]
    assert scores == sorted(scores, reverse=True)


def test_each_result_has_full_order_fields(client):
    """Response must include every Order field plus score."""
    r = client.get("/orders/semantic_search?q=high+value&top_k=1")
    body = r.json()
    assert set(body[0].keys()) == {"order_id", "customer_id", "order_date",
                                    "amount_usd", "currency", "score"}


def test_different_query_returns_different_top_match(client):
    r = client.get("/orders/semantic_search?q=recent+EUR+orders&top_k=1")
    body = r.json()
    # Order 3 was placed at angle π/4 to match this query.
    assert body[0]["order_id"] == "3"


def test_missing_index_returns_503(client_no_index):
    r = client_no_index.get("/orders/semantic_search?q=anything")
    assert r.status_code == 503
    assert "index not ready" in r.json()["detail"].lower()


def test_top_k_out_of_bounds_rejected():
    r = TestClient(app).get("/orders/semantic_search?q=hi&top_k=999")
    assert r.status_code == 422


def test_q_too_short_rejected():
    r = TestClient(app).get("/orders/semantic_search?q=x")
    assert r.status_code == 422


def test_top_k_clamped_to_index_size(client):
    """Asking for more rows than the index has returns whatever exists."""
    r = client.get("/orders/semantic_search?q=high+value&top_k=50")
    assert r.status_code == 200
    assert len(r.json()) == 5


def test_index_rebuild_visible_without_restart(client, index_paths):
    """After a fresh persist, the next request should reflect the new index."""
    db_path, idx_path = index_paths

    # First request gets the original 5-order index.
    r = client.get("/orders/semantic_search?q=high+value&top_k=5")
    assert len(r.json()) == 5

    # Persist a smaller 3-order index over the same path.
    import os
    smaller_vectors = np.vstack([
        _unit_vec(0), _unit_vec(np.pi / 8), _unit_vec(np.pi / 4),
    ]).reshape(3, 4)
    bundle = search.build(smaller_vectors, ["1", "2", "3"])
    new_mtime = os.path.getmtime(idx_path) + 10
    os.utime(idx_path, (new_mtime, new_mtime))   # force mtime delta
    search.persist(bundle, idx_path)

    # Same client, no restart: next request reflects the new index.
    r2 = client.get("/orders/semantic_search?q=high+value&top_k=5")
    assert len(r2.json()) == 3
