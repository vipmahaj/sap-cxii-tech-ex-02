"""Unit tests for orders.search — uses real FAISS with synthetic vectors,
so we don't need sentence-transformers or the MiniLM model download.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from orders import search


def _unit_vec(angle: float, dim: int = 4) -> np.ndarray:
    """Generate a deterministic L2-normalized vector at a given angle."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = float(np.cos(angle))
    v[1] = float(np.sin(angle))
    return v / np.linalg.norm(v)


def _build_test_bundle(n: int = 5, dim: int = 4) -> search.IndexBundle:
    """5 vectors at evenly-spaced angles, IDs '1'..'5'."""
    vectors = np.vstack([_unit_vec(i * np.pi / 8, dim) for i in range(n)])
    order_ids = [str(i + 1) for i in range(n)]
    return search.build(vectors, order_ids)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def test_build_returns_bundle_with_correct_shape():
    bundle = _build_test_bundle()
    assert bundle.dim == 4
    assert bundle.order_ids == ["1", "2", "3", "4", "5"]
    assert bundle.index.ntotal == 5


def test_build_rejects_size_mismatch():
    vecs = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="must match"):
        search.build(vecs, ["a", "b"])


def test_build_rejects_one_dimensional_input():
    with pytest.raises(ValueError, match="2-D"):
        search.build(np.zeros(4, dtype=np.float32), ["x"])


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def test_search_returns_self_as_top_match():
    """A vector should be its own nearest neighbour with score ≈ 1."""
    bundle = _build_test_bundle()
    # The third vector is at angle 2π/8 with score 1.0 against itself.
    query = _unit_vec(2 * np.pi / 8).reshape(1, -1)
    results = search.search(bundle, query, top_k=1)
    assert results[0][0] == "3"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_returns_results_in_descending_score():
    bundle = _build_test_bundle()
    query = _unit_vec(0).reshape(1, -1)  # aligns with vector "1"
    results = search.search(bundle, query, top_k=5)
    assert results[0][0] == "1"
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_clamps_to_index_size():
    """Asking for more rows than exist returns all rows."""
    bundle = _build_test_bundle(n=3)
    query = _unit_vec(0).reshape(1, -1)
    results = search.search(bundle, query, top_k=999)
    assert len(results) == 3


# --------------------------------------------------------------------------
# persist + load round-trip
# --------------------------------------------------------------------------


def test_persist_creates_three_files(tmp_path):
    bundle = _build_test_bundle()
    idx_path = str(tmp_path / "orders.idx")
    search.persist(bundle, idx_path)

    assert Path(idx_path).is_file()
    assert Path(search.map_path(idx_path)).is_file()
    assert Path(search.version_path(idx_path)).is_file()


def test_load_returns_equivalent_bundle(tmp_path):
    original = _build_test_bundle()
    idx_path = str(tmp_path / "orders.idx")
    search.persist(original, idx_path)

    loaded = search.load(idx_path)
    assert loaded is not None
    assert loaded.order_ids == original.order_ids
    assert loaded.dim == original.dim
    assert loaded.index.ntotal == original.index.ntotal


def test_load_returns_none_when_files_missing(tmp_path):
    assert search.load(str(tmp_path / "nope.idx")) is None


def test_load_returns_none_when_version_mismatches(tmp_path):
    bundle = _build_test_bundle()
    idx_path = str(tmp_path / "orders.idx")
    search.persist(bundle, idx_path)
    Path(search.version_path(idx_path)).write_text("v999")
    assert search.load(idx_path) is None


# --------------------------------------------------------------------------
# get_or_load_index (mtime-keyed cache)
# --------------------------------------------------------------------------


def test_cache_hit_returns_same_instance(tmp_path):
    bundle = _build_test_bundle()
    idx_path = str(tmp_path / "orders.idx")
    search.persist(bundle, idx_path)
    search.invalidate_cache()

    first = search.get_or_load_index(idx_path)
    second = search.get_or_load_index(idx_path)
    assert first is second  # exact same object — confirms cache hit


def test_cache_reload_on_mtime_change(tmp_path):
    """After ETL rebuilds the index, the next request should see the new bundle."""
    idx_path = str(tmp_path / "orders.idx")
    search.persist(_build_test_bundle(n=3), idx_path)
    search.invalidate_cache()

    first = search.get_or_load_index(idx_path)
    assert first.index.ntotal == 3

    # Force a different mtime then rebuild with 5 vectors.
    new_mtime = os.path.getmtime(idx_path) + 10
    os.utime(idx_path, (new_mtime, new_mtime))
    search.persist(_build_test_bundle(n=5), idx_path)

    second = search.get_or_load_index(idx_path)
    assert second is not first
    assert second.index.ntotal == 5


def test_cache_returns_none_when_file_missing(tmp_path):
    search.invalidate_cache()
    assert search.get_or_load_index(str(tmp_path / "missing.idx")) is None
