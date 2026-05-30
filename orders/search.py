"""FAISS index lifecycle.

ETL is the sole writer; the API only reads. Persistence is atomic — every
file (`.idx`, `.map.json`, `.version`) is written to a `.tmp` first, fsynced,
then renamed into place. In-flight reads continue against the old mmap
until they complete.

The API cache (get_or_load_index) checks the `.idx` mtime on each call and
reloads when a newer index appears, implementing the "rebuild visible
without restart" requirement in specs/acceptance.md.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


INDEX_VERSION = "v1"


# --------------------------------------------------------------------------
# Bundle — in-memory view of the persisted index
# --------------------------------------------------------------------------


@dataclass
class IndexBundle:
    """The trio of artefacts we need at query time."""

    index: object        # faiss.IndexFlatIP — `object` to avoid hard import here
    order_ids: list[str]
    dim: int


# --------------------------------------------------------------------------
# Build + persist
# --------------------------------------------------------------------------


def build(vectors: np.ndarray, order_ids: list[str]) -> IndexBundle:
    """Build a FAISS IndexFlatIP from L2-normalized vectors.

    IndexFlatIP gives us exact inner-product search. With normalized vectors,
    that's identical to cosine similarity.
    """
    import faiss  # local import — keep ETL/test paths free of faiss when unused

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
    if len(order_ids) != vectors.shape[0]:
        raise ValueError(
            f"vector count ({vectors.shape[0]}) must match order_ids count ({len(order_ids)})"
        )

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return IndexBundle(index=index, order_ids=list(order_ids), dim=dim)


def persist(bundle: IndexBundle, index_path: str) -> None:
    """Atomically persist index + sidecar map + version file.

    Writes everything to `.tmp`, fsyncs, then renames over the live files.
    On POSIX, rename is atomic — concurrent readers either see the old set
    or the new set, never half of each.
    """
    import faiss

    idx_path = Path(index_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_idx = idx_path.with_suffix(idx_path.suffix + ".tmp")
    tmp_map = Path(map_path(index_path) + ".tmp")
    tmp_ver = Path(version_path(index_path) + ".tmp")

    # Write each artefact to its temp file and fsync.
    faiss.write_index(bundle.index, str(tmp_idx))
    _atomic_write_text(tmp_map, json.dumps(bundle.order_ids))
    _atomic_write_text(tmp_ver, INDEX_VERSION)

    # Atomic rename into place. Last to land is the version file, which is
    # what load() checks first — so a partially-written set can't be loaded.
    tmp_idx.replace(idx_path)
    tmp_map.replace(map_path(index_path))
    tmp_ver.replace(version_path(index_path))


def _atomic_write_text(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


def load(index_path: str) -> IndexBundle | None:
    """Load index + sidecar map + version. Return None if any piece is missing
    or the version doesn't match the current code's INDEX_VERSION."""
    import faiss

    idx = Path(index_path)
    mp = Path(map_path(index_path))
    vp = Path(version_path(index_path))

    if not (idx.is_file() and mp.is_file() and vp.is_file()):
        return None
    if vp.read_text().strip() != INDEX_VERSION:
        return None

    index = faiss.read_index(str(idx))
    order_ids = json.loads(mp.read_text())
    return IndexBundle(index=index, order_ids=order_ids, dim=index.d)


# --------------------------------------------------------------------------
# Cached load — what the API actually calls
# --------------------------------------------------------------------------


_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, IndexBundle]] = {}


def get_or_load_index(index_path: str) -> IndexBundle | None:
    """Return a cached bundle, reloading if the file's mtime has changed.

    The first call after each ETL run pays the load cost (~tens of ms for
    small indexes). Subsequent calls hit the cache and are essentially free.
    """
    with _cache_lock:
        if not Path(index_path).is_file():
            return None
        mtime = os.path.getmtime(index_path)
        cached = _cache.get(index_path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        bundle = load(index_path)
        if bundle is not None:
            _cache[index_path] = (mtime, bundle)
        return bundle


def invalidate_cache(index_path: str | None = None) -> None:
    """Drop the cached bundle. Useful in tests; also called when an admin
    forces a reload."""
    with _cache_lock:
        if index_path is None:
            _cache.clear()
        else:
            _cache.pop(index_path, None)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def search(bundle: IndexBundle, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    """Return [(order_id, score)] sorted by score desc.

    query_vec must be L2-normalized — caller's responsibility.
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

    top_k = min(top_k, bundle.index.ntotal)
    if top_k == 0:
        return []

    scores, indices = bundle.index.search(query_vec, top_k)
    # scores, indices are (1, top_k) — we always search with one query.
    return [
        (bundle.order_ids[int(i)], float(s))
        for s, i in zip(scores[0], indices[0])
        if i != -1
    ]


# --------------------------------------------------------------------------
# Sidecar path helpers
# --------------------------------------------------------------------------


def map_path(index_path: str) -> str:
    return f"{index_path}.map.json"


def version_path(index_path: str) -> str:
    return f"{index_path}.version"


# --------------------------------------------------------------------------
# Rebuild — the hook ETL calls after a successful load
# --------------------------------------------------------------------------


def rebuild_from_db(db_path: str, index_path: str) -> None:
    """Rebuild the FAISS index from the current orders table.

    Reads every row, builds the embedding text via orders.embeddings.order_to_text,
    encodes with the configured sentence-transformers model, persists the
    new index atomically.

    Replaces the no-op stub used through Step 3. After Step 4, every
    successful `etl.py load` rebuilds the index.
    """
    from orders import config, db as db_helpers, embeddings
    from orders.logging_setup import get_logger

    log = get_logger("etl.search")
    settings = config.get_settings()

    # Pull every order in stable order so the index is deterministic for a
    # given DB state.
    with db_helpers.read_only(db_path) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT order_id, customer_id, order_date, amount_usd, currency "
            "FROM orders ORDER BY order_id"
        )]

    if not rows:
        log.info("faiss_rebuild_skipped", reason="no rows in orders table",
                 db_path=db_path, index_path=index_path)
        return

    texts = [embeddings.order_to_text(r) for r in rows]
    order_ids = [r["order_id"] for r in rows]

    log.info("faiss_rebuild_start", count=len(rows), model=settings.embed_model_name)
    vectors = embeddings.encode(settings.embed_model_name, texts)
    bundle = build(vectors, order_ids)
    persist(bundle, index_path)
    invalidate_cache(index_path)
    log.info("faiss_rebuild_done", count=len(rows), dim=bundle.dim, index_path=index_path)
