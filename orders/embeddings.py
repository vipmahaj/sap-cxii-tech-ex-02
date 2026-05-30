"""Sentence-transformers wrapper.

The model is loaded lazily and cached on first call. The same instance is
shared between ETL (encode all orders at load time) and the API (encode the
free-text query at request time).

Justification for `all-MiniLM-L6-v2` lives in docs/ai-layer.md §2.1.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np


EMBEDDING_TEMPLATE = (
    "order {order_id}: customer {customer_id} "
    "spent ${amount_usd:.2f} USD on {order_date}"
)

# MiniLM-L6-v2 produces 384-dimensional vectors. Hard-coded so any pipeline
# step that needs the dim (e.g. allocating a zero index) doesn't need to load
# the model just to ask.
EMBEDDING_DIM = 384


@lru_cache(maxsize=2)
def get_model(model_name: str):
    """Lazy-load and cache the sentence-transformer model.

    The first call pulls the model from HuggingFace (~80 MB for MiniLM) and
    instantiates a torch module — that's why we cache. lru_cache size = 2 so
    a single process can experiment with two models without thrashing.
    """
    from sentence_transformers import SentenceTransformer  # local import — heavy

    return SentenceTransformer(model_name)


def order_to_text(order: dict[str, object]) -> str:
    """Apply the fixed template from data-contract.md §4."""
    return EMBEDDING_TEMPLATE.format(**order)


def encode(model_name: str, texts: Sequence[str]) -> np.ndarray:
    """Return L2-normalized float32 vectors, shape (n, dim).

    Normalization lets us use faiss.IndexFlatIP and get cosine similarity
    directly from the inner-product score.
    """
    model = get_model(model_name)
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,           # L2-normalize on the GPU/CPU
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # Belt-and-suspenders: ensure float32 (FAISS requirement) and 2-D shape
    # even for a single-text input.
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors
