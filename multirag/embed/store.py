"""On-disk storage for per-page multi-vectors.

Late-interaction encoders emit a matrix per page, not a single vector: one row
per image patch. That matrix is what MaxSim scores against, so it is stored
whole, one .npy per page, mirroring the layout of data/pages.

Per-page files rather than one array per document keep embedding resumable at
page granularity and let Phase 3 stream a corpus into Qdrant without loading a
whole document at once.
"""

import os
from pathlib import Path

import numpy as np

from multirag import config

# Vectors are stored float16. These are L2-normalized values in [-1, 1], where
# fp16 carries ~3 decimal digits: the MaxSim score shift is far below retrieval
# noise, and it halves disk and read time across a corpus.
STORAGE_DTYPE = np.float16


def document_dir(document_id: str) -> Path:
    """Directory holding one document's page vectors."""
    return config.EMBEDDINGS_DIR / document_id


def vector_path(document_id: str, page_index: int) -> Path:
    """Path for one page's vector matrix. Zero-padded to sort in reading order."""
    return document_dir(document_id) / f"page_{page_index:04d}.npy"


def has_vectors(document_id: str, page_index: int) -> bool:
    """Whether this page has already been embedded."""
    return vector_path(document_id, page_index).exists()


def save_vectors(document_id: str, page_index: int, vectors: np.ndarray) -> Path:
    """Write one page's vector matrix atomically. Returns the path written.

    Same temp-then-rename discipline as the manifest writer, for the same
    reason: a half-written .npy that still loads is worse than one that fails
    loudly, because it would be indexed as if it were a complete page.
    """
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2D (patches, dim) matrix, got shape {vectors.shape}")

    path = vector_path(document_id, page_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    with open(temp_path, "wb") as handle:
        np.save(handle, vectors.astype(STORAGE_DTYPE, copy=False))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)

    return path
