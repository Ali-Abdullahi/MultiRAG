"""Content-addressed identity for documents and pages.

Identity derives from file bytes, never from filename or path. The same PDF
uploaded twice under different names collapses to one document, and re-running
ingestion overwrites the same records instead of accumulating duplicates.
"""

import hashlib
import uuid
from pathlib import Path

# Frozen namespace for uuid5. Must never change: it is what makes a page's
# Qdrant point id reproducible across machines, runs, and re-indexes.
PAGE_NAMESPACE = uuid.UUID("6f1d4f22-2b3a-5c7e-9a10-8c4b2e6d0f31")

# 16 hex chars = 64 bits of the digest. Collision risk is negligible at
# document-corpus scale, and short ids keep paths and log lines readable.
DOC_ID_LENGTH = 16

# Hash in chunks so a 500MB scanned PDF never lands in memory all at once.
_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the full hex sha256 of a file, read incrementally."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def doc_id(pdf_path: Path) -> str:
    """Stable id for a document, derived only from its bytes."""
    return sha256_file(pdf_path)[:DOC_ID_LENGTH]


def page_id(document_id: str, page_index: int) -> str:
    """Stable id for one page. Zero-padded so ids sort in reading order."""
    return f"{document_id}-p{page_index:04d}"


def page_point_id(page_identifier: str) -> str:
    """Deterministic UUID for a page, used as its Qdrant point id.

    Qdrant accepts only integers or UUIDs as point ids, so the human-readable
    page id is folded into a uuid5. Same page in, same uuid out - which turns
    re-indexing into an overwrite rather than a duplicate insert.
    """
    return str(uuid.uuid5(PAGE_NAMESPACE, page_identifier))
